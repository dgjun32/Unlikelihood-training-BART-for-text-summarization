from config import cfg
from model import Model
from train import train, validation
from textdatasets import GasDataLoader
from utils import inference, submission

import os
import torch
import torch.nn as nn
import datasets
import transformers
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader, RandomSampler
import torch.nn.functional as F

from torch.utils.data import TensorDataset, SequentialSampler, DataLoader, RandomSampler

from collections import defaultdict
from tqdm import tqdm, trange
from pprint import pprint

import fairseq
from fairseq.custom.metrics import TrainingMetrics, Metrics, ngram_metrics
from fairseq.custom.baseline_cross_entropy import CrossEntropyCriterionWCustomMetrics
from fairseq.custom.sequence_penalty_loss import SequencePenaltyCriterion
from fairseq.custom.evaluate_utils import batch_input_sequence_by_prefix_length

from collections import defaultdict
from tqdm import tqdm, trange
from pprint import pprint

import json
import os
import re
import random

####
from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, WarmupLinearSchedule, WEIGHTS_NAME, CONFIG_NAME
####

# model
model = transformers.BartForConditionalGeneration.from_pretrained(cfg.model.name)
model.cuda()
for param in model.parameters():
    param.requires_grad = True

# dataloader
train_val_data = GasDataLoader(cfg, mode = 'train')
train_loader = train_val_data._get_train_loader()
val_loader = train_val_data._get_val_loader()

# utils
tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(cfg.model.name)
metric = datasets.load_metric('rouge')

# optimizer & lr_scheduler
optimizer = eval(cfg.train.optimizer)(model.parameters(), lr = 2e-5)
lr_scheduler = eval(cfg.train.lr_schedule)(optimizer = optimizer, T_max = cfg.train.max_epochs, eta_min = 1e-7)

start_epoch = 0

for epoch in range(start_epoch, cfg.train.max_epochs):
    print('{}th epoch'.format(epoch+1))
    print('-'*50)
    # train epoch
    model.train()
    train(cfg, model, train_loader, optimizer, tokenizer)
    # validation 
    with torch.no_grad():
        r1, r2, rL = validation(model, val_loader, tokenizer, cfg.gen, metric)
    print('| val_rouge_1 : {} | val_rouge_2 : {} | val_rouge_L : {} |'.format(r1, r2, rL))
    print('='*50)
    torch.save(model.state_dict(), os.path.join(cfg.path.output, 'bart_finetune_{}.pth'.format(epoch)))

###################################### ul_seq Finetuning####################################################
parser = argparse.ArgumentParser(description='openGPT-2 analysis')

parser.add_argument('--mode', choices=['train', 'eval-singletoken', 'eval-completion', 'eval-both'], default='eval-singletoken')
parser.add_argument('--eval-split', choices=['train', 'valid', 'test'])
parser.add_argument('--model-name', choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2-medium')
parser.add_argument('--model-load-dir', type=str, default=None)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--data-base', type=str)
parser.add_argument('--num-train-epochs', type=int, default=1)
parser.add_argument('--batch-size-singletoken', type=int, default=1024)
parser.add_argument('--batch-size-completion', type=int, default=300)
parser.add_argument("--output-dir", default=None, type=str, required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")

# eval-completion
parser.add_argument('--prefix-length', type=int, default=50)
parser.add_argument('--continuation-length', type=int, default=100)
parser.add_argument('--top-k', type=int, default=1)
parser.add_argument('--top-p', type=float, default=0.0)

# custom training
parser.add_argument('--sequence-tune-rate', type=float, default=0.5)
parser.add_argument('--train-batch-size', type=int, default=300)
parser.add_argument('--report-metrics-every', type=int, default=10)
parser.add_argument('--save-every', type=int, default=1000)
parser.add_argument('--sequence-ngram-n', type=int, default=4)
parser.add_argument('--train-n-steps', type=int, default=10000)
parser.add_argument('--validate-every', type=int, default=10000)

# training loop
parser.add_argument("--adam-epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument('--max-grad-norm', type=int, default=1)
parser.add_argument("--max-steps", default=-1, type=int,
                    help="If > 0: set total number of training \
                        steps to perform. Override num_train_epochs.")
parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                    help="Number of updates steps to accumulate before\
                        performing a backward/update pass.")
parser.add_argument('--learning-rate', type=float, default=6.25e-5)
parser.add_argument("--warmup-steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")
parser.add_argument('--lr-schedule', type=str, default='warmup_linear')
parser.add_argument('--weight-decay', type=float, default=0.01)
parser.add_argument('--lm-coef', type=float, default=0.9)

args = parser.parse_args()
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.size(0) == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    logits = logits.squeeze(0)
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

# function for generating sequence given prefix(article) - used in ul_seq
def sample_sequence(model,
                    prefix_batch,
                    prefix_length=cfg.gen.input_length,
                    continuation_length=cfg.gen.max_length,
                    top_k=cfg.gen.top_k,
                    top_p=cfg.gen.top_p):
    '''
    prefix_batch : dictionary containing keys ['input_ids','attention_mask']
    '''
    assert prefix_batch['input_ids'].size(1) == prefix_length
    # generate continuation
    contiunation = model.generate(input_ids = prefix_batch['input_ids'],
                                  attention_mask = prefix_batch['attention_mask'],
                                  max_length = continuation_length,
                                  top_k = top_k,
                                  top_p = top_p,
                                  output_scores = False,
                                  return_dict_in_generate = True)
    return continuation['sequences'], continuation['hidden_states'] 

# function for computing token level MLE loss
def mle_loss(model, batch):
    model_output = model(input_ids=batch['input_ids'].cuda(),
                        attention_mask=batch['attention_mask'].cuda(),
                        decoder_attention_mask=batch['decoder_attention_mask'].cuda(),
                        labels=batch['decoder_input_ids'].cuda())
    mleloss = model_output['loss']
    return mleloss


def ul_seq(model, batch, cfg):
    '''
    model : transformers.BartForConditionalGeneration
    batch : Dict
    '''
    pred_toks, continuation_logits = sample_sequence(model, batch,
                                                    cfg.gen.input_length,
                                                    cfg.gen.max_length,
                                                    cfg.gen.top_k,
                                                    cfg.gen.top_p)
                                       
    mask = ngram_repeat_mask(pred_toks, cfg.ul_train.ngram_n).type_as(continuation_logits)

    lprobs = F.log_softmax(continuation_logits, dim=-1)
    pred_lprobs = lprobs.view(-1, lprobs.size(2)).gather(1, pred_toks.view(-1, 1))
    one_minus_probs = torch.clamp((1.0 - pred_lprobs.exp()), min=1e-20).view(pred_toks.size(0), pred_toks.size(1))
    loss = -torch.log(one_minus_probs) * mask
    loss = loss.sum()
    ntokens = pred_toks.numel()  # number of output tokens (tokens in completions)

    logging_output = {
        'seq_loss': loss.item(),
        'seq_sample_size': ntokens,
        'seq_ntokens': ntokens,
        'seq_nsentences': batch.size(0),
        'seq_repeat_mask': mask.sum().item(),
    }

    # Sum each statistic, which will be normalized by the number of sentences in `aggregate_logging_outputs`.
    stats = defaultdict(float)
    for tok_list in pred_toks.cpu().tolist():
        ms = ngram_metrics(tok_list)
        for k, v in ms.items():
            stats[k] += v
    for k, v in stats.items():
        logging_output[k] = v

    loss = loss / ntokens
    return loss, logging_output


def ngram_repeat_mask(xs, n):
    mask = torch.zeros_like(xs)
    for i, x in enumerate(xs):
        seen = set()
        xl = x.tolist()
        for j in range(len(x)-n):
            ng = tuple(xl[j:j+n])
            if ng in seen:
                mask[i, j:j+n] = 1
            seen.add(ng)
    return mask

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {}, n_gpu {}".format(device, n_gpu))

# set output dir
if not os.path.exists(cfg.path.output):
    os.makedirs(cfg.path.output)

# load tokenizer
tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(cfg.model.name)

# load pretrained model
model = tranformers.BartForConditionalGeneration.from_pretrained(cfg.model.name)
model.to(device)

'''
if args.mode == 'eval-singletoken' or args.mode == 'eval-both':
    eval_singletoken(model, args, dataset_paths)

# evaluation
if args.mode == 'eval-completion' or args.mode == 'eval-both':
    datasets = get_datasets(dataset_paths, max_len=args.batch_size_completion)
    eval_sampler = SequentialSampler(datasets[args.eval_split])
    eval_dataloader = DataLoader(datasets[args.eval_split], sampler=eval_sampler, batch_size=1)

    model.eval()

    with torch.no_grad():
       all_text_completions = []

        bpe_ngram_metrics = Metrics(pad=-1)
        word_ngram_metrics = Metrics(pad=-1)

        for i, batch in tqdm(enumerate(eval_dataloader), desc="Evaluating", total=len(eval_dataloader)):
            input_sequence = batch[0].cuda()
            if input_sequence.size(1) < args.prefix_length:
                continue

            # Predict the completions.
            batch = batch_input_sequence_by_prefix_length(input_sequence, args.prefix_length)
            bpe_completions, _ = sample_sequence(model, batch, args.prefix_length, args.continuation_length, args.top_k, args.top_p)
            bpe_completions = bpe_completions.tolist()

            # Extract continuations from the predicted completions.
            bpe_continuations = []
            text_continuations = []
            for bpe_completion in bpe_completions:
                bpe_continuations.append(bpe_completion[args.prefix_length:])
                text_continuations.append(get_text_continuation(bpe_completion, tokenizer, args))
                all_text_completions.append(tokenizer.decode(bpe_completion))

            # Only keep continuations with at least one 4-gram
            # (A short continuation may occur due to predicted whitespace, then tokenizing, despite being
            #  normal length in BPE tokens).
            text_continuations = [c for c in text_continuations if len(c) > 3]

            # Update metrics with this batch of continuations.
            bpe_ngram_metrics.update(bpe_continuations)
            word_ngram_metrics.update(text_continuations)

            # Save the (possibly intermediate) metrics.
            save_completion_metrics(bpe_metrics=bpe_ngram_metrics.report('bpe_%s' % args.eval_split),
                                    word_metrics=word_ngram_metrics.report('word_%s' % args.eval_split),
                                    text_completions=all_text_completions,
                                    config=model.config.to_dict(),
                                    args=args)
'''
# model training
if args.mode == 'train':
    if not os.path.exists(os.path.join(cfg.path.output, 'best')):
        os.makedirs(os.path.join(cfg.path.output, 'best'))

    token_loss = mle_loss
    # load dataset
    train_val_data = GasDataLoader(cfg, mode = 'train')
    train_loader = train_val_data._get_train_loader()
    val_loader = train_val_data._get_val_loader()

    # Setup optimizer
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_seq_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_seq_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # start training
    total_steps = 0
    best_ppl = 1e20
    for _ in trange(args.num_train_epochs, desc="Epoch"):
        logging_outputs = []
        epoch_loss = 0
        epoch_steps = 0
        tqdm_bar = tqdm(train_loader, desc="Training", total=cfg.train.n_steps)
        for step, batch in enumerate(tqdm_bar):
            optimizer.zero_grad()
            batch = tokenize(batch, tokenizer, mode = 'train')
            # randomly apply seq level ul_loss or token level mle_loss
            if torch.rand(1).item() < args.sequence_tune_rate:
                loss, batch_metrics = ul_seq(model, batch, args)

            # Token loss
            else:
                loss, batch_metrics = token_loss(model, batch, args)

            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            epoch_steps += 1
            total_steps += 1
            tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(epoch_loss/epoch_steps, scheduler.get_lr()[0])

            logging_outputs.append(batch_metrics)

            if epoch_steps % args.report_metrics_every == 0:
                logging_average = CrossEntropyCriterionWCustomMetrics.aggregate_logging_outputs(logging_outputs)
                temp = SequencePenaltyCriterion.aggregate_logging_outputs(logging_outputs)
                for k, v in temp.items():
                    logging_average[k] = v
                logging_average['ppl'] = 2 ** logging_average['loss']
                print(logging_average)
                logging_outputs = []

            if step == args.train_n_steps:
                break

            # save checkpoint
            if epoch_steps % args.save_every == 0:
                model_to_save = model.module if hasattr(model, 'module') else model
                output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(args.output_dir)

            # validation
            if total_steps % args.validate_every == 0:
                print("Validating...")
                validation_outputs = eval_singletoken(model, args, dataset_paths, train_iter=total_steps)
                if validation_outputs['ppl'] < best_ppl:
                    best_ppl = validation_outputs['ppl']
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(args.output_dir, 'best', WEIGHTS_NAME)
                    output_config_file = os.path.join(args.output_dir, 'best', CONFIG_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    tokenizer.save_vocabulary(os.path.join(args.output_dir, 'best'))
                    save_singletoken_metrics(validation_outputs, model.config.to_dict(), args,
                                                train_iter=total_steps, best=True)
