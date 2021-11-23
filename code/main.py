from config import cfg
from validation import validate
from textdatasets import GasDataLoader, tokenize
from utils import inference, submission

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import datasets
import transformers
from transformers import LogitsProcessorList, MinLengthLogitsProcessor
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader, RandomSampler
import torch.nn.functional as F

from collections import defaultdict
from tqdm import tqdm, trange
from pprint import pprint

import fairseq
from fairseq.custom.metrics import TrainingMetrics, Metrics, ngram_metrics
from fairseq.custom.baseline_cross_entropy import CrossEntropyCriterionWCustomMetrics
from fairseq.custom.sequence_penalty_loss import SequencePenaltyCriterion
from fairseq.custom.evaluate_utils import batch_input_sequence_by_prefix_length

import json
import os
import re
import random


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

print('Succesfully installed libraries')

# function for generating sequence given prefix(article) - used in ul_seq
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.size(0) == cfg.train.batch_size  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        for i in range(logits.shape[0]):
            for j in range(logits.shape[1]):
                if indices_to_remove[i,j]==True:
                    logits[i,j]=filter_value
                    
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


def sample_sequence_1(model, prefix_batch, prefix_length=cfg.gen.input_length,
                    continuation_length=cfg.gen.max_length):
    context = prefix_batch
    assert context['input_ids'].size(1) == prefix_length
    with torch.enable_grad():
        encoder_outputs = model(context['input_ids'], context['attention_mask'], output_hidden_states=True, output_attentions=True)
        enc_outputs =  (encoder_outputs['encoder_last_hidden_state'],
                        encoder_outputs['encoder_hidden_states'],
                        encoder_outputs['encoder_attentions'])
        out = model.greedy_search(input_ids = context['input_ids'],
                                attention_mask = context['attention_mask'],
                                pad_token_id = tokenizer.pad_token_id,
                                eos_token_id = tokenizer.eos_token_id,
                                bos_token_id = tokenizer.bos_token_id,
                                max_length = 50,
                                num_return_sequences = 1,
                                output_scores = True,
                                return_dict_in_generate = True,
                                encoder_outputs = enc_outputs)
        continuation_scores, output = torch.stack(out['scores'], dim=1), out['sequences']

    return output, continuation_scores

def sample_sequence(model, prefix_batch, prefix_length=cfg.gen.input_length,
                    continuation_length=cfg.gen.max_length,
                    top_k=cfg.gen.top_k,
                    top_p=cfg.gen.top_p):
    context = prefix_batch
    assert context['input_ids'].size(1) == prefix_length
    prev = context['input_ids']
    past = None
    continuation_logits = []
    output = []
    for i in range(continuation_length):
        out = model(input_ids = prev.cuda(), past_key_values=past, use_cache=True)
        logits, past = out['logits'], out['past_key_values']
        logits = logits[:, -1, :]

        if top_k == 1 and top_p == 0:
            # greedy algo
            prev = logits.argmax(dim=1, keepdim=True)
        else:
            # random sampling
            filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            prev = F.softmax(filtered_logits, dim=1).multinomial(num_samples=1)
        
        continuation_logits.append(logits)
        output.append(prev)

    continuation_scores = F.softmax(torch.stack(continuation_logits, 1), dim = 2)
    output = torch.stack(output, 1).squeeze(2)
    return output, continuation_scores

# function for computing token level MLE loss
def mle_loss(model, batch):
    '''
    Arguments
        model : transformers.BartForConditionalGeneration.pretrained_from(cfg.model.name)
        batch : dict(torch.FloatTensor)
    Returns:
        loss : torch.FloatTensor shape of (1,)
    '''
    model_output = model(input_ids=batch['input_ids'].cuda(),
                        attention_mask=batch['attention_mask'].cuda(),
                        decoder_attention_mask=batch['decoder_attention_mask'].cuda(),
                        labels=batch['decoder_input_ids'].cuda())
    mleloss = model_output['loss']
    return mleloss


def ul_seq(model, batch, cfg):
    '''
    Arguments
        model : transformers.BartForConditionalGeneration.pretrained_from(cfg.model.name)
        batch : dict(torch.FloatTensor)
    Returns:
        loss : torch.FloatTensor shape of (1,)
    '''
    pred_toks, continuation_scores = sample_sequence(model, batch,
                                                    cfg.gen.input_length,
                                                    cfg.gen.max_length)
    mask = ngram_repeat_mask(pred_toks, cfg.ul_train.ngram_n).type_as(continuation_scores)
    pred_lprobs = continuation_scores.reshape(-1, continuation_scores.size(2)).gather(1, pred_toks.reshape(-1, 1))
    one_minus_probs = torch.clamp((1.0 - pred_lprobs), min=1e-20).reshape(pred_toks.size(0), pred_toks.size(1))
    loss = -torch.log(one_minus_probs) * mask
    loss = loss.sum()
    ntokens = pred_toks.numel()  # number of output tokens (tokens in continuation)
    loss = loss / ntokens
    return loss


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




# execute model training
if __name__ == '__main__':
    # set random seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    print('Set Random Seed')
    
    # set device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('GPU device successfully allocated')

    # set output dir
    if not os.path.exists(cfg.path.output):
        os.makedirs(cfg.path.output)

    # load tokenizer & evaluation metric
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(cfg.model.name)
    metric = datasets.load_metric('rouge')

    # load pretrained model
    model = transformers.BartForConditionalGeneration.from_pretrained(cfg.model.name)
    model.cuda()
    print('Loaded pretrained model and tokenizer')

    token_loss = mle_loss
    
    # load dataset
    train_val_data = GasDataLoader(cfg, mode = 'train')
    train_loader = train_val_data._get_train_loader(cfg.train.batch_size)
    val_loader = train_val_data._get_val_loader(cfg.val.batch_size)
    print('Loaded dataset and loader')

    # Setup optimizer
    if cfg.train.max_steps > 0:
        t_total = cfg.train.max_steps
        cfg.train.max_epochs = cfg.train.max_steps//(len(train_loader) // cfg.train.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_loader) // cfg.train.gradient_accumulation_steps * cfg.train.max_epochs

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': cfg.train.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = eval(cfg.train.optimizer)(optimizer_grouped_parameters,
                                        lr=cfg.train.learning_rate, 
                                        eps=cfg.train.adam_eps)
    scheduler = eval(cfg.train.lr_schedule)(optimizer,
                                            T_0=cfg.train.warmup_steps, 
                                            T_mult=cfg.train.T_mult, 
                                            eta_min = cfg.train.adam_eps)

    print('Training......')
    # start training
    total_steps = 0
    for _ in trange(cfg.train.max_epochs, desc="Epoch"):
        epoch_loss = 0
        epoch_steps = 0
        tqdm_bar = tqdm(train_loader, desc="Training", total=cfg.train.n_steps)
        for step, batch in enumerate(tqdm_bar):
            optimizer.zero_grad()
            batch = tokenize(batch, tokenizer, mode = 'train')
            # randomly apply seq level ul_loss or token level mle_loss
            if torch.rand(1).item() < cfg.ul_train.p:
                loss = ul_seq(model, batch, cfg)

            # Token loss
            else:
                loss = token_loss(model, batch)

            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            epoch_steps += 1 # steps in epoch 
            total_steps += 1 # total steps
            tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(epoch_loss/epoch_steps, scheduler.get_lr()[0])
            '''
            if epoch_steps % cfg.val.report_metrics_every == 0:
                logging_average = CrossEntropyCriterionWCustomMetrics.aggregate_logging_outputs(logging_outputs)
                temp = SequencePenaltyCriterion.aggregate_logging_outputs(logging_outputs)
                for k, v in temp.items():
                    logging_average[k] = v
                logging_average['ppl'] = 2 ** logging_average['loss']
                print(logging_average)
                logging_outputs = []
            '''

            if step == cfg.train.n_steps:
                break

            # save checkpoint & validate
            if total_steps % cfg.val.validate_every == 0:
                # model validation (compute Rouge1, Rouge2, RougeL score)
                print("Validating...")
                with torch.no_grad():
                    r1, r2, rL = validate(model, val_loader, tokenizer, cfg.gen, metric)
                print('| val_rouge_1 : {} | val_rouge_2 : {} | val_rouge_L : {} |'.format(r1, r2, rL))
                print('='*50)
                # save checkpoint
                output_model_file = os.path.join(cfg.path.output, 'ul_seq_koBART_step_{}.pth'.format(total_steps))
                torch.save(model.state_dict(), output_model_file)
