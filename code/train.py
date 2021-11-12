import torch
import torch.nn as nn
import transformers
import datasets
import time

from textdatasets import tokenize

def train(cfg, model, train_loader, optimizer, tokenizer):
    total_steps = len(train_loader)
    for step, batch in enumerate(train_loader):
        # time
        start_time = time.time()
        # tensor size for reshape
        batch = tokenize(batch, tokenizer, mode = 'train')
        # forward prop
        loss = model(input_ids = batch['input_ids'].cuda(), 
                    attention_mask = batch['attention_mask'].cuda(),
                    decoder_input_ids = batch['decoder_input_ids'].cuda(),
                    decoder_attention_mask = batch['decoder_attention_mask'].cuda(),
                    labels = batch['label'].cuda())
        # optimizer step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        elapsed = time.time() - start_time
        expected = elapsed * (total_steps-step-1)
        h = int(expected//3600)
        m = int((expected%3600) // 60)
        # verbose
        if (step % 300 == 0) or (step == total_steps-1):
            print('[{} / {}] steps | loss : {} | until last step : {}h {}m'.format(step,
                                                                        total_steps,
                                                                        loss.item(),
                                                                        h, m))

def validation(cfg, model, val_dataloader, tokenizer, generation_param, metric):
    '''
    model : pytorch nn.Module
    val_dataloader : nn.DataLoader
    tokenizer : huggingface tokenizer
    generation_params : Dictionary
    metric : rouge score
    '''
    text_preds = []
    text_refs = []
    model.cuda()
    metric = datasets.load_metric('rouge')
    for i, batch in enumerate(val_dataloader):
        batch = tokenize(batch, tokenizer, mode = 'val')
        token_pred = model.generate(
            **generation_param,
            input_ids = batch['input_ids'],
            attention_mask = batch['attention_mask'],
            pad_token_id = tokenizer.pad_token_id,
            eos_token_id = tokenizer.eos_token_id,
            bos_token_id = tokenizer.bos_token_id,
            num_return_sequences = 1,
            early_stopping = True,
            )
        text_pred = tokenizer.batch_decode(
            token_pred,
            skip_special_tokens=True
            )
        text_ref = tokenizer.batch_decode(
            batch['decoder_input_ids'],
            skip_special_tokens=True
            )
        text_preds += text_pred
        text_refs += text_ref
        if i % 100 == 0:
            print('{} texts validated'.format(i*cfg.val.batch_size))
    rouge = metric.compute(predictions=text_preds, references=text_refs)
    rouge_1 = rouge['rouge1'].mid.fmeasure
    rouge_2 = rouge['rouge2'].mid.fmeasure
    rouge_L = rouge['rougeL'].mid.fmeasure
    return rouge_1, rouge_2, rouge_L