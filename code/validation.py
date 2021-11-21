import torch
import torch.nn as nn
import transformers
import datasets
import time

from textdatasets import tokenize

def validate(cfg, model, val_dataloader, tokenizer, generation_param, metric):
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