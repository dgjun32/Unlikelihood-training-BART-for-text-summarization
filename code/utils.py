from textdatasets import tokenize
import datasets
import os
import numpy as np
import pandas as pd

# function for post processing on model output
def extract_fst_sent(text):
    split = text.split('다.')
    result = split[0]+'다.'
    return result

# end to end inference function
def inference(model, test_loader, tokenizer, generation_params):
    text_preds = []
    ids = datasets.Dataset.from_json(os.path.join(cfg.path.data, cfg.path.test))['id']
    model.cuda()
    model.eval()
    for i,batch in enumerate(test_loader):
        # batch tokenization
        batch = tokenize(batch, tokenizer, cfg, mode = 'test')
        # text generation
        token_pred = model.generate(
            **generation_params,
            input_ids = batch['input_ids'].cuda(),
            attention_mask = batch['attention_mask'].cuda(),
            pad_token_id = tokenizer.pad_token_id,
            eos_token_id = 17546,
            bos_token_id = tokenizer.bos_token_id,
            num_return_sequences = 1,
            early_stopping = True
            )
        # token decoding
        text_pred = tokenizer.batch_decode(
            token_pred,
            skip_special_tokens=True
            )
        text_preds += text_pred
        if (i+1) % 50 == 0:
            print('{} th chunk processed'.format(i+1))
    text_preds = list(map(extract_fst_sent, text_preds))
    # returns dataframe containing id and summarization
    return pd.DataFrame({'id':ids, 'summary':text_preds})

'''
def inference(model, test_loader, tokenizer, generation_params):
    text_preds = []
    ids = datasets.Dataset.from_json(os.path.join(cfg.path.data, cfg.path.test))['id']
    model.cuda()
    model.eval()
    for i,batch in enumerate(test_loader):
        # batch tokenization
        batch = tokenize(batch, tokenizer, mode = 'test')
        # text generation
        token_pred = model.generate(
            **generation_params,
            input_ids = batch['input_ids'].cuda(),
            attention_mask = batch['attention_mask'].cuda(),
            pad_token_id = tokenizer.pad_token_id,
            eos_token_id = tokenizer.eos_token_id,
            bos_token_id = tokenizer.bos_token_id,
            early_stopping = True
            )
        # token decoding
        text_pred = tokenizer.batch_decode(
            token_pred,
            skip_special_tokens=True
            )
        text_preds += text_pred
        if i % 5 == 0:
            print('{} th chunk processed'.format(i))
    # returns dataframe containing id and summarization
    return pd.DataFrame({'id':ids, 'summary_1':text_preds})
'''

def submission(output_df, cfg):
    output_df['id'] = output_df['id'].astype(np.int64)
    sub = pd.read_csv(os.path.join(cfg.path.data, cfg.path.sub))
    sub = sub.merge(output_df, on='id', how='left')
    sub['summary'] = sub['summary_1']
    sub.drop('summary_1', axis=1, inplace = True)
    return sub