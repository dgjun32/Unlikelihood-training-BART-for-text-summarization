import torch
import torch.nn as nn
import pandas as pd
import transformers
import datasets
import itertools
import os
import gc

from transformers import PreTrainedTokenizerFast
from torch.utils.data import Dataset, DataLoader
from itertools import chain
class GasDataLoader:
    def __init__(self, cfg, mode = 'test'):
        self.cfg = cfg
        self.mode = mode
        self.json_to_dataset()
    
    def json_to_dataset(self):
        '''function for loading json and preprocessing'''
        # load train, val data
        if self.mode != 'test':
            train_news = datasets.Dataset.from_json(os.path.join(self.cfg.path.data, self.cfg.path.train_news), field = 'documents').remove_columns(self.cfg.dataset.news_useless_feats)
            train_law = datasets.Dataset.from_json(os.path.join(self.cfg.path.data, self.cfg.path.train_law), field = 'documents').remove_columns(self.cfg.dataset.law_useless_feats)
            val_news = datasets.Dataset.from_json(os.path.join(self.cfg.path.data, self.cfg.path.val_news), field = 'documents').remove_columns(self.cfg.dataset.news_useless_feats)
            dataset = datasets.concatenate_datasets([train_news, train_law, val_news]).train_test_split(0.01)
            self.trainset = dataset['train']
            self.valset = dataset['test']
            del train_news, train_law, val_news, dataset
        # load test data
        else:
            self.testset = datasets.Dataset.from_json(os.path.join(self.cfg.path.data, self.cfg.path.test))
        gc.collect()

        # preprocessing on texts
        if self.mode != 'test':
            self.trainset = self.trainset.map(
                lambda x:{'abstractive':'\n'.join(x['abstractive']),
                            'text':'\n'.join(list(chain(*[[j['sentence'] for j in i] for i in x['text']])))}
                )
            self.valset = self.valset.map(lambda x:{'abstractive':'\n'.join(x['abstractive']),
                            'text':'\n'.join(list(chain(*[[j['sentence'] for j in i] for i in x['text']])))}
                )
        else:
            self.testset = self.testset.map(
                lambda x:{'text':'\n'.join(x['article_original'])}
                )
            self.testset = self.testset.remove_columns(self.cfg.dataset.test_useless_feats)

    def _get_train_loader(self, batch_size):
        train_sampler = RandomSampler(self.trainset)
        return DataLoader(self.trainset, 
                        sampler = train_sampler,
                        batch_size = batch_size)
    
    def _get_val_loader(self, batch_size):
        return DataLoader(self.valset,
                          batch_size = batch_size)
    
    def _get_test_loader(self, batch_size):
        return DataLoader(self.testset,
                          batch_size = batch_size)

def tokenize(batch, tokenizer, mode = 'train'):
    if mode == 'train':
        texts, abstractives = batch['text'], batch['abstractive']
        input_token = tokenizer.batch_encode_plus(texts, max_length=1024, padding = 'max_length', return_tensors='pt')
        output_token = tokenizer.batch_encode_plus(abstractives, max_length=512, padding = 'max_length', return_tensors='pt')
        return {'input_ids': input_token['input_ids'],
                'attention_mask':input_token['attention_mask'],
                'decoder_input_ids':output_token['input_ids'],
                'decoder_attention_mask':output_token['attention_mask'],
                'labels':output_token['input_ids']}
    elif mode == 'val':
        texts, abstractives = batch['text'], batch['abstractive']
        input_token = tokenizer.batch_encode_plus(texts, max_length=1024, padding = 'max_length', return_tensors='pt')
        output_token = tokenizer.batch_encode_plus(abstractives,  max_length=512, padding = 'max_length', return_tensors='pt')
        return {'input_ids':input_token['input_ids'],
                'attention_mask':input_token['attention_mask'],
                'labels':output_token['input_ids']}
    else:
        texts = batch['text']
        input_token = tokenizer.batch_encode_plus(texts, max_length=1024, padding='max_length', return_tensors='pt')
        return {'input_ids':input_token['input_ids'],
                'attention_mask':input_token['attention_mask']}