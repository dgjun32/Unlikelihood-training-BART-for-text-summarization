from config import cfg
from model import SummarizationModel
from train import train, validation
from textdatasets import GasDataLoader
from utils import inference

import os
import torch
import torch.nn as nn
import datasets
import transformers

# model
model = SummarizationModel(cfg)
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