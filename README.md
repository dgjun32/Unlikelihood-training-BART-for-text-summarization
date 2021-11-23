# Sequence-level unlikelihood fine-tuning KoBART

## 1. Backgroud
Current Korean text abstractive generation models suffer from ```text degeneration```, which includes text repetition issue.

Applying advanced <b>stochastic decoding strategies</b> such as ```top-k sampling``` and ```nucleaus sampling``` can mitigate text repetition issue with open-ended language generation tasks. 

Recent research([Cho et al.2019](https://github.com/facebookresearch/unlikelihood_training)) suggests unlikelihood training to resolve text degeneration issue.

I applied unlikelihood training scheme to finetune pretrained ```KoBART```(Korean BART).

## 2. Data Preparation

1. Set directory : ```$cd data/```
2. Download ```.json```file to current dir from the <b>AI hub</b> [link](https://aihub.or.kr/aidata/8054)

## 3. Training

1. Set directory : ```$cd code/```
2. Training + Validation : ```$python main.py```

## 4. Demo

## 5. Result