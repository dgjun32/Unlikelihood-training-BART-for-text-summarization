import torch
import transformers
import argparse

from config import cfg
from textdatasets import GasDataLoader, tokenize 
from utils import inference


if __name__ == '__main__':
    
    # parsing
    parser = argparse.ArgumentParser(description='Summarize Text')
    parser.add_argument('--batch_size', dest='batch_size', default=16, type=int)
    parser.add_argument('--max_length', dest='max_length', default=128, type=int)
    parser.add_argument('--num_beams', dest='num_beams', default=5, type=int)
    parser.add_argument('--n_ngram', dest='n_ngram', default=3, type=int)
    args = parser.parse_args()
    # dataloader
    testdata = GasDataLoader(cfg, mode = 'test')
    test_loader = testdata._get_test_loader(batch_size=args.batch_size)
    # model & tokenizer
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(cfg.model.name)
    model = transformers.BartForConditionalGeneration.from_pretrained(cfg.model.name)
    weight_dir = os.path.join(cfg.path.model, 'ul_seq_koBART_step_60000.pth')
    model.load_state_dict(torch.load(weight_dir, map_location='cpu'))
    # inference
    best_param = {'max_length':args.max_length, 'num_beams':args.num_beams, 'n_ngram':args.n_ngram, do_sample:'False'}
    output = inference(model, test_loader, tokenizer, best_param)
    # save output
    output.to_csv(os.path.join(cfg.path.submission, 'summarization.csv'))

    