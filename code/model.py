import torch
import torch.nn as nn
import transformers

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = transformers.BartForConditionalGeneration.from_pretrained(cfg.model.name)
        self.cfg = cfg
    def forward(self,
                input_ids,
                attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
                labels):
        output = self.backbone(input_ids = input_ids,
                               attention_mask = attention_mask,
                               decoder_input_ids = decoder_input_ids,
                               decoder_attention_mask = decoder_attention_mask,
                               labels = labels)
        return output['loss']