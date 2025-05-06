from .transformer import Transformer
import torch.nn as nn

class QAModel(nn.Module):
    def __init__(self):
        super(QAModel, self).__init__()
        self.transformer = Transformer()

    def forward(self, input_ids, attention_mask):
        return self.transformer(input_ids, attention_mask)
