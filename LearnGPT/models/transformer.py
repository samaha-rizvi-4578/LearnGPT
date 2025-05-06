import torch
import torch.nn as nn
from transformers import BertModel

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc_start = nn.Linear(self.bert.config.hidden_size, 1)
        self.fc_end = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0]  # Get the hidden states

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Pass through the start and end prediction layers
        start_logits = self.fc_start(pooled_output).squeeze(-1)
        end_logits = self.fc_end(pooled_output).squeeze(-1)

        return start_logits, end_logits
