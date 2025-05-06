from transformers import BertTokenizer
import torch

def preprocess_data(dataset, tokenizer, max_len=384):
    input_ids = []
    attention_masks = []
    start_positions = []
    end_positions = []

    for data in dataset:
        context = data['context']
        question = data['question']
        answer = data['answers']
        start_idx = answer['answer_start'][0]
        answer_text = answer['text'][0]

        # Tokenize the context and question
        encoding = tokenizer.encode_plus(
            question, context,
            truncation=True,
            padding='max_length',
            max_length=max_len,
            return_tensors='pt'
        )

        # Store tokenized information
        input_ids.append(encoding['input_ids'].squeeze())
        attention_masks.append(encoding['attention_mask'].squeeze())

        # Compute start and end positions
        answer_start_token_idx = encoding.char_to_token(start_idx)
        answer_end_token_idx = encoding.char_to_token(start_idx + len(answer_text) - 1)
        start_positions.append(answer_start_token_idx)
        end_positions.append(answer_end_token_idx)

    return {
        'input_ids': torch.stack(input_ids),
        'attention_masks': torch.stack(attention_masks),
        'start_positions': torch.tensor(start_positions),
        'end_positions': torch.tensor(end_positions)
    }

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Preprocess the train and validation data
train_dataset = preprocess_data(train_data, tokenizer)
val_dataset = preprocess_data(val_data, tokenizer)
