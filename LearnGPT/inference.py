from transformers import BertTokenizer
from models.qa_model import QAModel

def get_answer_from_model(context, question, model, tokenizer):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt", padding='max_length', truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    start_logits, end_logits = model(input_ids, attention_mask)
    start_pred = torch.argmax(start_logits)
    end_pred = torch.argmax(end_logits)

    answer = context[start_pred:end_pred+1]
    return answer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = QAModel().load_state_dict(torch.load('model_weights.pth'))

# Example context and question
context = "The quick brown fox jumps over the lazy dog."
question = "What does the fox do?"

answer = get_answer_from_model(context, question, model, tokenizer)
print(f"Answer: {answer}")
