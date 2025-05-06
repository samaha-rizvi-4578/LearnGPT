from sklearn.metrics import f1_score, accuracy_score
from utils.eval_utils import evaluate_model

def evaluate_model_on_data(model, dataloader):
    model.eval()
    all_start_preds = []
    all_end_preds = []
    all_start_labels = []
    all_end_labels = []

    with torch.no_grad():
        for input_ids, attention_masks, start_positions, end_positions in dataloader:
            start_logits, end_logits = model(input_ids, attention_masks)
            start_preds = torch.argmax(start_logits, dim=1)
            end_preds = torch.argmax(end_logits, dim=1)

            all_start_preds.extend(start_preds.cpu().numpy())
            all_end_preds.extend(end_preds.cpu().numpy())
            all_start_labels.extend(start_positions.cpu().numpy())
            all_end_labels.extend(end_positions.cpu().numpy())

    em_score = accuracy_score(all_start_preds, all_start_labels)  # Exact Match
    f1 = f1_score(all_end_preds, all_end_labels, average='weighted')  # F1 Score
    return em_score, f1
