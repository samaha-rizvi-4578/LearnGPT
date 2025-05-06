import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW
from models.qa_model import QAModel
from utils.train_utils import train_epoch

# Hyperparameters
batch_size = 16
epochs = 3
learning_rate = 3e-5

# DataLoader
train_dataset = TensorDataset(train_data['input_ids'], train_data['attention_masks'], train_data['start_positions'], train_data['end_positions'])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(val_data['input_ids'], val_data['attention_masks'], val_data['start_positions'], val_data['end_positions'])
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Model, optimizer, and loss function
model = QAModel().to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    train_epoch(model, train_dataloader, optimizer, epoch)
    # Validation step can be added here
