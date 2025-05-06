from datasets import load_dataset

def load_squad_data():
    dataset = load_dataset("squad")
    return dataset

# Load training and validation data
train_data, val_data = load_squad_data()['train'], load_squad_data()['validation']

# Inspect a sample
print(train_data[0])  # Print the first sample in the training set
print(val_data[0])    # Print the first sample in the validation set