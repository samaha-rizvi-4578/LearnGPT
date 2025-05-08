# LearnGPT: A Mini Transformer-Based Educational QA Model

LearnGPT is a compact Transformer-based Question Answering model trained from scratch on the SQuAD v1.1 dataset. It focuses on answering factual questions from given contexts and is designed with extensibility for educational dialogue systems.

---

## 🚀 Features
- Fine-tuning a pre-trained BERT model for Question Answering
- Building a Transformer-based model from scratch
- Comparison of performance between fine-tuned and scratch-built models
- Custom tokenizer and vocabulary for the scratch-built model
- Advanced tokenization using `BertTokenizerFast` for the BERT-based model
- Pure PyTorch implementation of Encoder-only Transformer for the scratch-built model
- Trainable QA head for predicting answer spans
- Evaluation using Exact Match (EM) and F1 Score
- Inference-ready pipeline for both models

---

## 📚 Dataset
- Stanford Question Answering Dataset (SQuAD) v1.1
- Format: context, question, answer text, start position
- Preprocessing:
  - Scratch-built model: Custom tokenization, vocabulary building, and span alignment
  - BERT-based model: Tokenization using `BertTokenizerFast` with offset mapping for precise span alignment

---

## 🏗️ Model Architecture
### Fine-Tuned Model (BERT-Based)
- Pre-trained BERT model (`bert-base-uncased`) fine-tuned for QA
- QA Head: Linear layers for start/end position prediction
- Tokenization: Advanced tokenization using `BertTokenizerFast`
- Optimizer: Adam with a learning rate of `2e-5`
- Early stopping with patience of 3 epochs

### Scratch-Built Model
- Token Embedding + Positional Encoding
- Transformer Encoder (4 layers, 8 heads)
- QA Head: Linear layers for start/end position prediction
- Tokenization: Custom tokenizer with vocabulary built from training data
- Optimizer: Adam with a learning rate of `3e-4`

---

## 🏋️ Training
### Fine-Tuned Model (BERT-Based)
- Loss: Average of CrossEntropyLoss on start and end positions
- Optimizer: Adam
- Epochs: Configurable (default: 10)
- Batch Size: 8
- Learning Rate: 2e-5
- Early stopping with patience of 3 epochs
- Validation after each epoch with Exact Match (EM) and F1 Score evaluation

### Scratch-Built Model
- Loss: Average of CrossEntropyLoss on start and end positions
- Optimizer: Adam
- Epochs: Configurable (default: 3–5)
- Batch Size: 16
- Learning Rate: 3e-4
- Validation after each epoch with Exact Match (EM) and F1 Score evaluation

---

## 📊 Evaluation Metrics
- **Exact Match (EM):** Measures the percentage of predictions that match the ground truth exactly.
- **F1 Score:** Measures the overlap between the predicted and ground truth answers, considering both precision and recall.

---

## 🆚 Comparison of Models
| Feature                  | Scratch-Built Model                     | Fine-Tuned BERT Model                |
|--------------------------|------------------------------------------|---------------------------------------|
| **Architecture**         | Custom Transformer Encoder              | Pre-trained BERT (`bert-base-uncased`) |
| **Tokenization**         | Custom tokenizer and vocabulary         | `BertTokenizerFast` with offset mapping |
| **Training Speed**       | Slower due to training from scratch      | Faster due to pre-trained weights     |
| **Performance**          | Moderate EM and F1 scores               | Higher EM and F1 scores               |
| **Extensibility**        | Fully customizable                      | Limited to BERT architecture          |
| **Batch Size**           | 16                                      | 8                                     |
| **Learning Rate**        | 3e-4                                    | 2e-5                                  |

---

## 📄 Report Link
[Project Report](https://docs.google.com/document/d/11P7eEqbR9M4KSmNSmW4wQr7RUgiOG9MDhzMEdSBeRwg/edit?usp=sharing)

---

## 💻 The Coders

- Abdul Rehman Arain (https://github.com/tRzBlizzard)
- Syeda Samaha Batool Rizvi (https://github.com/samaha-rizvi-4578)