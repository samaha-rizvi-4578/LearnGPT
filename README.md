# LearnGPT: A Mini Transformer-Based Educational QA Model

LearnGPT is a compact Transformer-based Question Answering model trained from scratch on the SQuAD v1.1 dataset. It focuses on answering factual questions from given contexts and is designed with extensibility for educational dialogue systems.

---

## 🚀 Features
- Fine-tuning a pre-trained BERT model for Question Answering
- Building a Transformer-based model from scratch
- Comparison of performance between fine-tuned and scratch-built models
- Custom tokenizer and vocabulary
- Pure PyTorch implementation of Encoder-only Transformer
- Trainable QA head for predicting answer spans
- Evaluation using Exact Match and F1 Score
- Inference-ready pipeline

---

## 📚 Dataset
- Stanford Question Answering Dataset (SQuAD) v1.1
- Format: context, question, answer text, start position
- Preprocessing: tokenization, span alignment, padding

---

## 🏗️ Model Architecture
### Fine-Tuned Model
- Pre-trained BERT model (`bert-base-uncased`) fine-tuned for QA
- QA Head: Linear layers for start/end position prediction

### Scratch-Built Model
- Token Embedding + Positional Encoding
- Transformer Encoder (4 layers, 8 heads)
- QA Head: Linear layers for start/end position prediction

---

## 🏋️ Training
### Fine-Tuned Model
- Loss: Average of CrossEntropyLoss on start and end positions
- Optimizer: Adam
- Epochs: 3–5 (configurable)
- Batch Size: 16
- Learning Rate: 2e-5

### Scratch-Built Model
- Loss: Average of CrossEntropyLoss on start and end positions
- Optimizer: Adam
- Epochs: 3–5 (configurable)
- Batch Size: 16
- Learning Rate: 3e-4

---

## 📊 Evaluation Metrics
- Exact Match (EM)
- F1 Score

---

## 📄 Report Link
https://docs.google.com/document/d/11P7eEqbR9M4KSmNSmW4wQr7RUgiOG9MDhzMEdSBeRwg/edit?usp=sharing

---

## 💻 The Coders

- Abdul Rehman Arain (https://github.com/tRzBlizzard)
- Syeda Samaha Batool Rizvi (https://github.com/samaha-rizvi-4578)