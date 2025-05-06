# LearnGPT: A Mini Transformer-Based Educational QA Model

LearnGPT is a compact Transformer-based Question Answering model trained from scratch on the SQuAD v1.1 dataset. It focuses on answering factual questions from given contexts and is designed with extensibility for educational dialogue systems.

## 🚀 Features
- Custom tokenizer and vocabulary
- Pure PyTorch implementation of Encoder-only Transformer
- Trainable QA head for predicting answer spans
- Evaluation using Exact Match and F1 Score
- Inference-ready pipeline
- Educational assistant logic for extension

---

## 📚 Dataset
- Stanford Question Answering Dataset (SQuAD) v1.1
- Format: context, question, answer text, start position
- Preprocessing: tokenization, span alignment, padding

---

## 🏗️ Model Architecture
- Token Embedding + Positional Encoding
- Transformer Encoder (4 layers, 8 heads)
- QA Head: Linear layers for start/end position prediction

---

## 🏋️ Training
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

## 💡 Inference Usage

```python
answer = model.answer_question(context="...", question="...")
print("Answer:", answer)
```
---

## 📦 Setup & Usage
```
git clone https://github.com/yourusername/LearnGPT
cd LearnGPT
pip install -r requirements.txt
python train.py
python evaluate.py
```
---

## 📝 Project Structure
LearnGPT/
├── data/
├── models/
│   ├── transformer.py
│   └── qa_model.py
├── utils/
├── train.py
├── evaluate.py
├── inference.py
└── README.md

---

## 🔬 Results
| Metric   | Score |
| -------- | ----- |
| EM       | 71.2% |
| F1 Score | 82.6% |

---

## 📖 References
SQuAD v1.1: https://rajpurkar.github.io/SQuAD-explorer/
Vaswani et al., 2017: Attention Is All You Need
HuggingFace Datasets and Tokenizers

---

##👩‍💻 Author
Abdul Rehman Arain
Syeda Samaha Batool Rizvi
Final Year CS Student, FAST NUCES Karachi


