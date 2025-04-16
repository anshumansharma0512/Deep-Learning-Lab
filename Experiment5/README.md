# Sequence-to-Sequence Neural Machine Translation with Attention (PyTorch)

This project implements a **Sequence-to-Sequence (Seq2Seq)** model using **LSTM** networks in PyTorch to translate English to Spanish. It supports multiple **attention mechanisms** and includes BLEU score evaluation and attention visualization.

## Overview

The goal is to build a neural machine translation (NMT) model from scratch that can:

- Learn from English-Spanish sentence pairs (`spa.txt`)
- Translate English sentences into Spanish
- Visualize attention weights between input and output tokens
- Evaluate translation quality using **BLEU Score**

### Features

- Custom-built **Encoder-Decoder architecture**
- Optional **Bahdanau (Additive)** or **Luong (Multiplicative)** attention
- Support for **teacher forcing**
- BLEU score computation for model evaluation
- Visual attention heatmaps

---

## Dataset

The model uses the `spa.txt` dataset, a collection of **tab-separated English-Spanish sentence pairs**.

> 🔹 You can upload this file manually in Colab  
> 🔹 File format: `English_sentence \t Spanish_sentence` per line

##  Model Architecture

### 🔸 Encoder
- Embedding Layer
- LSTM

### 🔸 Attention (Optional)
- Bahdanau (Additive) OR Luong (Multiplicative)

### 🔸 Decoder
- Embedding Layer
- LSTM with Context Vector
- Fully Connected Layer + Softmax

---

##  Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- NLTK (`bleu_score`)
