#📜 Text Generation Using RNNs and LSTMs  

##🔹 **Overview**  
This project explores **text generation using Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks**.  
We compare two encoding techniques:  
1. **One-Hot Encoding**  
2. **Trainable Word Embeddings**  

The goal is to **train an RNN and LSTM model** on a dataset of 100 poems and compare their performance.  

---

## **📂 Dataset**  
The dataset (`poems-100.csv`) contains **100 poems**. The text is preprocessed and tokenized before training.  

---

## **📌 Implementation Details**  

### **🔹 Model Variants**  
We implement **four models**:  
✅ **RNN with One-Hot Encoding**  
✅ **LSTM with One-Hot Encoding**  
✅ **RNN with Trainable Embeddings**  
✅ **LSTM with Trainable Embeddings**

Model	Training   Time (s)	  Average Loss
RNN One-Hot	     ~50s	        ~2.3
LSTM One-Hot	   ~70s	        ~1.8
RNN Embeddings	 ~90s	        ~1.0
LSTM Embeddings	 ~115s	      ~0.5

🔹 ***Advantages & Disadvantages of Each Approach***
📌**One-Hot Encoding**
✅ Advantages:
->Simple and interpretable
->Works well with small datasets

❌ Disadvantages:
->Very memory-intensive (high-dimensional vectors)
->Fails to capture word relationships

📌 **Trainable Word Embeddings**
✅ Advantages:
->Efficient (low-dimensional representation)
->Captures semantic meaning (e.g., "king" & "queen" are related)
->Better generalization on unseen data

❌ Disadvantages:
->Requires more training data
->Initial embeddings are random, so performance improves over time

***🎯 Conclusion***
->LSTMs with embeddings perform best for text generation.
->One-Hot Encoding is inefficient and struggles with large vocabularies.
->Embeddings improve contextual understanding and generalization.
->LSTM should be preferred over RNN for sequential text tasks.
