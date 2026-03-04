# Next Word Predictor using LSTM

## 📌 Overview

This project implements a **Next Word Prediction system using Deep Learning**.
The model learns patterns from text and predicts the **most probable next word** based on a sequence of previous words.

Next-word prediction is a fundamental task in **Natural Language Processing (NLP)** and is widely used in:

* Chatbots
* Smart keyboard suggestions
* AI writing assistants
* Search engines
* Text generation systems

The model uses **Long Short-Term Memory (LSTM)** networks to capture sequential dependencies in language.

---

## ⚙️ Tech Stack

* Python
* TensorFlow
* Keras
* NumPy
* Natural Language Processing (NLP)

---

## 📂 Dataset

The dataset contains **human conversational sentences** stored in a `.txt` file.

Example format:

```
hello how are you
i am fine thank you
what are you doing
i am learning deep learning
machine learning is fascinating
```

Each line represents **one sentence**.

---

## 🧠 Project Pipeline

### 1. Load Dataset

```python
with open("human_chat.txt", "r", encoding="utf-8") as f:
    text = f.read()
```

---

### 2. Tokenization

Text is converted into numerical tokens.

```python
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
```

Example:

```
I love machine learning
↓
[1, 2, 3, 4]
```

---

### 3. Generate Input Sequences

Incremental sequences are created from sentences.

Example:

```
I love machine learning
```

Generated sequences:

```
[I, love]
[I, love, machine]
[I, love, machine, learning]
```

These sequences allow the model to learn:

```
previous words → next word
```

---

### 4. Padding

Sequences are padded to ensure equal length.

Example:

```
[0,0,1,2]
[0,1,2,3]
[1,2,3,4]
```

---

### 5. Split Input and Output

```python
X = padded_sequences[:, :-1]
y = padded_sequences[:, -1]
```

Where:

* **X** = input sequence
* **y** = next word label

---

## 🏗 Model Architecture

```
Input Text
     │
     ▼
Tokenizer
     │
     ▼
Sequence Generation
     │
     ▼
Padding
     │
     ▼
Embedding Layer
     │
     ▼
LSTM Layer
     │
     ▼
Dense Layer
     │
     ▼
Softmax
     │
     ▼
Next Word Prediction
```

---

### Model Implementation

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential()

model.add(Embedding(vocab_size, 100, input_length=max_len-1))
model.add(LSTM(150))
model.add(Dense(vocab_size, activation='softmax'))
```

---

## 🧪 Training

The model is trained using:

* **Loss Function:** Categorical Crossentropy
* **Optimizer:** Adam

```python
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(X, y, epochs=100)
```

---

## 🔮 Example Prediction

Input:

```
deep learning
```

Predicted output:

```
deep learning models
```

The model predicts the **most probable next word** based on learned patterns.

---

