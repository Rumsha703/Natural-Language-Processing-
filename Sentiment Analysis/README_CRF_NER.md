# CRF for Named Entity Recognition (NER)

> Implementation of Conditional Random Fields (CRF) for Named Entity Recognition using Python's `sklearn-crfsuite` library.

---

## Overview

This project implements **Conditional Random Fields (CRF)**, a powerful sequence labeling algorithm, to perform **Named Entity Recognition (NER)** — the task of identifying and classifying named entities (people, organizations, locations, etc.) in text.

---

## Features

- Custom feature extraction per word (case, context, position)
- CRF model training using `sklearn-crfsuite`
- Prediction of NER labels on new sentences
- Model evaluation with precision, recall, and F1-score

---

## Entity Labels

| Label | Description | Example |
|-------|-------------|---------|
| `ORG` | Organization | MetaBrains, AI |
| `PERSON` | Person name | Jane, John |
| `GPE` | Geo-political entity | Berlin, California |
| `O` | Non-entity (other) | is, works, in |

---

## Project Structure

```
crf-ner/
├── crf_ner.py        # Main pipeline script
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```

---

## Requirements

Python 3.7+ is required. Install dependencies with:

```bash
pip install sklearn-crfsuite pandas
```

| Library | Purpose |
|---------|---------|
| `sklearn-crfsuite` | CRF model training & evaluation |
| `pandas` | Data handling |

---

## Pipeline — Step by Step

### Step 1: Install & Import Libraries

```python
# Install the CRFsuite library
!pip install sklearn-crfsuite

# Import necessary libraries
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import pandas as pd
```

---

### Step 2: Prepare the Dataset

Each sentence is a list of `(word, label)` tuples.

```python
sentences = [
    [("MetaBrains", "ORG"), ("is", "O"), ("expanding", "O"),
     ("operations", "O"), ("to", "O"), ("Berlin", "GPE")],
    [("Jane", "PERSON"), ("works", "O"), ("at", "O"), ("MetaBrains", "ORG")],
    [("AI", "ORG"), ("launch", "O"), ("will", "O"),
     ("happen", "O"), ("in", "O"), ("California", "GPE")]
]

for sent in sentences:
    print(sent)
```

---

### Step 3: Feature Extraction

Features are extracted for each word — including its case, surrounding words, and position in the sentence.

```python
def word2features(sent, i):
    word = sent[i][0]
    features = {
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }

    if i > 0:
        prev_word = sent[i-1][0]
        features.update({'prev_word.lower()': prev_word.lower()})
    else:
        features['BOS'] = True  # Beginning of sentence

    if i < len(sent) - 1:
        next_word = sent[i+1][0]
        features.update({'next_word.lower()': next_word.lower()})
    else:
        features['EOS'] = True  # End of sentence

    return features


def extract_features_and_labels(sentences):
    X = [[word2features(sent, i) for i in range(len(sent))] for sent in sentences]
    y = [[label for _, label in sent] for sent in sentences]
    return X, y


# Extract features and labels
X, y = extract_features_and_labels(sentences)
```

---

### Step 4: Train the CRF Model

```python
# Initialize and train the CRF model
crf = sklearn_crfsuite.CRF()
crf.fit(X, y)
```

---

### Step 5: Make Predictions

```python
# Test sentence
test_sent = [("John", "O"), ("works", "O"), ("in", "O"), ("California", "GPE")]
X_test = [[word2features(test_sent, i) for i in range(len(test_sent))]]

# Predict the labels
y_pred = crf.predict(X_test)
print("Predicted Labels:", y_pred)
```

**Expected Output:**
```
Predicted Labels: [['O', 'O', 'O', 'GPE']]
```

---

### Step 6: Evaluate the Model

The model is evaluated using precision, recall, and F1-score for each entity class.

```python
labels = list(crf.classes_)
print(metrics.flat_classification_report(y, crf.predict(X), labels=labels))
```

**Expected Output:**
```
              precision    recall  f1-score   support

         GPE       1.00      1.00      1.00         2
           O       1.00      1.00      1.00         7
         ORG       1.00      1.00      1.00         3
      PERSON       1.00      1.00      1.00         1

   micro avg       1.00      1.00      1.00        13
   macro avg       1.00      1.00      1.00        13
weighted avg       1.00      1.00      1.00        13
```

---

## Complete Script

```python
!pip install sklearn-crfsuite

import sklearn_crfsuite
from sklearn_crfsuite import metrics
import pandas as pd

# Dataset
sentences = [
    [("MetaBrains", "ORG"), ("is", "O"), ("expanding", "O"),
     ("operations", "O"), ("to", "O"), ("Berlin", "GPE")],
    [("Jane", "PERSON"), ("works", "O"), ("at", "O"), ("MetaBrains", "ORG")],
    [("AI", "ORG"), ("launch", "O"), ("will", "O"),
     ("happen", "O"), ("in", "O"), ("California", "GPE")]
]

# Feature extraction
def word2features(sent, i):
    word = sent[i][0]
    features = {
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        features.update({'prev_word.lower()': sent[i-1][0].lower()})
    else:
        features['BOS'] = True
    if i < len(sent) - 1:
        features.update({'next_word.lower()': sent[i+1][0].lower()})
    else:
        features['EOS'] = True
    return features

def extract_features_and_labels(sentences):
    X = [[word2features(sent, i) for i in range(len(sent))] for sent in sentences]
    y = [[label for _, label in sent] for sent in sentences]
    return X, y

X, y = extract_features_and_labels(sentences)

# Train
crf = sklearn_crfsuite.CRF()
crf.fit(X, y)

# Predict
test_sent = [("John", "O"), ("works", "O"), ("in", "O"), ("California", "GPE")]
X_test = [[word2features(test_sent, i) for i in range(len(test_sent))]]
print("Predicted Labels:", crf.predict(X_test))

# Evaluate
labels = list(crf.classes_)
print(metrics.flat_classification_report(y, crf.predict(X), labels=labels))
```

---

## Notes

- The sample dataset is very small — use a larger annotated corpus (e.g., CoNLL-2003) for real-world performance.
- Add more features such as word suffixes, prefixes, or POS tags to improve accuracy.
- CRF works well for sequence labeling tasks like NER, POS tagging, and chunking.

---

## License

MIT License — free to use and modify for personal and commercial projects.
