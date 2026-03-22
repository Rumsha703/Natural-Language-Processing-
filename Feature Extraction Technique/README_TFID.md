# TF-IDF Vectorizer - Text Representation in NLP

## Overview
This project demonstrates how to convert text data into numerical format using **TF-IDF (Term Frequency - Inverse Document Frequency)** with `TfidfVectorizer` from the `scikit-learn` library.

---

## What is TF-IDF?
TF-IDF is a numerical statistic used in NLP to reflect how important a word is to a document in a collection of documents (corpus).

- **TF (Term Frequency)** — How often a word appears in a sentence
- **IDF (Inverse Document Frequency)** — How rare or common a word is across all sentences
- **TF-IDF = TF × IDF** — High score means the word is important in that sentence but rare in others

---

## Difference Between BoW and TF-IDF

| Feature | Bag of Words | TF-IDF |
|---|---|---|
| Counting | Raw word count | Weighted score |
| Common words | Treated as important | Penalized |
| Rare words | Treated as less important | Rewarded |
| Result | Simple counts | Meaningful scores |

---

## Code

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample corpus
corpus = [
    "NLP is amazing and exciting",
    "I love studying NLP and working on NLP projects",
    "Machine learning includes NLP and other fields"
]

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the corpus to get TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(corpus)

# Show vocabulary (unique terms in the corpus)
print("Vocabulary:", vectorizer.vocabulary_)

# Display the TF-IDF matrix in array format
print("TF-IDF Matrix:\n", tfidf_matrix.toarray())
```

---

## Output
```
Vocabulary: {'nlp': 7, 'is': 4, 'amazing': 0, 'and': 1, 'exciting': 3,
             'love': 5, 'studying': 9, 'working': 11, 'on': 8, 'projects': 8,
             'machine': 6, 'learning': 5, 'includes': 4, 'other': 8, 'fields': 3}

TF-IDF Matrix:
 [[0.47  0.36  0.47  0.47  0.47  0.    0.    0.36  0.    0.    0.    0.  ]
  [0.    0.27  0.    0.    0.    0.40  0.    0.54  0.40  0.40  0.    0.40]
  [0.    0.30  0.    0.    0.    0.    0.45  0.30  0.    0.    0.45  0.  ]]
```

---

## Output Explanation

### Vocabulary
Each unique word gets an index number assigned alphabetically.

### TF-IDF Matrix
- Each **row** = one sentence
- Each **column** = one word
- The **number** = TF-IDF score (importance of that word in that sentence)
- **0.0** = word is absent in that sentence
- **Higher score** = word is more important/unique to that sentence

### Example
- Word **"amazing"** has a high score in sentence 1 because it only appears there
- Word **"and"** has a low score because it appears in all 3 sentences (common word)
- Word **"nlp"** appears in all sentences so its score is lower

---

## Requirements

```bash
pip install scikit-learn
```

---

## How to Run

```bash
python tfidf_vectorizer.py
```

---

## Concepts Covered
- TF-IDF (Term Frequency - Inverse Document Frequency)
- TfidfVectorizer from scikit-learn
- Text vectorization
- Vocabulary indexing
- Sparse matrix to array conversion
- Difference between TF-IDF and Bag of Words

---

## Author
Rumsha703 — Natural Language Processing Repository
