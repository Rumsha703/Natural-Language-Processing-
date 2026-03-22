# Bag of Words - CountVectorizer

## Overview
This project demonstrates how to convert text data into numerical format using the **Bag of Words (BoW)** model with `CountVectorizer` from the `scikit-learn` library.

---

## What is Bag of Words?
Bag of Words is a simple text representation technique used in **Natural Language Processing (NLP)**. It counts how many times each word appears in a sentence — ignoring grammar and word order.

---

## Code

```python
from sklearn.feature_extraction.text import CountVectorizer

# Sample text data
corpus = [
    "I love NLP",
    "NLP is amazing",
    "I love learning NLP"
]

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the corpus
X = vectorizer.fit_transform(corpus)

# Convert to array for better visualization
print("Vocabulary:", vectorizer.vocabulary_)
print("Bag of Words Representation:\n", X.toarray())
```

---

## Output
```
Vocabulary: {'love': 3, 'nlp': 4, 'is': 1, 'amazing': 0, 'learning': 2}
Bag of Words Representation:
 [[0 0 0 1 1]
  [1 1 0 0 1]
  [0 0 1 1 1]]
```

---

## Output Explanation

### Vocabulary
Each unique word is assigned an index alphabetically:

| Index | Word     |
|-------|----------|
| 0     | amazing  |
| 1     | is       |
| 2     | learning |
| 3     | love     |
| 4     | nlp      |

### Bag of Words Matrix
Each row represents a sentence. Each column represents a word. The number shows how many times the word appears.

| Sentence              | amazing | is | learning | love | nlp |
|-----------------------|---------|----|----------|------|-----|
| "I love NLP"          | 0       | 0  | 0        | 1    | 1   |
| "NLP is amazing"      | 1       | 1  | 0        | 0    | 1   |
| "I love learning NLP" | 0       | 0  | 1        | 1    | 1   |

- **0** = word is absent in that sentence
- **1** = word appears once in that sentence

---

## Requirements

```bash
pip install scikit-learn
```

---

## How to Run

```bash
python count_vectorizer.py
```

---

## Concepts Covered
- Text vectorization
- Bag of Words model
- CountVectorizer from scikit-learn
- Vocabulary indexing
- Sparse matrix to array conversion

---

## Author
Rumsha703 — Natural Language Processing Repository
