# Machine Learning-Based Sentiment Analysis (Logistic Regression + TF-IDF)

A Python project that implements a complete sentiment analysis pipeline using **Logistic Regression** and **TF-IDF** feature extraction to classify product reviews as Positive or Negative.

## Overview

This project builds a full machine learning pipeline for sentiment analysis, covering data preparation, text preprocessing, feature extraction, model training, and evaluation. It uses **TF-IDF** (Term Frequency-Inverse Document Frequency) instead of simple bag-of-words, making it more effective at capturing meaningful words in reviews.

## Features

- Expanded real-world product review dataset (17 samples)
- TF-IDF vectorization with bigram support
- Logistic Regression binary classifier
- Accuracy score, classification report, and confusion matrix
- Specific example prediction for custom reviews

## Requirements

Install the required libraries:

```bash
pip install scikit-learn matplotlib seaborn
```

## Project Structure

```
Sentiment Analysis Using Logistic Regression/
│
├── logistic_sentiment.py    # Main script
└── README.md                # Project documentation
```

## Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | Data Collection — 17 labeled product reviews (Positive/Negative) |
| 2 | Text Preprocessing — TF-IDF with stop word removal and bigrams |
| 3 | Train/Test Split — 75% training, 25% testing |
| 4 | Model Training — Logistic Regression (`max_iter=1000`) |
| 5 | Evaluation — Accuracy, classification report, confusion matrix |
| 6 | Custom Prediction — Analyze specific unseen reviews |

## Usage

Run the main script:

```bash
python logistic_sentiment.py
```

## TF-IDF Vectorizer Settings

```python
TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `max_features` | 1000 | Limits vocabulary to top 1000 terms |
| `stop_words` | 'english' | Removes common words like "the", "is" |
| `ngram_range` | (1, 2) | Captures single words and two-word phrases |

## Example Output

```
Accuracy: 0.80

Classification Report:
              precision    recall  f1-score   support
    Negative       0.80      0.80      0.80         5
    Positive       0.80      0.80      0.80         5

Review: The product quality is amazing. Highly recommend it!
Predicted Sentiment: Positive

Review: I hated the customer service. Worst experience ever!
Predicted Sentiment: Negative

Review: It's okay. Not great, but not bad either.
Predicted Sentiment: Negative
```

## Sample Dataset

| Review | Label |
|--------|-------|
| "I absolutely loved this product! It exceeded all my expectations." | Positive |
| "Worst experience ever. The product broke after one use." | Negative |
| "Amazing quality and outstanding performance. Highly recommend!" | Positive |
| "Terrible! Do not waste your money on this." | Negative |
| "Good value for the price. I'm satisfied." | Positive |
| "Completely disappointed. The customer service was awful." | Negative |
| "Excellent! Works perfectly and is very durable." | Positive |
| "Horrible product. It's a complete waste of money." | Negative |
| "Best purchase ever! Exceptional quality and performance." | Positive |
| "Poor quality and terrible customer service. Avoid at all costs." | Negative |

*...and 7 more reviews*

## Technologies Used

| Library | Purpose |
|---------|---------|
| `scikit-learn` | TF-IDF, Logistic Regression, evaluation metrics |
| `matplotlib` | Plotting |
| `seaborn` | Confusion matrix heatmap |

## Troubleshooting: Low Accuracy

If your accuracy is lower than expected (e.g. 0.40 instead of 0.80), this is likely due to the small dataset size. Recommended fixes:

- **Try different `random_state` values** in `train_test_split` (e.g. `0`, `7`, `21`)
- **Reduce `test_size`** from `0.25` to `0.15`
- **Add more labeled reviews** — at least 500–1000 samples recommended
- **Use cross-validation** for more stable results:

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)
print(f"Cross-Validation Accuracy: {scores.mean():.2f}")
```

## Comparison: This Project vs Other Approaches

| Feature | Logistic Regression + TF-IDF | Naive Bayes + BoW | VADER |
|---------|-------------------------------|-------------------|-------|
| Training data required | Yes | Yes | No |
| Handles bigrams | Yes | No | No |
| Best for | Labeled datasets | Simple classification | Short texts |
| Accuracy (small data) | Moderate | Moderate | Rule-based |

## Limitations

- Dataset is small (17 samples) — results may vary with different random splits
- For production use, a larger and more diverse dataset is strongly recommended
- The model does not handle sarcasm or context-dependent language well

## License

This project is open-source and available for educational use.
