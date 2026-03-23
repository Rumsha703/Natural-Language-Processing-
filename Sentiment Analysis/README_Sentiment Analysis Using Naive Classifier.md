# Sentiment Analysis of Product Reviews

A machine learning project that performs binary sentiment classification on product reviews using a Naive Bayes classifier and the bag-of-words model.

## Overview

This project trains a model to classify product reviews as either **Positive** or **Negative** using Natural Language Processing (NLP) techniques from scikit-learn.

## Features

- Text vectorization using the bag-of-words model
- Multinomial Naive Bayes classification
- Model evaluation with accuracy score and classification report
- Confusion matrix visualization with heatmap

## Requirements

Install the required Python libraries:

```bash
pip install scikit-learn matplotlib seaborn
```

## Project Structure

```
sentiment-analysis/
│
├── sentiment_analysis.py   # Main script
└── README.md               # Project documentation
```

## How It Works

The project follows these steps:

1. **Import Libraries** — Load scikit-learn, matplotlib, and seaborn
2. **Load Dataset** — A sample dataset of 8 product reviews with corresponding labels
3. **Preprocess Data** — Convert text reviews into numerical features using `CountVectorizer` with English stop word removal
4. **Split Dataset** — Divide data into training (75%) and testing (25%) sets
5. **Train Model** — Fit a `MultinomialNB` (Naive Bayes) classifier on the training data
6. **Make Predictions** — Predict sentiments on the test set
7. **Evaluate Model** — Calculate accuracy, print a classification report, and display a confusion matrix heatmap

## Usage

Run the main script:

```bash
python sentiment_analysis.py
```

## Output

- **Accuracy Score** — Overall model accuracy on the test set
- **Classification Report** — Precision, recall, and F1-score for each class
- **Confusion Matrix** — Heatmap showing true vs. predicted sentiment labels

## Sample Dataset

| Review | Sentiment |
|--------|-----------|
| "This product is amazing! It exceeded all my expectations." | Positive |
| "Absolutely terrible! I will never buy this again." | Negative |
| "The quality is outstanding. I highly recommend it." | Positive |
| "Worst purchase I've ever made. It broke within a week." | Negative |
| "Great value for money. I'm very happy with it." | Positive |
| "Completely useless. Don't waste your money." | Negative |
| "I'm impressed with how durable this is." | Positive |
| "Terrible customer service. Very disappointing experience." | Negative |

## Technologies Used

| Library | Purpose |
|---------|---------|
| `scikit-learn` | Machine learning (vectorization, model, evaluation) |
| `matplotlib` | Plotting |
| `seaborn` | Confusion matrix heatmap |

## Limitations

- The dataset is very small (8 samples), so results are for demonstration purposes only
- For production use, a larger, more diverse dataset is recommended
- The bag-of-words model does not capture word order or context

## License

This project is open-source and available for educational use.
