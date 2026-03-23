# Sentiment Analysis Using VADER (NLTK)

A Python project that performs sentiment analysis on product reviews using the **VADER** (Valence Aware Dictionary and sEntiment Reasoner) model from the `nltk` library.

## Overview

VADER is a rule-based sentiment analysis tool specifically designed for social media text and short reviews. Unlike machine learning models, it requires no training data — it uses a built-in lexicon to score sentiment instantly.

## Features

- Rule-based sentiment scoring (no training required)
- Classifies reviews as **Positive**, **Negative**, or **Neutral**
- Generates four sentiment scores: `neg`, `neu`, `pos`, and `compound`
- DataFrame output for easy visualization
- Bar chart for sentiment distribution
- Supports analysis of individual custom reviews

## Requirements

Install the required libraries:

```bash
pip install nltk pandas matplotlib seaborn
```

## Project Structure

```
Sentiment Analysis Using VADER/
│
├── vader_sentiment.py    # Main script
└── README.md             # Project documentation
```

## How It Works

| Step | Description |
|------|-------------|
| 1 | Install and import `nltk`, `pandas`, `matplotlib`, `seaborn` |
| 2 | Download the `vader_lexicon` using `nltk.download()` |
| 3 | Prepare a dataset of product reviews |
| 4 | Initialize `SentimentIntensityAnalyzer` and score each review |
| 5 | Convert scores to a Pandas DataFrame |
| 6 | Categorize reviews using the compound score |
| 7 | Visualize sentiment distribution with a bar chart |
| 8 | Analyze specific custom reviews manually |

## Usage

Run the main script:

```bash
python vader_sentiment.py
```

## Sentiment Scoring

VADER returns four scores for each review:

| Score | Description |
|-------|-------------|
| `neg` | Proportion of negative sentiment (0 to 1) |
| `neu` | Proportion of neutral sentiment (0 to 1) |
| `pos` | Proportion of positive sentiment (0 to 1) |
| `compound` | Overall sentiment score (-1 to +1) |

## Sentiment Categories

Reviews are classified based on the compound score:

| Compound Score | Category |
|----------------|----------|
| > 0.05 | Positive |
| < -0.05 | Negative |
| Between -0.05 and 0.05 | Neutral |

## Example Output

```
Review: The product was not good, but the service was excellent!
Scores: {'neg': 0.113, 'neu': 0.532, 'pos': 0.355, 'compound': 0.6847}
Category: Positive
```

## Sample Dataset

| Review | Category |
|--------|----------|
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
| `nltk` | VADER sentiment analysis |
| `pandas` | DataFrame creation and manipulation |
| `matplotlib` | Plotting |
| `seaborn` | Sentiment distribution bar chart |

## VADER vs. Naive Bayes (Comparison)

| Feature | VADER | Naive Bayes |
|---------|-------|-------------|
| Training data required | No | Yes |
| Works out of the box | Yes | No |
| Handles mixed sentiment | Yes | Limited |
| Best for | Short texts, reviews | Labeled datasets |

## Limitations

- VADER may struggle with sarcasm or highly context-dependent language
- The sample dataset is small and for demonstration only
- For domain-specific text, a trained ML model may perform better

## License

This project is open-source and available for educational use.
