Hands-on Project: Text Preprocessing Pipeline
In this mini project, we will build a text preprocessing pipeline from scratch. The pipeline will take raw text as input and perform several steps of preprocessing to clean and prepare it for analysis.
The goal is to apply all the concepts we have learned so far: tokenization, stopword removal, stemming/lemmatization, and case conversion. We will also provide options for switching between stemming and lemmatization.

1. Problem Statement
When working with text data (such as product reviews, tweets, or news articles), the text is often noisy and contains unnecessary information. Preprocessing the text helps clean it and makes it suitable for tasks like sentiment analysis, classification, or topic modeling.
Objective:
Develop a text preprocessing pipeline that:
Tokenizes the input text
Removes stopwords
Converts text to lowercase
Applies either stemming or lemmatization
Returns the cleaned text ready for analysis

2. Step-by-Step Plan
We will follow these steps:
Input: Take raw text as input from the user.
Tokenization: Split the text into individual words.
Stopword Removal: Filter out common stopwords.
Case Conversion: Convert all text to lowercase.
Stemming/Lemmatization: Reduce words to their root form (based on user choice).
Output: Display the cleaned tokens.

3. Implementation
We will now define the preprocessing pipeline. The pipeline will take a piece of text as input and allow the user to choose between stemming and lemmatization.

4. Test the Pipeline
