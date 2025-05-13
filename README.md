# Sentiment analysis model
A model that analyses emotion in user comments.
## Table of contents
* [General Information](#general-information)
* [Technologies Used](#techonologies-used)
* [Features](#features)
* [Setup](#setup)
* [Project Status](#project-status)
## General Information
A model trained to analyze emotion in user comment. Then, it is applied in a data app to analyze comments.
## Technologies Used
* Python 3.12.1
* Numpy
* Pandas
* scikit-learn
* Naive Bayes model
* Streamlit
## Features
1. Dataset
Data includes 500 comments, which are crawled from social medias.
Dataset involves four columns: comment, positive, negative, neutral.
2. Data preprocessing
- Create labels and shuffle data
- Convert to lowercase
- Expand teencode and abbreviations.
- Tokenizing.
- Remove punctuation and non-words.
- Remove stopwords.
- Lemmatization.
3. Model
- Use CountVectorizer to vectorize text.
- Use Multinomial Naive Bayes to build and train model.
4. Data app
- Use streamlit to develop a simple data app, which allows user to enter a comment. Model will analyze the comment and predict the sentiment
## Setup
To run this project, install Python libraries:
* Numpy
* Pandas
* Matplotlib
* scikit-learn
* streamlit



