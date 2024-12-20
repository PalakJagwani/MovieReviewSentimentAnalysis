# Movie Review Sentiment Analysis Using Simple RNN

This project demonstrates a sentiment analysis model for classifying movie reviews as positive or negative using a Simple Recurrent Neural Network (RNN). The model is trained on a dataset of movie reviews and is capable of predicting the sentiment behind a given review text.

## Overview

The goal of this project is to implement a sentiment analysis model using a Simple RNN. The model is trained to classify movie reviews into two categories: positive or negative. A dataset containing labeled reviews is used to train the RNN model, and the final model is evaluated on its accuracy in predicting sentiments.

## Technologies Used

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Streamlit

## Dataset

The IMDB dataset is used for training the model which is a collection of movie reviews labeled with either "positive" or "negative" sentiment.

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/PalakJagwani/MovieReviewSentimentAnalysis.git
   cd MovieReviewSentimentAnalysis
   ```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

## Model Architecture
The Simple RNN model architecture is as follows:

- Embedding Layer: Converts words into vector representations.
- Simple RNN Layer: Processes the sequential nature of the review data.
- Dense Layer: Outputs the sentiment prediction (positive or negative).
- Activation: Uses a sigmoid activation function for binary classification.


## Live link for the project : Deployed using Streamlit
```
https://moviereviewsentimentanalysis-gugdn8a4xuter3hnpgsqyp.streamlit.app/
```

## This project was completed as part of the Complete Data Science Bootcamp course by Krish Naik on Udemy. 