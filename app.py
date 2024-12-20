import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

## Loading the word index
word_index = imdb.get_word_index()
reverse_word_index = {value : key for key, value in word_index.items()}

## Loading the model
model = load_model('Models/model.h5')

## Helper function for decoding the review
def decode_review(encoded_review) :
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

## Helper function to preprocess user input
def preprocess_text(text) :
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen = 500)
    return padded_review

## Creating Prediction function
def predict_sentiment(review) :
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return sentiment, prediction[0][0]

## Steamlit app
try : 
    st.title('IMDB Movie Review Sentiment Analysis')
    st.write('Enter a movie review to classify it as a Positive or a Negative Review.')

    user_input = st.text_area('Movie Review')

    if st.button('Classify') :
    
        sentiment, score = predict_sentiment(user_input)
    
        st.write(f'Sentiment : {sentiment}')
        st.write(f'Prediction Score : {score}')
    else :
        st.write('Please enter a movie review.')
except(Exception) as e :
    st.write(f'An error occurred, Try Again!')