import numpy as np
from tensorflow.keras.datasets import imdb
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


word_index=imdb.get_word_index()
reverse_word_index={value: key for key,value in word_index.items()}

model=load_model('simpleRnn_imdb.h5')
model.summary()

##fn to decode reviwes
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

## fn ti preprocess user input
def preprocess_text(test):
    words=test.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

##pred fn
def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment,prediction[0][0]


###streamlit app

import streamlit as st
st.title('IMDB movie review sentiment analysis')
st.write('Enter a movie review to classify it as positive or negative')

user_input=st.text_area('Movie Review')
if st.button('Classify'):
    preprocess_input=preprocess_text(user_input)


    prediction=model.predict(preprocess_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'


    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction_score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review')

