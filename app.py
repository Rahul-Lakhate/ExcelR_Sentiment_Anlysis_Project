import streamlit as st
import joblib
import re
from textblob import TextBlob

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    return text

def get_sentiment_label(pred):
    mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return mapping.get(pred, "Unknown")

 
st.title("Sentiment Analysis App")
st.write("Enter a product review to analyze its sentiment:")

user_input = st.text_area("Your Review", "")

if st.button("Analyze"):
    cleaned = clean_text(user_input)
    vect_text = vectorizer.transform([cleaned])
    prediction = model.predict(vect_text)[0]
    sentiment = get_sentiment_label(prediction)
    st.success(f"Predicted Sentiment: **{sentiment}**")
