import streamlit as st
import joblib
import re
import os

# ========== Load Model & Vectorizer ==========
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model, vectorizer = load_model()

# ========== Text Cleaning Function ==========
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", str(text).lower())
    return text

# ========== Predict Function ==========
def predict_sentiment(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map.get(prediction, "Unknown")

# ========== Streamlit UI ==========
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("🧠 Sentiment Analysis App")
st.markdown("Enter a review below to analyze its sentiment:")

user_input = st.text_area("💬 Your Review", height=150)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        result = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: **{result}**")
