import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from textblob import TextBlob
import joblib

# Load dataset
file_path = "P556.xlsx"
df = pd.read_excel(file_path)

# Clean and label text

def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", str(text).lower())
    return text

def get_sentiment_label(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.05:
        return 2  # Positive
    elif polarity < -0.05:
        return 0  # Negative
    else:
        return 1  # Neutral

# Apply cleaning and labeling
df = df.dropna(subset=['body'])
df['cleaned_text'] = df['body'].apply(clean_text)
df['label'] = df['body'].apply(get_sentiment_label)

# Split the dataset
X = df['cleaned_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train_tfidf, y_train)

# Save model and vectorizer
joblib.dump(xgb_model, 'model.pkl')
joblib.dump(tfidf, 'vectorizer.pkl')

print("Model and vectorizer saved as 'model.pkl' and 'vectorizer.pkl'")
