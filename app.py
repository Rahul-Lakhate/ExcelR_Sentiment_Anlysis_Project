import joblib

joblib.dump(xgb_model, 'model.pkl')
joblib.dump(tfidf, 'vectorizer.pkl')
