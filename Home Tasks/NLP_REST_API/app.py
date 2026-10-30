import re
import joblib

from fastapi import FastAPI
from pydantic import BaseModel

# Load saved model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

app = FastAPI()

# JSON input format
class SMS(BaseModel):
    message: str

# Clean message
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

# 9. POST prediction endpoint
@app.post("/predict")
def predict_sms(data: SMS):

    message = clean_text(data.message)

    features = vectorizer.transform([message])

    prediction = model.predict(features)[0]

    confidence = max(model.predict_proba(features)[0])

    # 10. Return prediction and confidence
    return {
        "prediction": prediction,
        "confidence": round(float(confidence), 2)
    }