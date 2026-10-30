import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df =pd.read_csv("spam.csv", encoding="latin-1")

# 2. Select message and label columns
if "v1" in df.columns:
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]
else:
    df = df[["Category", "Message"]]
    df.columns = ["label", "message"]

# 3. Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

df["message"] = df["message"].apply(clean_text)
df = df.dropna()

# 4. Convert text into TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["message"])
y = df["label"].str.lower()

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# 7. Evaluate model
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 8. Save model and vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and TF-IDF Vectorizer saved successfully!")