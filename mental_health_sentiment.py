 -----------------------------
# Sentiment Analysis for Mental Health
# -----------------------------
# Author: Srinidhi A (Example)
# Using Kaggle Dataset: https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health
# -----------------------------

import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------------
# 1. Load the dataset
# -----------------------------
# Change the file path to where you downloaded the Kaggle CSV file
df = pd.read_csv("sentiment.csv")   # e.g., 'train.csv' or 'mental_health.csv'

print("✅ Dataset Loaded Successfully")
print("Shape of data:", df.shape)
print("\nColumns:", df.columns)

# -----------------------------
# 2. Basic info and cleaning
# -----------------------------
print("\nMissing values:\n", df.isnull().sum())

# Assuming columns: 'text' and 'label' (adjust if names differ)
df = df.dropna(subset=['text', 'label'])
df = df[df['text'].str.strip() != ""]

# -----------------------------
# 3. Clean the text
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)      # remove URLs
    text = re.sub(r"@\w+", "", text)         # remove @mentions
    text = re.sub(r"#\w+", "", text)         # remove hashtags
    text = re.sub(r"[^a-z\s]", "", text)     # keep only letters
    text = re.sub(r"\s+", " ", text).strip() # remove extra spaces
    return text

df["clean_text"] = df["text"].apply(clean_text)

# -----------------------------
# 4. Visualize label distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x=df['label'])
plt.title("Distribution of Mental Health Sentiments")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# -----------------------------
# 5. Split dataset
# -----------------------------
X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 6. Feature Extraction (TF-IDF)
# -----------------------------
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# -----------------------------
# 7. Model Training
# -----------------------------
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# -----------------------------
# 8. Predictions
# -----------------------------
y_pred = model.predict(X_test_tfidf)

# -----------------------------
# 9. Evaluation
# -----------------------------
print("\n📊 Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# 10. Test with your own text
# -----------------------------
while True:
    user_text = input("\nEnter a text to analyze sentiment (or 'exit' to quit): ")
    if user_text.lower() == "exit":
        break
    user_text_clean = clean_text(user_text)
    text_tfidf = tfidf.transform([user_text_clean])
    prediction = model.predict(text_tfidf)[0]
    print("🧠 Predicted Sentiment:", prediction)