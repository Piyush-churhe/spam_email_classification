import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import string
import pickle
import kagglehub

class SimpleSpamClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = MultinomialNB()

    def preprocess_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
        return text

    def train(self, emails, labels):
        processed = [self.preprocess_text(email) for email in emails]
        X = self.vectorizer.fit_transform(processed)
        self.model.fit(X, labels)
        return self

    def predict(self, emails):
        processed = [self.preprocess_text(email) for email in emails]
        X = self.vectorizer.transform(processed)
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        return preds, probs

    def evaluate(self, emails, labels):
        preds, _ = self.predict(emails)
        acc = accuracy_score(labels, preds)
        print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(labels, preds, target_names=['Ham', 'Spam']))
        print("Confusion Matrix:")
        print(confusion_matrix(labels, preds))
        return acc

    def save_model(self, path='spam_classifier_khub.pkl'):
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'model': self.model
            }, f)
        print(f"✅ Model saved to: {path}")

    def load_model(self, path='spam_classifier_khub.pkl'):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.vectorizer = data['vectorizer']
        self.model = data['model']
        print(f"✅ Model loaded from: {path}")

def load_and_train():
    import os
    print("📦 Downloading dataset from KaggleHub...")
    path = kagglehub.dataset_download("venky73/spam-mails-dataset")

    print("✅ Dataset downloaded at:", path)
    print("📁 Available files:", os.listdir(path))

    
    csv_path = os.path.join(path, 'spam_ham_dataset.csv')  

    print(f"✅ Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']]
        df.columns = ['label', 'email']
    elif 'label' in df.columns and 'text' in df.columns:
        df = df[['label', 'text']]
        df.columns = ['label', 'email']
    else:
        raise ValueError("Unknown column names. Please check the dataset.")

    df['label'] = df['label'].map({'spam': 1, 'ham': 0})

    X_train, X_test, y_train, y_test = train_test_split(
        df['email'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    clf = SimpleSpamClassifier()
    clf.train(X_train.tolist(), y_train.tolist())
    clf.evaluate(X_test.tolist(), y_test.tolist())
    clf.save_model()
    return clf


def test_examples(classifier):
    print("\n📨 Testing Sample Emails")
    samples = [
        "Hi John, your meeting is confirmed for 10 AM tomorrow.",
        "Congratulations! You’ve won a $1000 Walmart gift card. Click now!",
        "Your order has been shipped and will arrive tomorrow.",
        "URGENT: Your account will be suspended unless you respond now!",
        "Let's catch up over lunch this week.",
        "Meeting moved to 3pm tomorrow. Conference room B.",
        "Get rich quick! Make $1000 per day from home"
    ]
    preds, probs = classifier.predict(samples)
    for i, email in enumerate(samples):
        label = "SPAM" if preds[i] == 1 else "HAM"
        confidence = max(probs[i]) * 100
        print(f"\nEmail: {email[:60]}...")
        print(f"Prediction: {label} (Confidence: {confidence:.1f}%)")

if __name__ == "__main__":
    classifier = load_and_train()
    test_examples(classifier)
