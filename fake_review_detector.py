import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class FakeReviewDetector:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words]
        return ' '.join(tokens)

    def extract_features(self, reviews_df):
        # Combine positive and negative reviews for feature extraction
        reviews_df['combined_reviews'] = reviews_df['Positive_Review'].fillna("") + " " + reviews_df['Negative_Review'].fillna("")

        features = {
            'review_length': reviews_df['combined_reviews'].str.len(),
            'caps_ratio': reviews_df['combined_reviews'].apply(
                lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
            ),
            'exclamation_count': reviews_df['combined_reviews'].str.count('!'),
            'question_count': reviews_df['combined_reviews'].str.count(r'\\?'),
            'reviewer_score': reviews_df['Reviewer_Score']
        }
        return pd.DataFrame(features)

    def train(self, reviews_df):
        # Label fake reviews based on simple heuristic (for demonstration)
        reviews_df['is_fake'] = reviews_df.apply(
            lambda row: 1 if row['Review_Total_Positive_Word_Counts'] > 50 and row['Reviewer_Score'] < 4 else 0,
            axis=1
        )

        # Preprocess reviews
        reviews_df['processed_text'] = (
            reviews_df['Positive_Review'].fillna("") + " " + reviews_df['Negative_Review'].fillna("")
        )
        reviews_df['processed_text'] = reviews_df['processed_text'].apply(self.preprocess_text)

        # Create TF-IDF features
        text_features = self.vectorizer.fit_transform(reviews_df['processed_text'])

        # Extract additional features
        additional_features = self.extract_features(reviews_df)

        # Combine all features
        X = np.hstack((text_features.toarray(), additional_features.values))
        y = reviews_df['is_fake']

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        self.classifier.fit(X_train, y_train)

        # Evaluate the model
        y_pred = self.classifier.predict(X_test)
        print("Model Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    def predict(self, positive_review, negative_review, reviewer_score):
        # Combine and preprocess review text
        combined_review = positive_review + " " + negative_review
        processed_text = self.preprocess_text(combined_review)
        text_features = self.vectorizer.transform([processed_text])

        # Extract additional features
        additional_features = pd.DataFrame({
            'review_length': [len(combined_review)],
            'caps_ratio': [
                sum(1 for c in combined_review if c.isupper()) / len(combined_review) if len(combined_review) > 0 else 0
            ],
            'exclamation_count': [combined_review.count('!')],
            'question_count': [combined_review.count('?')],
            'reviewer_score': [reviewer_score]
        })

        # Combine features
        features = np.hstack((text_features.toarray(), additional_features.values))

        # Predict
        prediction = self.classifier.predict(features)[0]
        probability = self.classifier.predict_proba(features)[0]

        return {
            'is_fake': bool(prediction),
            'confidence': float(probability[1] if prediction else probability[0])
        }

def main():
    # Load dataset
    file_path = '/home/rudddyy/Downloads/archive/Hotel_Reviews.csv'
    reviews_df = pd.read_csv(file_path)

    # Ensure required NLTK data is downloaded
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

    # Train and test the model
    detector = FakeReviewDetector()
    detector.train(reviews_df)

    # Test prediction
    positive_review = "Amazing service, great location!"
    negative_review = "Room was small but manageable."
    reviewer_score = 4.0

    result = detector.predict(positive_review, negative_review, reviewer_score)
    print(f"\nPositive Review: {positive_review}")
    print(f"Negative Review: {negative_review}")
    print(f"Prediction: {'Fake' if result['is_fake'] else 'Genuine'}")
    print(f"Confidence: {result['confidence']:.2f}")

if __name__ == "__main__":
    main()
