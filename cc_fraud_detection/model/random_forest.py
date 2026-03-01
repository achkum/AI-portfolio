import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def train_model(self, X_train, y_train):
        print("Training Random Forest supervised baseline...")
        self.rf.fit(X_train, y_train)

    def predict_anomaly(self, X_test):
        """Predict anomaly score (probability of class 1)."""
        probs = self.rf.predict_proba(X_test)
        # Probabilities for class 1 (Fraud)
        return probs[:, 1]
