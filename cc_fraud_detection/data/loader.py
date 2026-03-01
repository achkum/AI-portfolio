import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

class CreditCardDataLoader:
    def __init__(self, raw_path="data/creditcard.csv", random_state=42):
        self.raw_path = raw_path
        self.scaler = StandardScaler()
        self.random_state = random_state
        # Constants
        self.TARGET = 'Class'
        self.TIME = 'Time'
        self.FEATURES = [f'V{i}' for i in range(1, 29)] + ['Amount']

    def load_and_split(self):
        """Loads data and performs temporal split (Day 1 train, Day 2 test)."""
        print(f"Loading data from {self.raw_path}...")
        df = pd.read_csv(self.raw_path)
        
        # Temporal split
        day_boundary = 86400
        train_df = df[df[self.TIME] <= day_boundary].copy()
        test_df = df[df[self.TIME] > day_boundary].copy()
        
        # Scale
        train_df[self.FEATURES] = self.scaler.fit_transform(train_df[self.FEATURES])
        test_df[self.FEATURES] = self.scaler.transform(test_df[self.FEATURES])
        
        return train_df, test_df

    def get_unsupervised_train(self, train_df):
        """Returns only legitimate transactions for AE/VAE training."""
        legit = train_df[train_df[self.TARGET] == 0]
        return legit[self.FEATURES].values

    def get_supervised_data(self, df, oversample=False):
        """Returns X and y, optionally with SMOTE oversampling."""
        X = df[self.FEATURES].values
        y = df[self.TARGET].values
        
        if oversample:
            sm = SMOTE(random_state=self.random_state)
            X, y = sm.fit_resample(X, y)
        
        return X, y
