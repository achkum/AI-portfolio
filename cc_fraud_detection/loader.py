import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

class CreditCardDataLoader:
    def __init__(self, test_size=0.2, random_state=42):
        self.scaler = StandardScaler()
        self.test_size = test_size
        self.random_state = random_state
        self.TARGET = 'Class'
        self.FEATURES = [f'V{i}' for i in range(1, 29)] + ['Amount']

    def load_and_split(self, raw_path):
        df = pd.read_csv(raw_path)

        if df.isnull().sum().sum() > 0:
            df = df.dropna()

        self.eda_df = df.copy()

        train_df, test_df = train_test_split(
            df, test_size=self.test_size, random_state=self.random_state, stratify=df[self.TARGET]
        )

        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df[self.FEATURES] = self.scaler.fit_transform(train_df[self.FEATURES])
        test_df[self.FEATURES] = self.scaler.transform(test_df[self.FEATURES])

        return train_df, test_df

    def get_unsupervised_train(self, train_df):
        legit = train_df[train_df[self.TARGET] == 0]
        return legit[self.FEATURES].values

    def get_supervised_data(self, df):
        return df[self.FEATURES].values, df[self.TARGET].values
