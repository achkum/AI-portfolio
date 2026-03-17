import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import subprocess
import zipfile

KAGGLE_DATASET = 'mlg-ulb/creditcardfraud'

class CreditCardDataLoader:
    def __init__(self, test_size=0.2, random_state=42):
        self.scaler = StandardScaler()
        self.test_size = test_size
        self.random_state = random_state
        self.TARGET = 'Class'
        self.FEATURES = [f'V{i}' for i in range(1, 29)] + ['Amount']

    @staticmethod
    def download_from_kaggle(dest_dir):
        dest = Path(dest_dir)
        csv_path = dest / 'creditcard.csv'
        if csv_path.exists():
            print(f'Dataset already exists at {csv_path}')
            return csv_path

        print(f'Downloading dataset from Kaggle ({KAGGLE_DATASET})...')
        subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', KAGGLE_DATASET, '-p', str(dest)],
            check=True
        )
        zip_path = dest / 'creditcardfraud.zip'
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(dest)
            zip_path.unlink()
            print(f'Extracted to {csv_path}')
        return csv_path

    def load_and_split(self, raw_path):
        df = pd.read_csv(raw_path)

        if df.isnull().sum().sum() > 0:
            df = df.dropna()

        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            print(f'  Found {n_duplicates:,} duplicate rows — removing.')
            df = df.drop_duplicates().reset_index(drop=True)
        else:
            print('  No duplicate rows found.')

        self.eda_df = df.copy()

        train_df, test_df = train_test_split(
            df, test_size=self.test_size, random_state=self.random_state, stratify=df[self.TARGET]
        )

        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df[self.FEATURES] = self.scaler.fit_transform(train_df[self.FEATURES])
        test_df[self.FEATURES] = self.scaler.transform(test_df[self.FEATURES])

        train_fraud = train_df[self.TARGET].sum()
        test_fraud = test_df[self.TARGET].sum()
        print(f'  Split sizes — Total: {len(df):,} | '
              f'Train: {len(train_df):,} (fraud: {train_fraud:,}, '
              f'{100*train_fraud/len(train_df):.2f}%) | '
              f'Test: {len(test_df):,} (fraud: {test_fraud:,}, '
              f'{100*test_fraud/len(test_df):.2f}%)')

        return train_df, test_df

    def get_unsupervised_train(self, train_df):
        legit = train_df[train_df[self.TARGET] == 0]
        return legit[self.FEATURES].values

    def get_supervised_data(self, df):
        return df[self.FEATURES].values, df[self.TARGET].values
