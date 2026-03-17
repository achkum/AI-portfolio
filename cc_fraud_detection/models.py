import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from xgboost import XGBClassifier
from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def train_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict_anomaly(self, *args, **kwargs):
        pass


class Autoencoder(BaseModel, nn.Module):
    def __init__(self, input_dim=29, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 8, 16]
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], input_dim)
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def train_model(self, X_train, config):
        lr = config.get('learning_rate', 0.001)
        epochs = config.get('epochs', 50)
        batch_size = config.get('batch_size', 64)

        dataloader = DataLoader(TensorDataset(torch.FloatTensor(X_train)), batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()

        self.loss_history = []
        for epoch in range(epochs):
            total_loss = 0
            for (X_batch,) in dataloader:
                optimizer.zero_grad()
                loss = self.criterion(self.forward(X_batch), X_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            mean_loss = total_loss / len(dataloader)
            self.loss_history.append(mean_loss)
            if (epoch + 1) % 10 == 0:
                print(f"  AE Epoch [{epoch+1}/{epochs}]  Mean MSE: {mean_loss:.6f}")

    def predict_anomaly(self, X_test):
        self.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X_test)
            return torch.mean((X - self.forward(X))**2, dim=1).numpy()


class VAE(BaseModel, nn.Module):
    def __init__(self, input_dim=29, latent_dim=4, beta=0.5):
        super().__init__()
        self.beta = beta
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
        )
        self.fc_mean = nn.Linear(8, latent_dim)
        self.fc_logvar = nn.Linear(8, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8), nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mean(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        re = F.mse_loss(recon_x, x, reduction="none").sum(dim=1).mean()
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
        return re + self.beta * kld

    def train_model(self, X_train, config):
        lr = config.get('learning_rate', 0.001)
        epochs = config.get('epochs', 50)
        batch_size = config.get('batch_size', 64)

        dataloader = DataLoader(TensorDataset(torch.FloatTensor(X_train)), batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()

        beta_final = self.beta
        warmup_epochs = 20

        self.loss_history = []
        for epoch in range(epochs):
            self.beta = min(beta_final, beta_final * (epoch + 1) / warmup_epochs)
            total_loss = 0
            for (X_batch,) in dataloader:
                optimizer.zero_grad()
                recon, mu, logvar = self.forward(X_batch)
                loss = self.loss_function(recon, X_batch, mu, logvar)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            mean_loss = total_loss / len(dataloader)
            self.loss_history.append(mean_loss)
            if (epoch + 1) % 10 == 0:
                print(f"  VAE Epoch [{epoch+1}/{epochs}]  Mean MSE+KLD: {mean_loss:.6f}")

    def predict_anomaly(self, X_test):
        self.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X_test)
            mu, _ = self.encode(X)
            recon = self.decode(mu)
            return torch.mean((X - recon)**2, dim=1).numpy()


class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100, max_depth=30, min_samples_leaf=2,
                 random_state=42):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, random_state=random_state,
            class_weight='balanced', n_jobs=-1
        )

    def train_model(self, X_train, y_train):
        self.rf.fit(X_train, y_train)

    def predict_anomaly(self, X_test):
        return self.rf.predict_proba(X_test)[:, 1]


class LogisticRegressionModel(BaseModel):
    def __init__(self, random_state=42, max_iter=1000):
        self.model = LogisticRegression(
            random_state=random_state, max_iter=max_iter,
            class_weight='balanced', solver='lbfgs'
        )

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_anomaly(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]


class XGBoostModel(BaseModel):
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1,
                 subsample=0.8, colsample_bytree=0.8, random_state=42):
        self.model = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree, random_state=random_state,
            scale_pos_weight=None, eval_metric='logloss', n_jobs=1,
        )

    def train_model(self, X_train, y_train):
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        self.model.set_params(scale_pos_weight=neg / pos)
        self.model.fit(X_train, y_train)

    def predict_anomaly(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]


class OneClassSVMModel(BaseModel):
    def __init__(self, kernel='rbf', gamma='scale', nu=0.01, max_samples=10000):
        self.model = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
        self.max_samples = max_samples

    def train_model(self, X_train):
        if self.max_samples and len(X_train) > self.max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_train), self.max_samples, replace=False)
            X_train = X_train[idx]
        self.model.fit(X_train)

    def predict_anomaly(self, X_test):
        return -self.model.decision_function(X_test)


class HybridModel(BaseModel):
    def __init__(self, n_estimators=100, max_depth=30, min_samples_leaf=2,
                 random_state=42):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, random_state=random_state,
            class_weight='balanced', n_jobs=-1
        )

    def train_model(self, X_train, y_train, scores_dict):
        X_aug = np.hstack([X_train, np.column_stack(list(scores_dict.values()))])
        self.rf.fit(X_aug, y_train)

    def predict_anomaly(self, X_test, scores_dict):
        X_aug = np.hstack([X_test, np.column_stack(list(scores_dict.values()))])
        return self.rf.predict_proba(X_aug)[:, 1]


class ModelEvaluator:
    def calculate_auprc(self, y_true, y_scores):
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        return auc(recall, precision)

    def calculate_roc_auc(self, y_true, y_scores):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        return auc(fpr, tpr)

    def calculate_recall_at_fpr(self, y_true, y_scores, fpr_threshold=0.01):
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        valid = np.where(fpr <= fpr_threshold)[0]
        idx = valid[-1] if len(valid) > 0 else 0
        return tpr[idx], thresholds[idx]

    def evaluate_all(self, y_true, model_scores_dict, fpr_thresholds=[0.001, 0.005, 0.01]):
        results = []
        for model_name, y_scores in model_scores_dict.items():
            row = {
                'Model': model_name,
                'AUPRC': self.calculate_auprc(y_true, y_scores),
                'ROC-AUC': self.calculate_roc_auc(y_true, y_scores),
            }
            for fpr_t in fpr_thresholds:
                recall, _ = self.calculate_recall_at_fpr(y_true, y_scores, fpr_t)
                row[f'Recall@{fpr_t*100:.1f}%FPR'] = recall
            results.append(row)
        return pd.DataFrame(results)
