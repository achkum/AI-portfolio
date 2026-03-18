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
from sklearn.metrics import roc_curve, average_precision_score, roc_auc_score
from xgboost import XGBClassifier
from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def train_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict_anomaly(self, *args, **kwargs):
        pass

    @staticmethod
    def _print_class_distribution(X_train, y_train):
        n_legit = int((y_train == 0).sum())
        n_fraud = int((y_train == 1).sum())
        print(f"    Training on : {len(X_train):,} samples  (legit: {n_legit:,} | fraud: {n_fraud:,})")


class _AutoencoderBase(BaseModel, nn.Module):
    """Shared training loop infrastructure for Autoencoder and VAE."""

    @staticmethod
    def _make_dataloader(X_train, batch_size):
        return DataLoader(
            TensorDataset(torch.FloatTensor(X_train)),
            batch_size=batch_size, shuffle=True
        )

    @abstractmethod
    def _compute_loss(self, X_batch):
        pass

    @abstractmethod
    def _reconstruct(self, X):
        pass

    def _on_epoch_start(self, epoch):
        pass

    def _run_training_loop(self, dataloader, optimizer, epochs, loss_label):
        self.loss_history = []
        for epoch in range(epochs):
            self._on_epoch_start(epoch)
            total_loss = 0
            for (X_batch,) in dataloader:
                optimizer.zero_grad()
                loss = self._compute_loss(X_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            mean_loss = total_loss / len(dataloader)
            self.loss_history.append(mean_loss)
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch [{epoch+1:>2}/{epochs}]  {loss_label}: {mean_loss:.6f}")
        print(f"    Training complete — final loss: {mean_loss:.6f}")

    def predict_anomaly(self, X_test):
        self.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X_test)
            recon = self._reconstruct(X)
            return torch.mean((X - recon) ** 2, dim=1).numpy()


class Autoencoder(_AutoencoderBase):
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

    def _compute_loss(self, X_batch):
        return self.criterion(self.forward(X_batch), X_batch)

    def _reconstruct(self, X):
        return self.forward(X)

    def train_model(self, X_train, config):
        lr = config.get('learning_rate', 0.001)
        epochs = config.get('epochs', 50)
        batch_size = config.get('batch_size', 64)

        input_dim = X_train.shape[1]
        print(f"    Architecture : {input_dim} → 16 → 8 → 16 → {input_dim}")
        print(f"    Config       : lr={lr},  epochs={epochs},  batch_size={batch_size}")
        print(f"    Training on  : {len(X_train):,} samples (legitimate transactions only)")

        dataloader = self._make_dataloader(X_train, batch_size)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()
        self._run_training_loop(dataloader, optimizer, epochs, 'Mean MSE')


class VAE(_AutoencoderBase):
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
        # Deterministic inference: return the posterior mean only.
        # Stochastic sampling is intentionally omitted to produce stable,
        # reproducible anomaly scores at evaluation time.
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

    def _compute_loss(self, X_batch):
        recon, mu, logvar = self.forward(X_batch)
        return self.loss_function(recon, X_batch, mu, logvar)

    def _reconstruct(self, X):
        mu, _ = self.encode(X)
        return self.decode(mu)

    def _on_epoch_start(self, epoch):
        self.beta = min(self._beta_final, self._beta_final * (epoch + 1) / self._warmup_epochs)

    def train_model(self, X_train, config):
        lr = config.get('learning_rate', 0.001)
        epochs = config.get('epochs', 50)
        batch_size = config.get('batch_size', 64)

        input_dim = X_train.shape[1]
        latent_dim = self.fc_mean.out_features
        self._beta_final = self.beta
        self._warmup_epochs = 20
        print(f"    Architecture : {input_dim} → 16 → 8 → latent({latent_dim}) → 8 → 16 → {input_dim}")
        print(f"    Config       : lr={lr},  epochs={epochs},  batch_size={batch_size},  β={self._beta_final}  (warmup={self._warmup_epochs} epochs)")
        print(f"    Training on  : {len(X_train):,} samples (legitimate transactions only)")

        dataloader = self._make_dataloader(X_train, batch_size)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()
        self._run_training_loop(dataloader, optimizer, epochs, 'Mean MSE+KLD')


class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100, max_depth=30, min_samples_leaf=2,
                 random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, random_state=random_state,
            class_weight='balanced', n_jobs=-1
        )

    def train_model(self, X_train, y_train):
        print(f"    Params      : n_estimators={self.model.n_estimators},  max_depth={self.model.max_depth},  "
              f"min_samples_leaf={self.model.min_samples_leaf},  class_weight=balanced")
        self._print_class_distribution(X_train, y_train)
        self.model.fit(X_train, y_train)
        print(f"    Training complete.")

    def predict_anomaly(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]


class LogisticRegressionModel(BaseModel):
    def __init__(self, random_state=42, max_iter=1000):
        self.model = LogisticRegression(
            random_state=random_state, max_iter=max_iter,
            class_weight='balanced', solver='lbfgs'
        )

    def train_model(self, X_train, y_train):
        print(f"    Params      : solver={self.model.solver},  max_iter={self.model.max_iter},  class_weight=balanced")
        self._print_class_distribution(X_train, y_train)
        self.model.fit(X_train, y_train)
        print(f"    Converged in {self.model.n_iter_[0]} iterations.")

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
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        spw = neg / pos
        self.model.set_params(scale_pos_weight=spw)
        print(f"    Params      : n_estimators={self.model.n_estimators},  max_depth={self.model.max_depth},  "
              f"lr={self.model.learning_rate},  subsample={self.model.subsample},  colsample_bytree={self.model.colsample_bytree}")
        print(f"    Class weight: scale_pos_weight={spw:.1f}  (legit: {neg:,} | fraud: {pos:,})")
        self.model.fit(X_train, y_train)
        print(f"    Training complete.")

    def predict_anomaly(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]


class OneClassSVMModel(BaseModel):
    def __init__(self, kernel='rbf', gamma='scale', nu=0.01, max_samples=10000):
        self.model = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
        self.max_samples = max_samples

    def train_model(self, X_train):
        n_original = len(X_train)
        if self.max_samples and len(X_train) > self.max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_train), self.max_samples, replace=False)
            X_train = X_train[idx]
        print(f"    Params      : kernel={self.model.kernel},  gamma={self.model.gamma},  nu={self.model.nu}")
        print(f"    Training on : {len(X_train):,} samples (subsampled from {n_original:,} legitimate transactions)")
        self.model.fit(X_train)
        print(f"    Training complete.")

    def predict_anomaly(self, X_test):
        return -self.model.decision_function(X_test)


class HybridModel(BaseModel):
    def __init__(self, n_estimators=100, max_depth=30, min_samples_leaf=2,
                 random_state=42):
        self._rf_model = RandomForestModel(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, random_state=random_state
        )

    def train_model(self, X_train, y_train, scores_dict):
        stacked_features = list(scores_dict.keys())
        total_features = X_train.shape[1] + len(scores_dict)
        print(f"    Stacked scores : {stacked_features}")
        print(f"    Input shape    : {X_train.shape[1]} original + {len(scores_dict)} score features = {total_features} total")
        X_aug = np.hstack([X_train, np.column_stack(list(scores_dict.values()))])
        self._rf_model.train_model(X_aug, y_train)

    def predict_anomaly(self, X_test, scores_dict):
        X_aug = np.hstack([X_test, np.column_stack(list(scores_dict.values()))])
        return self._rf_model.predict_anomaly(X_aug)


class ModelEvaluator:
    def calculate_auprc(self, y_true, y_scores):
        return average_precision_score(y_true, y_scores)

    def calculate_roc_auc(self, y_true, y_scores):
        return roc_auc_score(y_true, y_scores)

    def calculate_recall_at_fpr(self, y_true, y_scores, fpr_threshold=0.01):
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        valid = np.where(fpr <= fpr_threshold)[0]
        idx = valid[-1] if len(valid) > 0 else 0
        return tpr[idx], thresholds[idx]

    def evaluate_all(self, y_true, model_scores_dict, fpr_thresholds=None):
        if fpr_thresholds is None:
            fpr_thresholds = [0.001, 0.005, 0.01]
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
