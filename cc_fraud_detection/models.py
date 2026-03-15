import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, roc_curve


class Autoencoder(nn.Module):
    def __init__(self, input_dim=29, hidden_dims=[16, 8, 16]):
        super().__init__()
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
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-2)
        self.train()

        for epoch in range(epochs):
            total_loss = 0
            for (X_batch,) in dataloader:
                optimizer.zero_grad()
                loss = self.criterion(self.forward(X_batch), X_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"  AE Epoch [{epoch+1}/{epochs}]  Mean MSE: {total_loss/len(dataloader):.6f}")

    def predict_anomaly(self, X_test):
        self.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X_test)
            return torch.mean((X - self.forward(X))**2, dim=1).numpy()


class VAE(nn.Module):
    def __init__(self, input_dim=29, latent_dim=4, hidden_dims=[16, 8]):
        super().__init__()
        self.encoder_fc = nn.Linear(input_dim, hidden_dims[0])
        self.fc_mean = nn.Linear(hidden_dims[0], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[0], latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, hidden_dims[1])
        self.decoder_out = nn.Linear(hidden_dims[1], input_dim)

    def encode(self, x):
        h = F.relu(self.encoder_fc(x))
        return self.fc_mean(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        return self.decoder_out(F.relu(self.decoder_fc(z)))

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # Sum errors per sample, then average the total across the batch
        RE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (RE + KLD) / x.size(0)

    def train_model(self, X_train, config):
        lr = config.get('learning_rate', 0.001)
        epochs = config.get('epochs', 50)
        batch_size = config.get('batch_size', 64)

        dataloader = DataLoader(TensorDataset(torch.FloatTensor(X_train)), batch_size=batch_size, shuffle=True)
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-2)
        self.train()

        for epoch in range(epochs):
            total_loss = 0
            for (X_batch,) in dataloader:
                optimizer.zero_grad()
                recon, mu, logvar = self.forward(X_batch)
                loss = self.loss_function(recon, X_batch, mu, logvar)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                # Dividing by 29 to show average error per feature, similar to AE scale
                print(f"  VAE Epoch [{epoch+1}/{epochs}]  Mean Loss: {(total_loss/len(dataloader))/29:.6f}")

    def predict_anomaly(self, X_test):
        self.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X_test)
            recon, _, _ = self.forward(X)
            return torch.mean((X - recon)**2, dim=1).numpy()


class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)

    def train_model(self, X_train, y_train):
        self.rf.fit(X_train, y_train)

    def predict_anomaly(self, X_test):
        return self.rf.predict_proba(X_test)[:, 1]


class ModelEvaluator:
    def calculate_auprc(self, y_true, y_scores):
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        return auc(recall, precision)

    def calculate_recall_at_fpr(self, y_true, y_scores, fpr_threshold=0.01):
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        idx = np.argmin(np.abs(fpr - fpr_threshold))
        return tpr[idx], thresholds[idx]

    def evaluate_all(self, y_true, model_scores_dict, fpr_thresholds=[0.001, 0.005, 0.01]):
        results = []
        for model_name, y_scores in model_scores_dict.items():
            row = {'Model': model_name, 'AUPRC': self.calculate_auprc(y_true, y_scores)}
            for fpr_t in fpr_thresholds:
                recall, _ = self.calculate_recall_at_fpr(y_true, y_scores, fpr_t)
                row[f'Recall@{fpr_t*100:.1f}%FPR'] = recall
            results.append(row)
        return pd.DataFrame(results)
