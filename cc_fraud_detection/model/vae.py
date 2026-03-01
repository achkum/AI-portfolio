import torch
import torch.nn as nn
import torch.nn.functional as F
from model.optimizer import ManualOptimizer
from torch.utils.data import DataLoader, TensorDataset

class VAE(nn.Module):
    def __init__(self, input_dim=29, latent_dim=4, hidden_dims=[16, 8]):
        super(VAE, self).__init__()
        self.encoder_fc = nn.Linear(input_dim, hidden_dims[0])
        self.fc_mean = nn.Linear(hidden_dims[0], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[0], latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, hidden_dims[1])
        self.decoder_out = nn.Linear(hidden_dims[1], input_dim)

    def encode(self, x):
        h = F.relu(self.encoder_fc(x))
        return self.fc_mean(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.decoder_fc(z))
        return self.decoder_out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction Loss (MSE) + KL Divergence
        RE = F.mse_loss(recon_x, x, reduction='mean') # Mean per sample
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return RE + (KLD / x.size(0)) # Normalizing KLD by batch size

    def train_model(self, X_train, config):
        lr = config.get('learning_rate', 0.001)
        epochs = config.get('epochs', 50)
        batch_size = config.get('batch_size', 64)

        X_train_tensor = torch.FloatTensor(X_train)
        dataset = TensorDataset(X_train_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = ManualOptimizer(self.parameters(), lr=lr)
        self.train()

        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (X_batch,) in enumerate(dataloader):
                optimizer.zero_grad()
                recon_batch, mu, logvar = self.forward(X_batch)
                loss = self.loss_function(recon_batch, X_batch, mu, logvar)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"VAE Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.6f}")

    def predict_anomaly(self, X_test):
        self.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test)
            recon_batch, mu, logvar = self.forward(X_test_tensor)
            # Use MSE as reconstruction-based anomaly score
            anomaly_score = torch.mean((X_test_tensor - recon_batch)**2, dim=1)
            return anomaly_score.numpy()
