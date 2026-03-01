import torch
import torch.nn as nn
from model.optimizer import ManualOptimizer
from torch.utils.data import DataLoader, TensorDataset

class Autoencoder(nn.Module):
    def __init__(self, input_dim=29, hidden_dims=[16, 8, 16]):
        super(Autoencoder, self).__init__()
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
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

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
                reconstructed = self.forward(X_batch)
                loss = self.criterion(reconstructed, X_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"AE Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.6f}")

    def predict_anomaly(self, X_test):
        self.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test)
            reconstructed = self.forward(X_test_tensor)
            # Use MSE per sample as anomaly score
            anomaly_score = torch.mean((X_test_tensor - reconstructed)**2, dim=1)
            return anomaly_score.numpy()
