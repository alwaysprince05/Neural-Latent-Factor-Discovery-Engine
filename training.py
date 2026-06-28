"""
training.py
Training utilities for autoencoder model.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from autoencoder import Autoencoder

def train_autoencoder(X, input_dim, hidden_dims, latent_dim, epochs=100, batch_size=32, lr=1e-3, val_split=0.2, device='cpu'):
    """
    Train autoencoder on standardized returns.
    Returns: trained model, train/val losses, latent factors
    """
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = Autoencoder(input_dim, hidden_dims, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            x_hat, _ = model(x)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= train_size
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                x_hat, _ = model(x)
                loss = criterion(x_hat, x)
                val_loss += loss.item() * x.size(0)
        val_loss /= val_size
        val_losses.append(val_loss)

    # Extract latent factors for all data
    model.eval()
    with torch.no_grad():
        _, latents = model(X_tensor.to(device))
    return model, train_losses, val_losses, latents.cpu().numpy()
