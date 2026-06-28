"""
autoencoder.py
Deep autoencoder for latent factor extraction using PyTorch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        """
        input_dim: number of assets
        hidden_dims: list of hidden layer sizes
        latent_dim: size of latent space
        """
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)

        dec_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(prev_dim, h))
            dec_layers.append(nn.ReLU())
            prev_dim = h
        dec_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
