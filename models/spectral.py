"""Variational autoencoder that learns MALDI spectral representations."""

import torch
from torch import nn


class SpectralAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 301, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 512), nn.GELU(), nn.Dropout(0.1),
                                     nn.Linear(512, 256), nn.GELU(), nn.Linear(256, latent_dim))
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 256), nn.GELU(),
                                     nn.Linear(256, 512), nn.GELU(), nn.Linear(512, input_dim))

    def encode(self, x: torch.Tensor):
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu
