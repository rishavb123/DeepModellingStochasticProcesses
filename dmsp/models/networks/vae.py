from typing import Any, Tuple

import torch
import torch.nn as nn


class VAE(nn.Module):

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        reconstruction_loss_f: nn.Module | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

        self.reconstruction_loss_f = (
            nn.BCELoss() if reconstruction_loss_f is None else reconstruction_loss_f
        )

    def set_device(self, device: str | None):
        self.device = None if device is None else torch.device(device=device)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.reconstruction_loss_f = self.reconstruction_loss_f.to(self.device)

    def reparameterization(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        epsilon = torch.randn_like(mean, device=self.device)
        return mean + std * epsilon

    def sample_noise(self, n_samples: int = 1) -> torch.Tensor:
        return torch.randn((n_samples, self.latent_dim), device=self.device)

    def sample(self, n_samples: int = 1):
        with torch.no_grad():
            noise = self.sample_noise(n_samples=n_samples)
            return self.decoder(noise)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        x_hat, mean, logvar = self.forward(x=x)
        reconstruction_loss = self.reconstruction_loss_f(x_hat, x)
        kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return reconstruction_loss + kl_divergence_loss

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        z = self.reparameterization(mean=mean, std=std)
        x_hat = self.decoder(z)
        return x_hat, mean, logvar


class ConditionedVAE(VAE):

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        reconstruction_loss_f: nn.Module | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            encoder, decoder, latent_dim, reconstruction_loss_f, *args, **kwargs
        )

    def sample_noise(self, n_samples: int = 1) -> torch.Tensor:
        return torch.randn((n_samples, self.latent_dim), device=self.device)

    def sample(self, x: torch.Tensor, n_samples: int = 1):
        batch_size = x.shape[0]
        with torch.no_grad():
            if len(x.shape) == 2:
                x = x.repeat((n_samples, 1, 1)).swapaxes(
                    0, 1
                )  # (batch_size, n_samples, lookback * d)
            noise = self.sample_noise(n_samples=batch_size * n_samples).reshape(
                (batch_size, n_samples, self.latent_dim)
            )  # (batch_size, n_samples, latent_dim)
            return self.decoder((noise, x))  # (batch_size, n_samples, d)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_hat, mean, logvar = self.forward(x=x, y=y)
        reconstruction_loss = self.reconstruction_loss_f(y_hat, y)
        kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
        return reconstruction_loss + kl_divergence_loss

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encoder((y, x))
        std = torch.exp(0.5 * logvar)
        z = self.reparameterization(mean=mean, std=std)
        y_hat = self.decoder((z, x))
        return y_hat, mean, logvar
