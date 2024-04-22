"""Noise sampler for standard gaussian noise."""

import torch

from dmsp.models.noise.base_noise import BaseNoise


class GaussianNoise(BaseNoise):

    def __init__(self, noise_size: int) -> None:
        super().__init__(noise_size)

    def sample(
        self, num_samples: int = 1, device: torch.device | None = None
    ) -> torch.Tensor:
        return torch.randn(size=(num_samples, self.noise_size), device=device)
