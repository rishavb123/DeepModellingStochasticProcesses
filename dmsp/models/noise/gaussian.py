"""Noise sampler for standard gaussian noise."""

import torch

from dmsp.models.noise.base_noise import BaseNoise


class GaussianNoise(BaseNoise):

    def __init__(self, noise_size: int) -> None:
        super().__init__(noise_size)

    def sample(self) -> torch.Tensor:
        return torch.randn(size=(self.noise_size,))
