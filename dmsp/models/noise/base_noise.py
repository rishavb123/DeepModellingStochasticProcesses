"""File for the base noise sampler."""

import abc
import torch


class BaseNoise(abc.ABC):
    """Base noise class."""

    def __init__(self, noise_size: int) -> None:
        super().__init__()
        self.noise_size = noise_size

    @abc.abstractmethod
    def sample(
        self, n_samples: int = 1, device: torch.device | None = None
    ) -> torch.Tensor:
        pass
