"""File for the base noise sampler."""

import abc
import torch


class BaseNoise(abc.ABC):
    """Base noise class."""

    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def sample(self) -> torch.Tensor:
        pass
