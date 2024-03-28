import abc
import torch


class NoiseSampler(abc.ABC):

    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def sample(t: torch.Tensor):
        pass
