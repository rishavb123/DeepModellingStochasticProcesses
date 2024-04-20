"""Python file for the base model trainer class to define the overall api for our models."""

from typing import Dict, List

import abc
import numpy as np
import torch
import torch.utils.data


class BaseTrainer(abc.ABC):
    """Abstract class for training stochastic time series sampling models."""

    def __init__(self) -> None:
        """Constructor for a model trainer"""
        super().__init__()

    @abc.abstractmethod
    def preprocess(self, trajectory_list: List[np.ndarray]) -> torch.utils.data.Dataset:
        """The data preprocessor for this model trainer.

        Args:
            trajectory_list (List[np.ndarray]): The list of trajectories.

        Returns:
            torch.utils.data.Dataset: The torch dataset compatible with this trainer.
        """
        pass

    @abc.abstractmethod
    def sample(
        self, batch_data: torch.Tensor, n_samples: int = 1, traj_length: int = 1
    ) -> torch.Tensor:
        """Generates samples using the underlying model on a batch of data.

        Args:
            batch_data (torch.Tensor): The preprocessed batch of data to condition on and generate future samples for.
            n_samples (int, optional): The number of samples to generate. Defaults to 1.
            traj_length (int, optional): The length of the trajectories to generate. Defaults to 1.

        Returns:
            torch.Tensor: The future trajectory samples.
        """
        pass

    @abc.abstractmethod
    def load_model(self, path: str) -> None:
        """Loads the model from a specified path.

        Args:
            path (str): The path to load the model from.
        """
        pass

    @abc.abstractmethod
    def train(self, train_batch: torch.Tensor) -> Dict[str, float]:
        """Trains the model using a batch of data.

        Args:
            train_batch (torch.Tensor): The preprocessed data to train on.

        Returns:
            Dict[str, float]: Any training metrics to log.
        """
        return {}

    @abc.abstractmethod
    def save_model(self, path: str) -> None:
        """Saves the model to a specified path.

        Args:
            path (str): The path to save the model to.
        """
        pass

    def eval(self, eval_batch: torch.Tensor) -> Dict[str, float]:
        """The evaluation method of the model. Takes in a batch to perform evaluation on and then returns eval metrics.

        Args:
            eval_batch (torch.Tensor): The batch of eval timeseries data.

        Returns:
            Dict[str, float]: Any evaluation metrics to log.
        """
        return {}
