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

    def validate_traj_lst(
        self, trajectory_list: List[np.ndarray], sample_from_lookback: int = 0
    ) -> List[np.ndarray]:
        return trajectory_list

    @abc.abstractmethod
    def sample(
        self,
        trajectory_list: List[np.ndarray],
        n_samples: int = 1,
        traj_length: int = 1,
        sample_from_lookback: int = 0,
    ) -> List[np.ndarray]:
        """Generates samples using the model on a list of trajectories.

        Args:
            trajectory_list (List[np.ndarray]): The list of trajectories to continue on.
            n_samples (int, optional): The numbers of samples to generate per trajectory. Defaults to 1.
            traj_length (int, optional): The length of each continuation trajectory to generate per sample. Defaults to 1.
            sample_from_lookback (int, optional): Starts the sampling from this many timesteps before the last realized step in the trajectories. Defaults to 0.

        Returns:
            List[np.ndarray]: The list of continuation trajectory samples for each trajectory in the input. Each ndarray should be of shape (n_samples, traj_length, d) where d is the dimension of each step.
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
    def save_model(self, path: str) -> None:
        """Saves the model to a specified path.

        Args:
            path (str): The path to save the model to.
        """
        pass

    @abc.abstractmethod
    def train(
        self, train_batch: torch.Tensor | List[torch.Tensor], epoch: int
    ) -> Dict[str, float]:
        """Trains the model using a batch of data.

        Args:
            train_batch (torch.Tensor | List[torch.Tensor]): The preprocessed data to train on.
            epoch (int): The current epoch number that this train batch is from.

        Returns:
            Dict[str, float]: Any training metrics to log.
        """
        return {}

    def eval(self, eval_batch: torch.Tensor | List[torch.Tensor]) -> Dict[str, float]:
        """The evaluation method of the model. Takes in a batch to perform evaluation on and then returns eval metrics.

        Args:
            eval_batch (torch.Tensor | List[torch.Tensor]): The batch of eval timeseries data.

        Returns:
            Dict[str, float]: Any evaluation metrics to log.
        """
        return {}
