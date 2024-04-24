"""File for numpy dataset."""

from typing import Tuple

import numpy as np
import torch

from torch.utils.data import Dataset

class NumpyDataset(Dataset):
    """Dataset for numpy array to load into gpu memory as necessary."""

    def __init__(
        self, X: np.ndarray, y: np.ndarray, device: str, dtype: torch.dtype
    ) -> None:
        """Constructor for numpy dataset.

        Args:
            X (np.ndarray): Input data (n_examples, d * lookback).
            y (np.ndarray): Label data (n_examples, d).
            device (str): Device to send tensors to.
            dtype (torch.dtype): Data type of tensors.
        """
        self.X = X
        self.y = y
        self.device = device
        self.dtype = dtype

    def __len__(self) -> int:
        """Returns length of dataset.

        Returns:
            int: Length of dataset.
        """
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns item from dataset at index.

        Args:
            idx (int): Index of the dataset.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input and label data.
        """
        X = torch.tensor(self.X[idx, :], device=torch.device('cpu'), dtype=self.dtype)
        y = torch.tensor(self.y[idx, :], device=torch.device('cpu'), dtype=self.dtype)
        return X, y
