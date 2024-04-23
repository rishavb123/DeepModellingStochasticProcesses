"""The base data loader for the timeseries experiments."""

from typing import List

import abc
import os
import shutil
import glob
import numpy as np


class BaseLoader(abc.ABC):
    """The BaseLoader class for all timeseries datasets used in this project."""

    def __init__(self, path: str) -> None:
        """The constructor for the data loader.

        Args:
            path (str): The path to load/save this data set from/to.
        """
        self.path = path
        self.data: List[np.ndarray] | None = None

    def load(self, force_redownload: bool = False) -> None:
        """Loads the dataset into memory (either from the disk or using the _download data function). Stores the data in self.data.

        Args:
            force_redownload (bool, optional): Whether or not to force a redownload of the data. Defaults to False.
        """
        if os.path.exists(self.path) and not force_redownload:
            self.data = BaseLoader.read_from_path(self.path)
        else:
            if os.path.exists(self.path):
                shutil.rmtree(self.path)
            self.data = self._download_data()
            self.save_to_path(self.path, self.data)

    @abc.abstractmethod
    def _download_data(self) -> List[np.ndarray]:
        """Downloads the dataset from the internet or manually constructs it.

        Returns:
            List[np.ndarray]: The dataset.
        """
        pass

    def split_data(self, split_proportions: List[float]) -> List[List[np.ndarray]]:
        """Splits the data into the proportions specified by the input while maintaining an ordering.

        Args:
            split_proportions (List[float]): The split proportions (ordered by older data to newer data).

        Returns:
            List[List[np.ndarray]]: The data split as specified.
        """

        total_length = 0
        for traj in self.data:
            total_length += traj.shape[0]

        res = []
        current_set = []
        cum_prop = split_proportions[0]
        prop_ind = 0
        begin_prop = 0
        for traj in self.data:
            end_prop = begin_prop + traj.shape[0] / total_length

            if cum_prop >= begin_prop and cum_prop < end_prop and end_prop < 1.0:
                split_ind = int(
                    len(traj) * (cum_prop - begin_prop) / (end_prop - begin_prop)
                )
                current_set.append(traj[:split_ind, :])
                res.append(current_set)

                current_set = []
                current_set.append(traj[split_ind:, :])

                prop_ind += 1
                cum_prop += split_proportions[prop_ind]
            else:
                current_set.append(traj)

            begin_prop = end_prop
        res.append(current_set)

        return res

    @staticmethod
    def read_from_path(path: str) -> List[np.ndarray]:
        """Reads a dataset from a specified path.

        Args:
            path (str): The path to read the dataset from.

        Returns:
            List[np.ndarray]: The dataset.
        """
        return [
            np.load(str(fpath)) for fpath in sorted(glob.glob(f"{path}/trajectory_*"))
        ]

    @staticmethod
    def save_to_path(path: str, data: List[np.ndarray] | None) -> None:
        """Saves some data to a specified path.

        Args:
            path (str): The path to save to.
            data (List[np.ndarray] | None): The data to save.
        """
        os.makedirs(path, exist_ok=True)
        if data is None:
            return
        for i, traj in enumerate(data):
            np.save(f"{path}/trajectory_{i}.npy", traj)
