from typing import List

import abc
import os
import numpy as np


class BaseLoader(abc.ABC):

    def __init__(self, path: str) -> None:
        self.path = path
        self.data: List[np.ndarray] | None = None

    def load(self) -> None:
        if os.path.exists(self.path):
            self.data = BaseLoader.read_from_path(self.path)
        else:
            self.data = self._download_data()
            self.save_to_path(self.path, self.data)

    @abc.abstractmethod
    def _download_data(self) -> List[np.ndarray]:
        pass

    @staticmethod
    def read_from_path(path: str) -> List[np.ndarray]:
        return [np.load(f"{path}/{fname}") for fname in sorted(os.listdir(path=path))]

    @staticmethod
    def save_to_path(path: str, data: List[np.ndarray] | None) -> None:
        os.makedirs(path)
        if data is None:
            return
        for i, traj in enumerate(data):
            np.save(f"{path}/trajectory_{i}.npy", traj)
