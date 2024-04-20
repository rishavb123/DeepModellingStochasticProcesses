"""A toy data loader example to test the base loader and show how to use it."""

from typing import List

import numpy as np

from dmsp.datasets.base_loader import BaseLoader


class ToyLoader(BaseLoader):
    """A toy loader that counts up in each trajectory (with some noise)."""

    def __init__(self):
        """The constructor of the toy loader."""
        super().__init__(path=f"./data/counting/")

    def _download_data(self) -> List[np.ndarray]:
        """Constructs the data.

        Returns:
            List[np.ndarray]: The data.
        """
        np.random.seed(0)
        return [
            x.reshape((*x.shape, 1))
            for x in [
                np.arange(100) + np.random.normal(size=100),
                np.arange(3, 52) + 15 + np.random.normal(size=49),
                np.arange(-100, 0) + 12 + np.random.normal(size=100),
            ]
        ]

    def split_data(self, split_proportions: List[float]) -> List[List[np.ndarray]]:
        """Duplicates the data for train and test sets.

        Args:
            split_proportions (List[float]): The proportions to use (ignored for this loader).

        Returns:
            List[List[np.ndarray]]: The list of split datasets.
        """
        return [self.data for _ in split_proportions]


if __name__ == "__main__":
    loader = ToyLoader()
    loader.load()
    print(loader.data)
