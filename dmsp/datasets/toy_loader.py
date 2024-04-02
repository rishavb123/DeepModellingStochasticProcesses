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
            np.arange(100) + np.random.normal(size=100),
            np.arange(3, 52) + 15 + np.random.normal(size=49),
            np.arange(-100, 0) + 12 + np.random.normal(size=100),
        ]


if __name__ == "__main__":
    loader = ToyLoader()
    loader.load()
    print(loader.data)
