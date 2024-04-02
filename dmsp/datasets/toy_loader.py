import numpy as np

from dmsp.datasets.base_loader import BaseLoader


class ToyLoader(BaseLoader):

    def __init__(self):
        super().__init__(path=f"./data/counting/")

    def get_data(self):
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
