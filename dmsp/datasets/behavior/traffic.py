"""Data loader for traffic dataset on kaggle: https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset"""

from typing import List

import numpy as np
import pandas as pd
import opendatasets as od

from dmsp.datasets.base_loader import BaseLoader


class TrafficLoader(BaseLoader):

    URL = "https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset"
    NAME = "daily-power-production-data-of-solar-power-plant"  # TODO: fix this
    FNAME = "Solar_Energy_Production.csv"  # TODO: fix this

    def __init__(self) -> None:
        """Constructor fo rhte solar power production loader."""
        super().__init__(f"./data/behavior/traffic_dataset")

    def split_data(self, split_proportions: List[float]) -> List[List[np.ndarray]]:
        n = self.data[0].shape[0]
        splits = []
        last_index = 0
        for proportion in split_proportions:
            new_index = last_index + int(n * proportion)
            splits.append([self.data[0][last_index:new_index]])
            last_index = new_index
        return splits

    def _download_data(self) -> List[np.ndarray]:
        """Downloads and loads in the data using the open datasets python module.

        Returns:
            List[np.ndarray]: The dataset.
        """
        od.download_kaggle_dataset(TrafficLoader.URL, data_dir=f"{self.path}")

        df = pd.read_csv(f"{self.path}/{TrafficLoader.NAME}/{TrafficLoader.FNAME}")[
            ["id", "date", "kWh"]
        ]

        # TODO: finish this
        import sys

        sys.exit(1)

        return [df.values]


if __name__ == "__main__":
    loader = TrafficLoader()
    loader.load(force_redownload=True)

    import matplotlib.pyplot as plt

    plt.plot(loader.data[0][:, 0])
    plt.show()
