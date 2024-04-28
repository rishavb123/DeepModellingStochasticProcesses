"""Data loader for the solar power production dataset on kaggle: https://www.kaggle.com/datasets/pythonafroz/daily-power-production-data-of-solar-power-plant/data"""

from typing import List

import numpy as np
import pandas as pd
import opendatasets as od

import os

from dmsp.datasets.base_loader import BaseLoader


class LakeLoader(BaseLoader):

    URL = "https://www.kaggle.com/competitions/acea-water-prediction/data"
    NAME = "lake-characteristics/acea-water-prediction"
    FNAME = "Lake_Bilancino.csv"

    # KEEP_UNTIL_DATE = "2023-03-05 16:00:00"
    # REMOVE_IDS = [594148]
    REMOVE_IDS = []

    def __init__(self) -> None:
        """Constructor for the lake characteristics dataset"""
        super().__init__(f"./data/physical/lake-characteristics")

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
        if not os.path.exists(self.path):
            od.download_kaggle_dataset(
                LakeLoader.URL, data_dir=f"{self.path}"
            )

        # print(self.path)
        # print(LakeLoader.NAME)
        # print(LakeLoader.FNAME)
        df = pd.read_csv(
            # f"{self.path}/{LakeLoader.FNAME}"
            "./data/physical/lake-characteristics/acea-water-prediction/Lake_Bilancino.csv"
        )
        # df = df[["id", "date", "kWh"]]
        # df = df[~df["id"].isin(LakeLoader.REMOVE_IDS)]
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
        df = df.pivot_table(index="Date")
        """
        df = (
            df.pivot_table(index="date", columns="id", values="kWh", fill_value=None)
            .ffill(axis=1)
            .bfill(axis=1)
        )
        df = df[df.index < "2023-03-05 16:00:00"]
        """
        df = df[df.index > '2004-01-01']

        data = df.values
        data -= data.mean(axis=0) * (data.min(axis=0) != 0)
        data /= data.std(axis=0)

        return [data]


if __name__ == "__main__":
    loader = LakeLoader()
    loader.load(force_redownload=False)

    import matplotlib.pyplot as plt

    print(loader.data[0].shape)
    plt.plot(loader.data[0][:, 1])
    plt.show()
