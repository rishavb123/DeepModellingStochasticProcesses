"""Data loader for the solar power production dataset on kaggle: https://www.kaggle.com/datasets/pythonafroz/daily-power-production-data-of-solar-power-plant/data"""

from typing import List

import numpy as np
import pandas as pd
import opendatasets as od

from dmsp.datasets.base_loader import BaseLoader


class SolarPowerProductionLoader(BaseLoader):

    URL = "https://www.kaggle.com/datasets/pythonafroz/daily-power-production-data-of-solar-power-plant/data"
    NAME = "daily-power-production-data-of-solar-power-plant"
    FNAME = "Solar_Energy_Production.csv"

    KEEP_UNTIL_DATE = "2023-03-05 16:00:00"
    REMOVE_IDS = [594148]

    def __init__(self) -> None:
        """Constructor fo rhte solar power production loader."""
        super().__init__(f"./data/behavior/solar_power_production")

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
        od.download_kaggle_dataset(
            SolarPowerProductionLoader.URL, data_dir=f"{self.path}"
        )

        df = pd.read_csv(
            f"{self.path}/{SolarPowerProductionLoader.NAME}/{SolarPowerProductionLoader.FNAME}"
        )[["id", "date", "kWh"]]
        df = df[~df["id"].isin(SolarPowerProductionLoader.REMOVE_IDS)]
        df["date"] = pd.to_datetime(df["date"])
        df = df.pivot_table(index="date", columns="id", values="kWh", fill_value=None).ffill(axis=1).bfill(axis=1)
        df = df[df.index < "2023-03-05 16:00:00"]

        data = df.values
        data -= data.mean()
        data /= data.std()

        return [data]


if __name__ == "__main__":
    loader = SolarPowerProductionLoader()
    loader.load(force_redownload=True)

    import matplotlib.pyplot as plt

    plt.plot(loader.data[0][:, 0])
    plt.show()
