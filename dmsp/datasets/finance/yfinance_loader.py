"""Data loader for the yfinance module"""

from typing import Any, Dict, List

import numpy as np
import yfinance as yf

from dmsp.datasets.base_loader import BaseLoader


class YFinanceLoader(BaseLoader):
    """Yahoo Finance data loader class."""

    def __init__(
        self,
        symbols: List[str],
        download_kwargs: List[Dict[str, Any]],
        columns: List[str] | None = None,
        normalize_by_first_price: bool = False,
    ) -> None:
        """Constructor for the yfinance loader.

        Args:
            symbols (List[str]): The list of symbols to include in the dataset.
            download_kwargs (List[Dict[str, Any]]): The kwargs to use for each trajectory.
            columns (List[str] | None, optional): The columns from the yfinance return to use. Defaults to None.
            normalize_by_first_price (bool, optional): Whether or not to normalize the data by the first price downloaded. Defaults to False.
        """
        self.symbols = symbols
        self.columns = ["Adj Close", "Volume"] if columns is None else columns
        feature_names = []
        for column in self.columns:
            for symbol in symbols:
                feature_names.append(f"{symbol} {column}")

        super().__init__(
            path=f"./data/finance/{'_'.join(symbols)}__{'__'.join([f'{k}_{kwargs[k]}' for kwargs in download_kwargs for k in kwargs])}__{'__'.join([c.replace(' ', '_') for c in columns])}_{normalize_by_first_price}/",
            feature_names=feature_names
        )
        self.download_kwargs = download_kwargs
        self.normalize_by_first_price = normalize_by_first_price

    def _download_data(self) -> List[np.ndarray]:
        """Downloads the dataset using the yfinance download function.

        Returns:
            List[np.ndarray]: The dataset.
        """
        data = []
        for download_kwargs in self.download_kwargs:
            df = yf.download(
                self.symbols, **download_kwargs
            )  # Download data from yfinance
            df = df[self.columns]  # Extract price and volume information
            df = df / df.iloc[0]
            data.append(df.to_numpy())
        return data


if __name__ == "__main__":
    loader = YFinanceLoader(
        symbols=["SPY", "AMZN", "QQQ", "MSFT"],
        download_kwargs=[
            {"start": "2022-01-01", "end": "2022-12-31"},
            {"start": "2019-01-01", "end": "2021-12-31"},
            {"start": "2023-01-01", "end": "2023-12-31"},
        ],
        columns=["Adj Close"],
        normalize_by_first_price=True,
    )
    import pdb; pdb.set_trace()
    loader.load()
    for d in loader.data:
        print(d.shape)

    import matplotlib.pyplot as plt

    data = loader.data[0]

    for j in range(4):
        plt.plot(data[:, j])

    plt.show()
