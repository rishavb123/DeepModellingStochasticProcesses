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
    ) -> None:
        """Constructor for the yfinance loader.

        Args:
            symbols (List[str]): The list of symbols to include in the dataset.
            download_kwargs (List[Dict[str, Any]]): The kwargs to use for each trajectory.
            columns (List[str] | None, optional): The columns from the yfinance return to use. Defaults to None.
        """
        super().__init__(
            f"./data/finance/{'_'.join(symbols)}__{'__'.join([f'{k}_{kwargs[k]}' for kwargs in download_kwargs for k in kwargs])}/"
        )
        self.symbols = symbols
        self.download_kwargs = download_kwargs
        self.columns = ["Adj Close", "Volume"] if columns is None else columns

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
    )
    loader.load()
    for d in loader.data:
        print(d.shape)
