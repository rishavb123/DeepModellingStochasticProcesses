"""Data loader for the yfinance module"""

from typing import List, Tuple

import numpy as np
import yfinance as yf

from dmsp.datasets.base_loader import BaseLoader


class YFinanceLoader(BaseLoader):
    """Yahoo Finance data loader class."""

    def __init__(
        self,
        symbols: List[str],
        intervals: List[Tuple[str, str]],
        columns: List[str] | None = None,
    ) -> None:
        super().__init__(
            f"./data/finance/{'_'.join(symbols)}_{'_'.join([f'{start}to{end}' for (start, end) in intervals])}/"
        )
        self.symbols = symbols
        self.intervals = intervals
        self.columns = ["Adj Close", "Volume"] if columns is None else columns

    def _download_data(self) -> List[np.ndarray]:
        data = []
        for start, end in self.intervals:  # Loop through each interval
            df = yf.download(
                self.symbols, start=start, end=end
            )  # Download data from yfinance
            df = df[self.columns]  # Extract price and volume information
            data.append(df.to_numpy())
        return data


if __name__ == "__main__":
    loader = YFinanceLoader(
        symbols=["SPY", "AAPL"],
        intervals=[
            ("2022-01-01", "2022-12-31"),
            ("2019-01-01", "2021-12-31"),
            ("2023-01-01", "2023-12-31"),
        ],
    )
    loader.load()
    print(loader.data)
    for d in loader.data:
        print(d.shape)
