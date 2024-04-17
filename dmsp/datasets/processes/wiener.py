"""Data loader for a weiner process"""

# from typing import

from typing import List
import numpy as np

from dmsp.datasets.base_loader import BaseLoader


class WienerLoader(BaseLoader):

    def __init__(
        self,
        mu: float,
        std: float,
        initial_value: float = 0,
        num_traj: int = 100,
        traj_length: int = 100,
    ) -> None:
        super().__init__(
            path=f"./data/processes/weiner_{mu}_{std}_{initial_value}_{num_traj}_{traj_length}"
        )
        self.mu = mu
        self.std = std
        self.initial_value = initial_value
        self.num_traj = num_traj
        self.traj_length = traj_length

    def _download_data(self) -> List[np.ndarray]:
        np.random.seed(0)
        data = np.concatenate(
            (
                np.zeros((self.num_traj, 1)) + self.initial_value,
                np.random.normal(
                    loc=self.mu, scale=self.std, size=(self.num_traj, self.traj_length)
                ),
            ),
            axis=1,
        ).cumsum(axis=1)
        return list(data)


class VolatileWienerLoader(WienerLoader):

    def __init__(
        self,
        mu: float,
        std: float,
        initial_value: float = 0,
        num_traj: int = 100,
        traj_length: int = 100,
        thresh: float = 1.0,
    ) -> None:
        super().__init__(mu, std, initial_value, num_traj, traj_length)
        self.thresh = thresh
        self.path = f"{self.path}_{self.thresh}"

    def _download_data(self) -> List[np.ndarray]:
        data = super()._download_data()
        for i in range(len(data)):
            diff = np.diff(np.concatenate(([0], data[i])))
            diff[np.abs(diff) > self.thresh] *= 2
            data[i] = diff.cumsum()
        return data


if __name__ == "__main__":
    loader = VolatileWienerLoader(
        mu=0.05,
        std=10,
        initial_value=0,
        num_traj=1000,
        traj_length=10000,
    )
    loader.load()
    data = np.array(loader.data)
    import matplotlib.pyplot as plt

    plt.plot(data[0])
    plt.show()

    for i in range(100):
        plt.plot(data[i])
    plt.show()

    plt.plot(data.mean(axis=0))
    plt.show()
