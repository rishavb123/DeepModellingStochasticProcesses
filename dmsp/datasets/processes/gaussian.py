"""Data loader for a gaussian process"""

from typing import List

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, RBF
import matplotlib.pyplot as plt

from dmsp.datasets.base_loader import BaseLoader


class GaussianLoader(BaseLoader):

    def __init__(
        self,
        kernel: Kernel,
        n_traj: int = 100,
        traj_length: int = 100,
        step_size: float = 1.0,
    ) -> None:
        super().__init__(
            path=f"./data/processes/gaussian_{kernel.__class__.__name__.lower()}_{n_traj}_{traj_length}_{step_size}"
        )
        self.kernel = kernel
        self.n_traj = n_traj
        self.traj_length = traj_length

        self.gpr = GaussianProcessRegressor(kernel=kernel)
        self.kernel = kernel
        self.step_size = step_size

    def _download_data(self) -> List[np.ndarray]:
        """
        np.random.seed(0)
        data = np.concatenate(
            (
                np.zeros((self.n_traj, 1)) + self.initial_value,
                np.random.normal(
                    loc=self.mu, scale=self.std, size=(self.n_traj, self.traj_length)
                ),
            ),
            axis=1,
        ).cumsum(axis=1)
        return list(data)
        """

        # Generate input data
        X = np.linspace(0, self.traj_length * self.step_size, self.traj_length).reshape(
            -1, 1
        )

        # Sample from the Gaussian Process
        data = self.gpr.sample_y(X, n_samples=self.n_traj, random_state=0).T
        print(data.shape)
        return list(data)


if __name__ == "__main__":
    loader = GaussianLoader(kernel=RBF(), n_traj=10, traj_length=1000, step_size=0.01)
    loader.load()
    data = np.array(loader.data)
    import matplotlib.pyplot as plt

    plt.plot(data[0])
    plt.show()

    for i in range(loader.n_traj):
        plt.plot(data[i])
    plt.show()

    plt.plot(data.mean(axis=0))
    plt.show()
