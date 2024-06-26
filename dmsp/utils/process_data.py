from typing import List
import numpy as np
import torch
from dmsp.utils.numpy_dataset import NumpyDataset


def validate_traj_list(
    trajectory_list: List[np.ndarray], lookback: int, sample_from_lookback: int = 0
) -> List[np.ndarray]:
    return [
        traj
        for traj in trajectory_list
        if traj.shape[0] > lookback + sample_from_lookback
    ]


def preprocess(
    trajectory_list: List[np.ndarray],
    device,
    dtype,
    stream_data,
    lookback: int,
    lookforward: int = 1,
) -> torch.utils.data.Dataset:
    X = []
    y = []

    for traj in trajectory_list:
        for t in range(lookback + 1, traj.shape[0] - lookforward):
            X.append(np.diff(traj[t - lookback - 1 : t, :], axis=0).flatten())
            y.append(np.diff(traj[t - 1 : t + lookforward, :], axis=0).flatten())
            # y.append(traj[t, :] - traj[t - 1, :])

    X = np.array(X)
    y = np.array(y)

    X = torch.tensor(X, device=device, dtype=dtype)  # (n_examples, lookback * d)
    y = torch.tensor(y, device=device, dtype=dtype)  # (n_examples, d)

    if stream_data:
        return NumpyDataset(X, y, device, dtype)

    else:
        return torch.utils.data.TensorDataset(X, y)
