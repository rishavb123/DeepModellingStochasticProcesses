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
    dims_to_diff: List[bool] = None,
    lookforward: int = 1,
) -> torch.utils.data.Dataset:
    X = []
    y = []

    if not dims_to_diff:
        dims_to_diff = [ True ] * trajectory_list[0].shape[1]
    assert len(dims_to_diff) == trajectory_list[0].shape[1]

    """
    for traj in trajectory_list:
        for t in range(lookback + 1, traj.shape[0] - lookforward):
            res_X = []
            res_y = []
            for j in range(len(dims_to_diff)):
                if dims_to_diff[j]:
                    res_X.append(np.diff(traj[t - lookback - 1 : t, j], axis=0).flatten())
                    res_y.append(np.diff(traj[t - 1 : t + lookforward, j], axis=0).flatten())
                else:
                    res_X.append(traj[t - lookback : t, j].flatten())
                    res_y.append(traj[t : t + lookforward, j].flatten())
            X.append(np.concatenate(res_X))
            y.append(np.concatenate(res_y))
            # y.append(traj[t, :] - traj[t - 1, :])
    """
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
