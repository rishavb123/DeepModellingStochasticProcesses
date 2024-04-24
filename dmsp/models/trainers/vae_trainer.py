from typing import Dict, List

import numpy as np
from torch import Tensor
from torch.utils.data.dataset import Dataset

from dmsp.models.trainers.base_trainer import BaseTrainer


class VAETrainer(BaseTrainer):

    def __init__(self) -> None:
        super().__init__()

    def preprocess(self, trajectory_list: List[np.ndarray]) -> Dataset:
        return super().preprocess(trajectory_list)

    def validate_traj_lst(
        self, trajectory_list: List[np.ndarray], sample_from_lookback: int = 0
    ) -> List[np.ndarray]:
        return super().validate_traj_lst(trajectory_list, sample_from_lookback)

    def sample(
        self,
        trajectory_list: List[np.ndarray],
        n_samples: int = 1,
        traj_length: int = 1,
        sample_from_lookback: int = 0,
    ) -> List[np.ndarray]:
        return super().sample(
            trajectory_list, n_samples, traj_length, sample_from_lookback
        )

    def load_model(self, path: str) -> None:
        return super().load_model(path)

    def save_model(self, path: str) -> None:
        return super().save_model(path)

    def train(self, train_batch: Tensor | List[Tensor]) -> Dict[str, float]:
        return super().train(train_batch)

    def eval(self, eval_batch: Tensor | List[Tensor]) -> Dict[str, float]:
        return super().eval(eval_batch)
