from typing import Any, Dict, List

import numpy as np
import hydra
import torch
import torch.utils.data.dataset

from dmsp.models.trainers.base_trainer import BaseTrainer
from dmsp.models.networks.vae import ConditionedVAE


class ConditionalVAETrainer(BaseTrainer):

    def __init__(
        self,
        lookback: int,
        vae: ConditionedVAE,
        optimizer_cls: str = "torch.optim.Adam",
        optimizer_kwargs: Dict[str, Any] | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.lookback = lookback

        self.device = torch.device(device)
        self.dtype = dtype

        self.optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            {
                "_target_": optimizer_cls,
                **({} if optimizer_kwargs is None else optimizer_kwargs),
            },
            params=self.prediction_model.parameters(),
            _convert_="partial",
        )

        self.vae = vae.to(device=self.device)

    def preprocess(
        self, trajectory_list: List[np.ndarray]
    ) -> torch.utils.data.dataset.Dataset:
        X = []
        y = []

        for traj in trajectory_list:
            for t in range(self.lookback + 1, traj.shape[0]):
                X.append(np.diff(traj[t - self.lookback - 1 : t, :], axis=0).flatten())
                y.append(traj[t, :] - traj[t - 1, :])

        X = np.array(X)
        y = np.array(y)

        X = torch.tensor(X, device=self.device, dtype=self.dtype)
        y = torch.tensor(y, device=self.device, dtype=self.dtype)

        return torch.utils.data.TensorDataset(X, y)

    def validate_traj_lst(
        self, trajectory_list: List[np.ndarray], sample_from_lookback: int = 0
    ) -> List[np.ndarray]:
        return [
            traj
            for traj in trajectory_list
            if traj.shape[0] > self.lookback + sample_from_lookback
        ]

    def sample(
        self,
        trajectory_list: List[np.ndarray],
        n_samples: int = 1,
        traj_length: int = 1,
        sample_from_lookback: int = 0,
    ) -> List[np.ndarray]:
        n_traj = len(trajectory_list)
        d = trajectory_list[0].shape[1]

        if sample_from_lookback == 0:
            X = [
                np.diff(traj[-self.lookback - 1 :, :], axis=0).flatten()
                for traj in trajectory_list
            ]
        else:
            X = [
                np.diff(
                    traj[
                        -self.lookback
                        - 1
                        - sample_from_lookback : -sample_from_lookback,
                        :,
                    ],
                    axis=0,
                ).flatten()
                for traj in trajectory_list
            ]
        X = np.array(X)
        X = torch.tensor(X, device=self.device, dtype=self.dtype).swapaxes(
            0, 1
        )  # (n_traj, lookback * d)

        samples = np.zeros((n_traj, n_samples, 1 + traj_length, d))

        for i, traj in enumerate(trajectory_list):
            if sample_from_lookback == 0:
                samples[i, :, 0, :] = np.repeat(traj[-1:, :], repeats=n_samples, axis=0)
            else:
                samples[i, :, 0, :] = np.repeat(
                    traj[-1 - sample_from_lookback : -sample_from_lookback, :],
                    repeats=n_samples,
                    axis=0,
                )

        for t in range(1, 1 + traj_length):
            with torch.no_grad():
                yhat: torch.Tensor = self.vae.sample(x=X)  # (n_traj, n_samples, d)
                samples[:, :, t, :] = yhat.detach().cpu().numpy()
                X[:, :, :-d] = X[:, :, d:]
                X[:, :, -d:] = yhat

        return list(samples.cumsum(axis=2)[:, :, 1:, :])

    def load_model(self, path: str) -> None:
        self.vae.load_state_dict(torch.load(path))

    def save_model(self, path: str) -> None:
        torch.save(self.vae.state_dict(), path)

    def train(self, train_batch: torch.Tensor | List[torch.Tensor]) -> Dict[str, float]:
        self.optimizer.zero_grad()

        X, y = train_batch

        loss = self.vae.loss(x=X, y=y)
        loss.backward()
        self.optimizer.step()

        return {"train/loss": loss.item()}

    def eval(self, eval_batch: torch.Tensor | List[torch.Tensor]) -> Dict[str, float]:
        with torch.no_grad():
            X, y = eval_batch
            loss = self.vae.loss(x=X, y=y)
        return {"eval/loss": loss.item()}
