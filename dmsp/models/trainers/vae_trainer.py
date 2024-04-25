from typing import Any, Dict, List

import numpy as np
import hydra
import torch
import torch.utils.data.dataset

from dmsp.models.trainers.base_trainer import BaseTrainer
from dmsp.models.networks.vae import ConditionedVAE
from dmsp.utils.process_data import validate_traj_list, preprocess


class ConditionalVAETrainer(BaseTrainer):

    def __init__(
        self,
        lookback: int,
        vae: ConditionedVAE,
        optimizer_cls: str = "torch.optim.Adam",
        optimizer_kwargs: Dict[str, Any] | None = None,
        stream_data: bool = False,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.lookback = lookback

        self.device = torch.device(device)
        self.dtype = dtype
        self.stream_data = stream_data

        self.vae = vae.to(device=self.device)
        self.vae.set_device(self.device)

        self.optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            {
                "_target_": optimizer_cls,
                **({} if optimizer_kwargs is None else optimizer_kwargs),
            },
            params=self.vae.parameters(),
            _convert_="partial",
        )

    def validate_traj_list(
        self, trajectory_list: List[np.ndarray], sample_from_lookback: int = 0
    ) -> List[np.ndarray]:
        return validate_traj_list(
            trajectory_list=trajectory_list,
            lookback=self.lookback,
            sample_from_lookback=sample_from_lookback,
        )

    def preprocess(self, trajectory_list: List[np.ndarray]) -> torch.utils.data.Dataset:
        return preprocess(
            trajectory_list=trajectory_list,
            device=self.device,
            dtype=self.dtype,
            stream_data=self.stream_data,
            lookback=self.lookback,
            lookforward=1,
        )

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
        X = [X for _ in range(n_samples)]
        X = np.array(X)
        X = torch.tensor(X, device=self.device, dtype=self.dtype).swapaxes(
            0, 1
        )  # (n_traj, n_samples, lookback * d)

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
                yhat: torch.Tensor = self.vae.sample(
                    x=X, n_samples=n_samples
                )  # (n_traj, n_samples, d)
                samples[:, :, t, :] = yhat.detach().cpu().numpy()
                X[:, :, :-d] = X[:, :, d:]
                X[:, :, -d:] = yhat

        return list(samples.cumsum(axis=2)[:, :, 1:, :])

    def load_model(self, path: str) -> None:
        self.vae.load_state_dict(torch.load(path))

    def save_model(self, path: str) -> None:
        torch.save(self.vae.state_dict(), path)

    def train(
        self, train_batch: torch.Tensor | List[torch.Tensor], epoch: int
    ) -> Dict[str, float]:
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
