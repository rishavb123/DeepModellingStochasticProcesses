from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import hydra
import logging
import inspect

from dmsp.models.trainers.base_trainer import BaseTrainer
from dmsp.models.noise.base_noise import BaseNoise
from dmsp.utils.process_data import validate_traj_list, preprocess


class SampleDensityMLETrainer(BaseTrainer):

    def __init__(
        self,
        lookback: int,
        prediction_model: nn.Module,
        noise_model: BaseNoise,
        optimizer_cls: str = "torch.optim.Adam",
        optimizer_kwargs: Dict[str, Any] | None = None,
        k: int = 1,
        n_train_generated_samples: int = 30,
        use_log_loss_for_backprop: bool = True,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        stream_data: bool = True,
    ) -> None:
        super().__init__()

        self.lookback = lookback

        self.device = torch.device(device)
        self.dtype = dtype
        self.prediction_model = prediction_model.to(device=self.device)
        self.noise_model = noise_model

        self.optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            {
                "_target_": optimizer_cls,
                **({} if optimizer_kwargs is None else optimizer_kwargs),
            },
            params=self.prediction_model.parameters(),
            _convert_="partial",
        )

        self.k = k
        self.n_train_generated_samples = n_train_generated_samples

        self.use_log_loss_for_backprop = use_log_loss_for_backprop
        self.mse_loss = torch.nn.MSELoss()

        self.stream_data = stream_data

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
                noise = self.noise_model.sample(
                    n_samples=n_traj * n_samples,
                    device=self.device,
                )  # (n_traj * n_samples, noise_size)

                yhat: torch.Tensor = self.prediction_model(
                    (noise, X.reshape((-1, X.shape[-1])))
                ).reshape(
                    (n_traj, n_samples, d)
                )  # (n_traj, n_samples, d)
                samples[:, :, t, :] = yhat.detach().cpu().numpy()
                X[:, :, :-d] = X[:, :, d:]
                X[:, :, -d:] = yhat

        return list(samples.cumsum(axis=2)[:, :, 1:, :])

    def load_model(self, path: str) -> None:
        self.prediction_model.load_state_dict(torch.load(path))

    def save_model(self, path: str) -> None:
        torch.save(self.prediction_model.state_dict(), path)

    def calculate_loss(
        self, batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        X, y = batch  # (batch_size, lookback * d), (batch_size, d)

        batch_size, d = y.shape

        X = X.unsqueeze(1).repeat(
            (1, self.n_train_generated_samples, 1)
        )  # (batch_size, n_samples, lookback * d)

        noise = self.noise_model.sample(
            n_samples=batch_size * self.n_train_generated_samples,
            device=self.device,
        )  # (batch_size * n_samples, noise_size)

        yhats: torch.Tensor = self.prediction_model(
            (noise, X.reshape((-1, X.shape[-1])))
        ).reshape(
            (batch_size, self.n_train_generated_samples, d)
        )  # (batch_size, n_samples, d)

        distances = torch.norm(
            yhats - y.unsqueeze(1), dim=-1
        )  # (batch_size, n_samples)
        kth_closest_indices = torch.topk(distances, k=self.k, largest=False, dim=1)[1][
            :, -1
        ]  # (batch_size)
        kth_closest_samples = yhats[torch.arange(batch_size), kth_closest_indices, :]

        loss: torch.Tensor = self.mse_loss(kth_closest_samples, y)

        log_loss = torch.log(loss)

        return loss, log_loss

    def train(
        self, train_batch: List[torch.Tensor], epoch: int
    ) -> torch.Dict[str, float]:

        self.optimizer.zero_grad()

        loss, log_loss = self.calculate_loss(batch=train_batch)

        if self.use_log_loss_for_backprop:
            log_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()

        return {"train/loss": loss.item(), "train/log_loss": log_loss.item()}

    def eval(self, eval_batch: List[torch.Tensor]) -> Dict[str, float]:
        with torch.no_grad():
            loss, log_loss = self.calculate_loss(batch=eval_batch)
        return {"eval/loss": loss.item(), "eval/log_loss": log_loss.item()}
