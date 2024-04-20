from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import hydra
import logging

from dmsp.models.trainers.base_trainer import BaseTrainer


logger = logging.getLogger(__name__)


class StochasticityLossTrainer(BaseTrainer):

    def __init__(
        self,
        lookback: int,
        prediction_model: nn.Module,
        optimizer_cls: str = "torch.optim.Adam",
        optimizer_kwargs: Dict[str, Any] | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.lookback = lookback

        self.device = torch.device(device)
        self.dtype = dtype
        self.prediction_model = prediction_model.to(device=self.device)
        self.optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            {
                "_target_": optimizer_cls,
                **({} if optimizer_kwargs is None else optimizer_kwargs),
            },
            params=self.prediction_model.parameters(),
            _convert_="partial",
        )

        self.mse_loss = torch.nn.MSELoss()

    def preprocess(self, trajectory_list: List[np.ndarray]) -> torch.utils.data.Dataset:
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

    def sample(
        self,
        trajectory_list: List[np.ndarray],
        n_samples: int = 1,
        traj_length: int = 1,
    ) -> List[np.ndarray]:
        pass

    def load_model(self, path: str) -> None:
        self.prediction_model.load_state_dict(torch.load(path))

    def save_model(self, path: str) -> None:
        torch.save(self.prediction_model.state_dict(), path)

    def train(self, train_batch: List[torch.Tensor]) -> torch.Dict[str, float]:
        X, y = train_batch

        self.optimizer.zero_grad()

        yhat: torch.Tensor = self.prediction_model(X)
        loss: torch.Tensor = self.mse_loss(yhat, y)

        loss.backward()
        self.optimizer.step()

        return {"train/loss": loss.item()}

    def eval(
        self, eval_batch: List[torch.Tensor], visualize: bool = False
    ) -> Dict[str, float]:
        if visualize:
            logger.info("No visualization for this trainer yet!")

        X, y = eval_batch

        with torch.no_grad():
            yhat: torch.Tensor = self.prediction_model(X)
            loss: torch.Tensor = self.mse_loss(yhat, y)

        return {"eval/loss": loss.item()}
