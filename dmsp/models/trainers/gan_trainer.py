from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import hydra

from dmsp.models.trainers.base_trainer import BaseTrainer
from dmsp.models.noise.base_noise import BaseNoise
from dmsp.utils.process_data import validate_traj_list, preprocess


class ConditionalGANTrainer(BaseTrainer):

    def __init__(
        self,
        lookback: int,
        generator: nn.Module,
        discriminator: nn.Module,
        noise_model: BaseNoise,
        discriminator_lookforward: int = 1,
        optimizer_cls: str = "torch.optim.Adam",
        optimizer_kwargs: Dict[str, Any] | None = None,
        discriminator_steps_per_generator_step: int = 1,
        generator_prediction_loss_weight: float = 0.0,
        stream_data: bool = False,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.lookback = lookback
        self.discriminator_lookforward = discriminator_lookforward
        self.discriminator_steps_per_generator_step = (
            discriminator_steps_per_generator_step
        )
        self.generator_prediction_loss_weight = generator_prediction_loss_weight
        self.stream_data = stream_data

        self.device = torch.device(device=device)
        self.dtype = dtype

        self.generator = generator
        self.discriminator = discriminator
        self.noise_model = noise_model

        self.generator_optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            {
                "_target_": optimizer_cls,
                **({} if optimizer_kwargs is None else optimizer_kwargs),
            },
            params=self.generator.parameters(),
            _convert_="partial",
        )
        self.discriminator_optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            {
                "_target_": optimizer_cls,
                **({} if optimizer_kwargs is None else optimizer_kwargs),
            },
            params=self.discriminator.parameters(),
            _convert_="partial",
        )

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

        self.train_batch_count = 0

    def validate_traj_lst(
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
            lookforward=self.discriminator_lookforward,
        )

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

    def calculate_loss(
        self, batch: torch.Tensor | List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        X, y = batch  # (batch_size, lookback * d), (batch_size, lookforward * d)

        batch_size = y.shape[0]
        d = y.shape[1] // self.discriminator_lookforward

        noise = self.noise_model.sample(
            n_samples=batch_size * self.discriminator_lookforward, device=self.device
        ).reshape(
            (self.discriminator_lookforward, batch_size, -1)
        )  # (lookforward, batch_size, noise_dim)

        Z = X.detach().clone()  # (batch_size, lookback * d)

        generated_samples = []

        for _ in range(self.discriminator_lookforward):
            forward_sample = self.generator((noise, Z))  # (batch_size, d)
            Z[:, :-d] = Z[:, d:]
            Z[:, -d:] = forward_sample
            generated_samples.append(forward_sample)

        generated_samples = torch.cat(
            generated_samples, dim=1
        )  # (batch_size, lookforward * d)
        real_samples = y  # (batch_size, lookforward * d)

        generated_discriminator_output = self.discriminator(
            (generated_samples, X)
        )  # (batch_size, 1)
        real_discriminator_output = self.discriminator(
            (real_samples, X)
        )  # (batch_size, 1)

        discriminator_generated_loss = self.bce_loss(
            generated_discriminator_output,
            torch.zeros_like(generated_discriminator_output),
        )
        discriminator_real_loss = self.bce_loss(
            real_discriminator_output, torch.ones_like(real_discriminator_output)
        )
        discriminator_loss = discriminator_generated_loss + discriminator_real_loss

        generator_loss = self.bce_loss(
            generated_discriminator_output,
            torch.ones_like(generated_discriminator_output),
        )

        if self.generator_prediction_loss_weight != 0:
            prediction_noise = self.noise_model.sample(n_samples=batch_size)
            generator_prediction = self.generator((prediction_noise, X))
            generator_loss += self.generator_prediction_loss_weight * self.mse_loss(
                generator_prediction, y[:, :d]
            )

        return generator_loss, discriminator_loss

    def train(
        self, train_batch: torch.Tensor | List[torch.Tensor], epoch: int
    ) -> Dict[str, Any]:
        generator_loss, discriminator_loss = self.calculate_loss(batch=train_batch)

        self.train_batch_count += 1

        if (
            self.train_batch_count % (self.discriminator_steps_per_generator_step + 1)
            == 0
        ):
            self.generator_optimizer.zero_grad()
            generator_loss.backward()
            self.generator_optimizer.step()
        else:
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

        return {
            "train/generator_loss": generator_loss.item(),
            "train/discriminator_loss": discriminator_loss.item(),
        }

    def eval(self, eval_batch: torch.Tensor | List[torch.Tensor]) -> Dict[str, float]:
        with torch.no_grad():
            generator_loss, discriminator_loss = self.calculate_loss(batch=eval_batch)

        return {
            "eval/generator_loss": generator_loss.item(),
            "eval/discriminator_loss": discriminator_loss.item(),
        }
