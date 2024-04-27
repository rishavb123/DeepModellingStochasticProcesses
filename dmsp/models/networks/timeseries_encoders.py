"""Class for timeseries encoders."""

from typing import Any, Callable

import torch
import torch.nn as nn


class TimeseriesEncoder(nn.Module):
    """Generic time series encoder class."""

    def __init__(
        self,
        data_dim: int,
        encoder: nn.Module,
        post_process_f: Callable[[Any], torch.Tensor] | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Constructor for Timeseries Encoder.

        Args:
            data_dim (int): The data dimension.
            encoder (nn.Module): The encoder that takes a tensor of shape (batch_size, sequence_length, data_dim).
            post_process_f (Callable[[Any], torch.Tensor] | None, optional): An optional post processing function. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.data_dim = data_dim
        self.encoder = encoder
        self.post_process_f = (
            (lambda x: x) if post_process_f is None else post_process_f
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function for the timeseries encoder.

        Args:
            x (torch.Tensor): The flattened time series data of size (batch_size, sequence_length * data_dim).

        Returns:
            torch.Tensor: The encoded vector.
        """
        return self.post_process_f(
            self.encoder(x.reshape((x.shape[0], -1, self.data_dim)))
        )
