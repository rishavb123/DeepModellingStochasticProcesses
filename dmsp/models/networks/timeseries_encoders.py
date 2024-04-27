from typing import Any, Callable

import torch
import torch.nn as nn


class TimeseriesEncoder(nn.Module):

    def __init__(
        self,
        data_dim: int,
        encoder: nn.Module,
        post_process_f: Callable[[Any], torch.Tensor] | None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.data_dim = data_dim
        self.encoder = encoder
        self.post_process_f = (
            (lambda x: x) if post_process_f is None else post_process_f
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.post_process_f(
            self.encoder(x.reshape((x.shape[0], -1, self.data_dim)))
        )
