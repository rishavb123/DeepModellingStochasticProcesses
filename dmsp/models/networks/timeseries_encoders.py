import torch
import torch.nn as nn


class LSTMEncoder(nn.LSTM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _hidden = super().forward(x)
        return output
