import torch.nn as nn


class StopTokenPredictor(nn.Module):
    """
    Predicts whether decoding should stop.
    """

    def __init__(
        self,
        decoder_dim=256
    ):
        super().__init__()

        self.linear = nn.Linear(
            decoder_dim,
            1
        )

    def forward(self, x):
        return self.linear(x)