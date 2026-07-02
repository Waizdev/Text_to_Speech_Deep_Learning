import torch.nn as nn


class MelProjection(nn.Module):
    """
    Projects decoder hidden state to a mel spectrogram frame.
    """

    def __init__(
        self,
        decoder_dim=256,
        n_mels=80
    ):
        super().__init__()

        self.linear = nn.Linear(
            decoder_dim,
            n_mels
        )

    def forward(self, x):
        return self.linear(x)