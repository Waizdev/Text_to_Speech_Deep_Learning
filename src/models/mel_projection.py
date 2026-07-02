import torch
import torch.nn as nn

from configs.config import Config


class MelProjection(nn.Module):
    """
    Projects decoder output to mel spectrogram.
    """

    def __init__(
        self,
        decoder_dim: int = Config.DECODER_DIM,
        encoder_dim: int = Config.ENCODER_DIM,
        n_mels: int = Config.N_MELS,
    ):
        super().__init__()

        self.linear = nn.Linear(
            decoder_dim + encoder_dim,
            n_mels,
        )

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        context_vector: torch.Tensor,
    ):

        x = torch.cat(
            [decoder_hidden, context_vector],
            dim=-1,
        )

        mel = self.linear(x)

        return mel