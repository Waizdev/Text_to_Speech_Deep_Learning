import torch
import torch.nn as nn

from configs.config import Config


class StopTokenPredictor(nn.Module):
    """
    Predicts whether the decoder should stop generating.
    """

    def __init__(
        self,
        decoder_dim: int = Config.DECODER_DIM,
        encoder_dim: int = Config.ENCODER_DIM,
    ):
        super().__init__()

        self.linear = nn.Linear(
            decoder_dim + encoder_dim,
            1
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

        stop_token = self.linear(x)

        return stop_token