import torch
import torch.nn as nn

from configs.config import Config


class Prenet(nn.Module):
    """
    Tacotron2 Prenet.

    The Prenet transforms the previous mel-spectrogram frame into
    a compact representation before it is passed to the decoder.

    Architecture:
        Linear
            ↓
        ReLU
            ↓
        Dropout
            ↓
        Linear
            ↓
        ReLU
            ↓
        Dropout
    """

    def __init__(
        self,
        input_dim: int = Config.N_MELS,
        hidden_dim: int = Config.PRENET_DIM,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),
        )

    def forward(
        self,
        mel_frame: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mel_frame:
                Shape -> [Batch, N_Mels]

        Returns:
            Shape -> [Batch, Prenet Dim]
        """

        return self.layers(mel_frame)