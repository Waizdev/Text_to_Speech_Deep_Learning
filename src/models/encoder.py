import torch
import torch.nn as nn

from configs.config import Config


class Encoder(nn.Module):
    """
    Tacotron2 Encoder.

    Architecture:
        Embedding
            ↓
        3 × Conv1D + BatchNorm + ReLU + Dropout
            ↓
        Bidirectional LSTM
            ↓
        Encoder Representations
    """

    def __init__(
        self,
        embedding_dim: int = Config.EMBEDDING_DIM,
        encoder_dim: int = Config.ENCODER_DIM,
        kernel_size: int = 5,
        num_convolutions: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim

        self.convolutions = nn.ModuleList()

        for _ in range(num_convolutions):

            self.convolutions.append(

                nn.Sequential(

                    nn.Conv1d(
                        in_channels=embedding_dim,
                        out_channels=embedding_dim,
                        kernel_size=kernel_size,
                        padding=(kernel_size - 1) // 2,
                    ),

                    nn.BatchNorm1d(
                        embedding_dim
                    ),

                    nn.ReLU(),

                    nn.Dropout(
                        dropout
                    ),
                )
            )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=encoder_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings:
                Shape -> [Batch, Text Length, Embedding Dim]

        Returns:
            Encoder outputs:
                Shape -> [Batch, Text Length, Encoder Dim]
        """

        x = embeddings.transpose(1, 2)

        for conv in self.convolutions:
            x = conv(x)

        x = x.transpose(1, 2)

        outputs, _ = self.lstm(x)

        return outputs