import torch
import torch.nn as nn
from configs.config import Config


class Encoder(nn.Module):
    """
    Tacotron2 Encoder.

    Architecture:
    Embedding Output
            │
            ▼
    3 × Conv1D + BatchNorm + ReLU + Dropout
            │
            ▼
        Bidirectional LSTM
            │
            ▼
      Encoder Representations
    """

    def __init__(
        self,
        embedding_dim: int = Config.EMBEDDING_DIM,
        encoder_dim: int = Config.ENCODER_DIM,
        kernel_size=5,
        num_convolutions=3,
        dropout=0.5
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim
        self.dropout = dropout

        # --------------------------------------------------
        # Convolution Layers
        # --------------------------------------------------

        self.convolutions = nn.ModuleList()

        for _ in range(num_convolutions):

            conv = nn.Sequential(

                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=embedding_dim,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2
                ),

                nn.BatchNorm1d(
                    embedding_dim
                ),

                nn.ReLU(),

                nn.Dropout(dropout)
            )

            self.convolutions.append(conv)

        # --------------------------------------------------
        # Bidirectional LSTM
        # --------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=encoder_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        """
        Forward Pass

        Input:
            x -> [Batch, Text Length, Embedding Dim]

        Output:
            output -> [Batch, Text Length, Encoder Dim]
        """

        # ---------------------------------------------
        # Conv1D expects:
        #
        # [Batch, Channels, Sequence]
        #
        # Current:
        #
        # [Batch, Sequence, Channels]
        # ---------------------------------------------

        x = x.transpose(1, 2)

        # ---------------------------------------------
        # Pass through convolution blocks
        # ---------------------------------------------

        for conv in self.convolutions:
            x = conv(x)

        # ---------------------------------------------
        # LSTM expects:
        #
        # [Batch, Sequence, Features]
        # ---------------------------------------------

        x = x.transpose(1, 2)

        output, _ = self.lstm(x)

        return output