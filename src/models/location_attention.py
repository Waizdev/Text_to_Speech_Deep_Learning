import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import Config


class LocationSensitiveAttention(nn.Module):
    """
    Tacotron2 Location-Sensitive Attention.

    Inputs
    ------
    encoder_outputs :
        [Batch, Text Length, Encoder Dim]

    attention_hidden :
        [Batch, Attention RNN Dim]

    previous_attention :
        [Batch, Text Length]

    Outputs
    -------
    context :
        [Batch, Encoder Dim]

    attention_weights :
        [Batch, Text Length]
    """

    def __init__(
        self,
        encoder_dim: int = Config.ENCODER_DIM,
        attention_rnn_dim: int = Config.ATTENTION_RNN_DIM,
        attention_dim: int = Config.ATTENTION_DIM,
    ):
        super().__init__()

        self.location_conv = nn.Conv1d(
            in_channels=1,
            out_channels=Config.ATTENTION_LOCATION_FILTERS,
            kernel_size=Config.ATTENTION_LOCATION_KERNEL_SIZE,
            padding=(Config.ATTENTION_LOCATION_KERNEL_SIZE - 1) // 2,
            bias=False,
        )

        self.location_dense = nn.Linear(
            Config.ATTENTION_LOCATION_FILTERS,
            attention_dim,
            bias=False,
        )

        self.encoder_projection = nn.Linear(
            encoder_dim,
            attention_dim,
            bias=False,
        )

        self.decoder_projection = nn.Linear(
            attention_rnn_dim,
            attention_dim,
            bias=False,
        )

        self.energy = nn.Linear(
            attention_dim,
            1,
            bias=True,
        )

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        attention_hidden: torch.Tensor,
        previous_attention: torch.Tensor,
    ):
        """
        Forward pass.
        """

        # -----------------------------
        # Location Features
        # -----------------------------

        location_features = self.location_conv(
            previous_attention.unsqueeze(1)
        )

        location_features = location_features.transpose(1, 2)

        location_features = self.location_dense(
            location_features
        )

        # -----------------------------
        # Encoder Projection
        # -----------------------------

        encoder_features = self.encoder_projection(
            encoder_outputs
        )

        # -----------------------------
        # Decoder Projection
        # -----------------------------

        decoder_features = self.decoder_projection(
            attention_hidden
        ).unsqueeze(1)

        # -----------------------------
        # Energy
        # -----------------------------

        scores = self.energy(
            torch.tanh(
                encoder_features
                + decoder_features
                + location_features
            )
        ).squeeze(-1)

        attention_weights = F.softmax(
            scores,
            dim=1,
        )

        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs,
        ).squeeze(1)

        return context, attention_weights