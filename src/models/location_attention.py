import torch
import torch.nn as nn
import torch.nn.functional as F


class LocationSensitiveAttention(nn.Module):
    """
    Location Sensitive Attention used in Tacotron2.
    """

    def __init__(
        self,
        encoder_dim=256,
        attention_rnn_dim=256,
        attention_dim=128,
        location_filters=32,
        location_kernel_size=31
    ):
        super().__init__()

        # Encoder projection
        self.encoder_projection = nn.Linear(
            encoder_dim,
            attention_dim,
            bias=False
        )

        # Attention RNN projection
        self.decoder_projection = nn.Linear(
            attention_rnn_dim,
            attention_dim,
            bias=False
        )

        # Location convolution
        self.location_conv = nn.Conv1d(
            in_channels=1,
            out_channels=location_filters,
            kernel_size=location_kernel_size,
            padding=(location_kernel_size - 1) // 2,
            bias=False
        )

        # Location projection
        self.location_projection = nn.Linear(
            location_filters,
            attention_dim,
            bias=False
        )

        # Energy layer
        self.energy = nn.Linear(
            attention_dim,
            1,
            bias=True
        )

    def forward(
        self,
        encoder_outputs,
        attention_hidden,
        previous_attention
    ):
        """
        encoder_outputs:
            [B, T, Encoder Dim]

        attention_hidden:
            [B, Attention RNN Dim]

        previous_attention:
            [B, T]
        """

        encoder_features = self.encoder_projection(
            encoder_outputs
        )

        decoder_features = self.decoder_projection(
            attention_hidden
        ).unsqueeze(1)

        location_features = self.location_conv(
            previous_attention.unsqueeze(1)
        )

        location_features = location_features.transpose(1, 2)

        location_features = self.location_projection(
            location_features
        )

        energies = self.energy(
            torch.tanh(
                encoder_features +
                decoder_features +
                location_features
            )
        ).squeeze(-1)

        attention_weights = F.softmax(
            energies,
            dim=1
        )

        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)

        return context, attention_weights