import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Additive (Bahdanau) Attention
    """

    def __init__(
        self,
        encoder_dim=256,
        decoder_dim=256,
        attention_dim=128
    ):
        super().__init__()

        self.encoder_projection = nn.Linear(
            encoder_dim,
            attention_dim
        )

        self.decoder_projection = nn.Linear(
            decoder_dim,
            attention_dim
        )

        self.energy = nn.Linear(
            attention_dim,
            1
        )

    def forward(
        self,
        encoder_outputs,
        decoder_hidden
    ):
        """
        encoder_outputs:
            [Batch, Seq Length, Encoder Dim]

        decoder_hidden:
            [Batch, Decoder Dim]
        """

        encoder_projection = self.encoder_projection(
            encoder_outputs
        )

        decoder_projection = self.decoder_projection(
            decoder_hidden
        ).unsqueeze(1)

        scores = self.energy(
            torch.tanh(
                encoder_projection +
                decoder_projection
            )
        ).squeeze(-1)

        attention_weights = F.softmax(
            scores,
            dim=1
        )

        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)

        return context, attention_weights