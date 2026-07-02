import torch
import torch.nn as nn

from models.attention import Attention


class Decoder(nn.Module):
    """
    Simplified Tacotron2 Decoder.

    Input:
        encoder_outputs -> [Batch, Seq Length, Encoder Dim]

    Output:
        mel_output      -> [Batch, N_Mels]
        stop_token      -> [Batch, 1]
        attention       -> [Batch, Seq Length]
    """

    def __init__(
        self,
        encoder_dim=256,
        decoder_dim=256,
        attention_dim=128,
        n_mels=80
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.n_mels = n_mels

        # ---------------------------------------
        # Attention
        # ---------------------------------------

        self.attention = Attention(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim
        )

        # ---------------------------------------
        # Decoder LSTM
        # ---------------------------------------

        self.decoder_lstm = nn.LSTMCell(
            input_size=encoder_dim + n_mels,
            hidden_size=decoder_dim
        )

        # ---------------------------------------
        # Mel Projection
        # ---------------------------------------

        self.mel_projection = nn.Linear(
            decoder_dim,
            n_mels
        )

        # ---------------------------------------
        # Stop Token Prediction
        # ---------------------------------------

        self.stop_projection = nn.Linear(
            decoder_dim,
            1
        )

    def initialize_states(self, batch_size, device):
        """
        Initialize hidden and cell states.
        """

        hidden = torch.zeros(
            batch_size,
            self.decoder_dim,
            device=device
        )

        cell = torch.zeros(
            batch_size,
            self.decoder_dim,
            device=device
        )

        return hidden, cell

    def forward(
        self,
        encoder_outputs,
        previous_mel=None
    ):
        """
        Parameters
        ----------
        encoder_outputs:
            [Batch, Seq Length, Encoder Dim]

        previous_mel:
            [Batch, N_Mels]

        Returns
        -------
        mel_output
        stop_token
        attention_weights
        """

        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device

        if previous_mel is None:

            previous_mel = torch.zeros(
                batch_size,
                self.n_mels,
                device=device
            )

        hidden, cell = self.initialize_states(
            batch_size,
            device
        )

        # ---------------------------------------
        # Attention
        # ---------------------------------------

        context, attention_weights = self.attention(
            encoder_outputs,
            hidden
        )

        # ---------------------------------------
        # Decoder Input
        # ---------------------------------------

        decoder_input = torch.cat(
            [
                previous_mel,
                context
            ],
            dim=1
        )

        hidden, cell = self.decoder_lstm(
            decoder_input,
            (hidden, cell)
        )

        # ---------------------------------------
        # Predict Mel
        # ---------------------------------------

        mel_output = self.mel_projection(
            hidden
        )

        # ---------------------------------------
        # Predict Stop Token
        # ---------------------------------------

        stop_token = self.stop_projection(
            hidden
        )

        return (
            mel_output,
            stop_token,
            attention_weights
        )