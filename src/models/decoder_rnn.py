import torch
import torch.nn as nn


class DecoderRNN(nn.Module):
    """
    Decoder LSTM used after the attention mechanism.
    """

    def __init__(
        self,
        attention_dim=256,
        encoder_dim=256,
        decoder_dim=256
    ):
        super().__init__()

        self.hidden_size = decoder_dim

        self.lstm = nn.LSTMCell(
            input_size=attention_dim + encoder_dim,
            hidden_size=decoder_dim
        )

    def initialize_states(
        self,
        batch_size,
        device
    ):

        hidden = torch.zeros(
            batch_size,
            self.hidden_size,
            device=device
        )

        cell = torch.zeros(
            batch_size,
            self.hidden_size,
            device=device
        )

        return hidden, cell

    def forward(
        self,
        attention_hidden,
        context,
        hidden,
        cell
    ):

        decoder_input = torch.cat(
            [
                attention_hidden,
                context
            ],
            dim=1
        )

        hidden, cell = self.lstm(
            decoder_input,
            (hidden, cell)
        )

        return hidden, cell