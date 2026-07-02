import torch
import torch.nn as nn


class AttentionRNN(nn.Module):
    """
    Attention LSTM used in Tacotron2.

    Inputs:
        Prenet Output
        +
        Previous Attention Context

    Output:
        Attention Hidden State
    """

    def __init__(
        self,
        prenet_dim=256,
        encoder_dim=256,
        attention_rnn_dim=256
    ):
        super().__init__()

        self.hidden_size = attention_rnn_dim

        self.lstm = nn.LSTMCell(
            input_size=prenet_dim + encoder_dim,
            hidden_size=attention_rnn_dim
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
        prenet_output,
        attention_context,
        hidden,
        cell
    ):

        lstm_input = torch.cat(
            [
                prenet_output,
                attention_context
            ],
            dim=1
        )

        hidden, cell = self.lstm(
            lstm_input,
            (hidden, cell)
        )

        return hidden, cell