import torch
import torch.nn as nn

from configs.config import Config


class DecoderRNN(nn.Module):
    """
    Tacotron2 Decoder LSTM.
    """

    def __init__(
        self,
        attention_rnn_dim: int = Config.ATTENTION_RNN_DIM,
        encoder_dim: int = Config.ENCODER_DIM,
        decoder_dim: int = Config.DECODER_DIM,
    ):
        super().__init__()

        self.decoder_rnn = nn.LSTMCell(
            input_size=attention_rnn_dim + encoder_dim,
            hidden_size=decoder_dim,
        )

    def initialize_states(
        self,
        batch_size: int,
        device: torch.device,
    ):

        hidden = torch.zeros(
            batch_size,
            Config.DECODER_DIM,
            device=device,
        )

        cell = torch.zeros(
            batch_size,
            Config.DECODER_DIM,
            device=device,
        )

        return hidden, cell

    def forward(
        self,
        attention_hidden: torch.Tensor,
        context_vector: torch.Tensor,
        hidden_state: torch.Tensor,
        cell_state: torch.Tensor,
    ):

        decoder_input = torch.cat(
            [attention_hidden, context_vector],
            dim=-1,
        )

        hidden_state, cell_state = self.decoder_rnn(
            decoder_input,
            (hidden_state, cell_state),
        )

        return hidden_state, cell_state