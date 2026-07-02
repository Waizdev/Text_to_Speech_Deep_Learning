import torch
import torch.nn as nn

from configs.config import Config


class AttentionRNN(nn.Module):
    """
    Tacotron2 Attention RNN.

    Inputs:
        - Prenet output
        - Previous attention context

    Output:
        - Attention hidden state
    """

    def __init__(
        self,
        prenet_dim: int = Config.PRENET_DIM,
        encoder_dim: int = Config.ENCODER_DIM,
        attention_rnn_dim: int = Config.ATTENTION_RNN_DIM,
    ) -> None:
        super().__init__()

        self.attention_rnn = nn.LSTMCell(
            input_size=prenet_dim + encoder_dim,
            hidden_size=attention_rnn_dim,
        )

    def initialize_states(
        self,
        batch_size: int,
        device: torch.device,
    ):
        """
        Initialize hidden and cell states.
        """

        hidden = torch.zeros(
            batch_size,
            Config.ATTENTION_RNN_DIM,
            device=device,
        )

        cell = torch.zeros(
            batch_size,
            Config.ATTENTION_RNN_DIM,
            device=device,
        )

        return hidden, cell

    def forward(
        self,
        prenet_output: torch.Tensor,
        context_vector: torch.Tensor,
        hidden_state: torch.Tensor,
        cell_state: torch.Tensor,
    ):
        """
        Args:
            prenet_output:
                [Batch, PRENET_DIM]

            context_vector:
                [Batch, ENCODER_DIM]

            hidden_state:
                [Batch, ATTENTION_RNN_DIM]

            cell_state:
                [Batch, ATTENTION_RNN_DIM]
        """

        rnn_input = torch.cat(
            [prenet_output, context_vector],
            dim=-1,
        )

        hidden_state, cell_state = self.attention_rnn(
            rnn_input,
            (hidden_state, cell_state),
        )

        return hidden_state, cell_state