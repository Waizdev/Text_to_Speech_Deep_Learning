import torch.nn as nn
import torch.nn.functional as F


class Prenet(nn.Module):
    """
    Tacotron2 Prenet.

    Applies two fully connected layers with
    ReLU activation and dropout.
    """

    def __init__(
        self,
        input_dim=80,
        hidden_dim=256,
        dropout=0.5
    ):
        super().__init__()

        self.layer1 = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.layer2 = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.dropout = dropout

    def forward(self, x):

        x = F.relu(
            self.layer1(x)
        )

        x = F.dropout(
            x,
            p=self.dropout,
            training=True
        )

        x = F.relu(
            self.layer2(x)
        )

        x = F.dropout(
            x,
            p=self.dropout,
            training=True
        )

        return x