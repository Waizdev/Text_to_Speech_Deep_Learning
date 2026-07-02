import torch
import torch.nn as nn

from configs.config import Config


class PostNet(nn.Module):
    """
    Tacotron2 PostNet.

    Refines the predicted mel spectrogram.
    """

    def __init__(
        self,
        n_mels: int = Config.N_MELS,
        postnet_dim: int = Config.POSTNET_DIM,
        kernel_size: int = Config.POSTNET_KERNEL_SIZE,
        num_layers: int = Config.POSTNET_CONV_LAYERS,
        dropout: float = Config.POSTNET_DROPOUT,
    ):
        super().__init__()

        layers = []

        # First layer
        layers.append(
            nn.Sequential(
                nn.Conv1d(
                    n_mels,
                    postnet_dim,
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
                nn.BatchNorm1d(postnet_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
            )
        )

        # Middle layers
        for _ in range(num_layers - 2):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        postnet_dim,
                        postnet_dim,
                        kernel_size,
                        padding=(kernel_size - 1) // 2,
                    ),
                    nn.BatchNorm1d(postnet_dim),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                )
            )

        # Final layer
        layers.append(
            nn.Sequential(
                nn.Conv1d(
                    postnet_dim,
                    n_mels,
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
                nn.BatchNorm1d(n_mels),
                nn.Dropout(dropout),
            )
        )

        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        mel: torch.Tensor,
    ):
        """
        Input:
            [Batch, N_Mels, Time]

        Output:
            [Batch, N_Mels, Time]
        """

        x = mel

        for layer in self.layers:
            x = layer(x)

        return x