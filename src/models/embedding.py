import torch
import torch.nn as nn

from configs.config import Config


class CharacterEmbedding(nn.Module):
    """
    Converts a sequence of character IDs into dense embedding vectors.

    Input Shape:
        [batch_size, sequence_length]

    Output Shape:
        [batch_size, sequence_length, embedding_dim]
    """

    def __init__(
        self,
        vocab_size: int = Config.VOCAB_SIZE,
        embedding_dim: int = Config.EMBEDDING_DIM,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

    def forward(
        self,
        text: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text: Tensor of character IDs.

        Returns:
            Embedded character vectors.
        """
        return self.embedding(text)