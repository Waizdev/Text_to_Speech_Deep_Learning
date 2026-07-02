import torch.nn as nn


class CharacterEmbedding(nn.Module):
    """
    Character embedding layer.

    Converts character IDs into dense vectors.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

    def forward(self, x):

        return self.embedding(x)