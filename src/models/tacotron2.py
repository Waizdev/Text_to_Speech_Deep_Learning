import torch
import torch.nn as nn

from models.embedding import CharacterEmbedding
from models.encoder import Encoder
from models.prenet import Prenet
from models.attention_rnn import AttentionRNN
from models.location_attention import LocationSensitiveAttention
from models.decoder_rnn import DecoderRNN
from models.mel_projection import MelProjection
from models.stop_token import StopTokenPredictor


class Tacotron2(nn.Module):
    """
    Simplified Tacotron2 Architecture
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        encoder_dim=256,
        attention_dim=128,
        decoder_dim=256,
        n_mels=80
    ):
        super().__init__()

        ##################################
        # Encoder
        ##################################

        self.embedding = CharacterEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim
        )

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            hidden_size=encoder_dim // 2
        )

        ##################################
        # Decoder Components
        ##################################

        self.prenet = Prenet(
            input_dim=n_mels,
            hidden_dim=decoder_dim
        )

        self.attention_rnn = AttentionRNN(
            prenet_dim=decoder_dim,
            encoder_dim=encoder_dim,
            attention_rnn_dim=decoder_dim
        )

        self.attention = LocationSensitiveAttention(
            encoder_dim=encoder_dim,
            attention_rnn_dim=decoder_dim,
            attention_dim=attention_dim
        )

        self.decoder_rnn = DecoderRNN(
            attention_dim=decoder_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim
        )

        self.mel_projection = MelProjection(
            decoder_dim=decoder_dim,
            n_mels=n_mels
        )

        self.stop_predictor = StopTokenPredictor(
            decoder_dim=decoder_dim
        )

        self.n_mels = n_mels
        self.encoder_dim = encoder_dim

    def forward(
        self,
        text,
        max_decoder_steps=100
    ):

        ##################################
        # Encoder
        ##################################

        embedded = self.embedding(text)

        encoder_outputs = self.encoder(
            embedded
        )

        batch_size = text.size(0)
        device = text.device

        ##################################
        # Initial States
        ##################################

        previous_mel = torch.zeros(
            batch_size,
            self.n_mels,
            device=device
        )

        attention_hidden, attention_cell = \
            self.attention_rnn.initialize_states(
                batch_size,
                device
            )

        decoder_hidden, decoder_cell = \
            self.decoder_rnn.initialize_states(
                batch_size,
                device
            )

        context = torch.zeros(
            batch_size,
            self.encoder_dim,
            device=device
        )

        attention_weights = torch.zeros(
            batch_size,
            encoder_outputs.size(1),
            device=device
        )

        ##################################
        # Decoder Loop
        ##################################

        mel_outputs = []
        stop_outputs = []
        alignments = []

        for _ in range(max_decoder_steps):

            prenet_output = self.prenet(
                previous_mel
            )

            attention_hidden, attention_cell = \
                self.attention_rnn(
                    prenet_output,
                    context,
                    attention_hidden,
                    attention_cell
                )

            context, attention_weights = \
                self.attention(
                    encoder_outputs,
                    attention_hidden,
                    attention_weights
                )

            decoder_hidden, decoder_cell = \
                self.decoder_rnn(
                    attention_hidden,
                    context,
                    decoder_hidden,
                    decoder_cell
                )

            mel_frame = self.mel_projection(
                decoder_hidden
            )

            stop = self.stop_predictor(
                decoder_hidden
            )

            mel_outputs.append(
                mel_frame.unsqueeze(1)
            )

            stop_outputs.append(
                stop.unsqueeze(1)
            )

            alignments.append(
                attention_weights.unsqueeze(1)
            )

            previous_mel = mel_frame

        mel_outputs = torch.cat(
            mel_outputs,
            dim=1
        )

        stop_outputs = torch.cat(
            stop_outputs,
            dim=1
        )

        alignments = torch.cat(
            alignments,
            dim=1
        )

        return (
            mel_outputs,
            stop_outputs,
            alignments
        )