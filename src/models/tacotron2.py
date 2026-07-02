import torch
import torch.nn as nn

from configs.config import Config

from models.embedding import CharacterEmbedding
from models.encoder import Encoder
from models.prenet import Prenet
from models.attention_rnn import AttentionRNN
from models.location_attention import LocationSensitiveAttention
from models.decoder_rnn import DecoderRNN
from models.mel_projection import MelProjection
from models.stop_token import StopTokenPredictor
from models.postnet import PostNet


class Tacotron2(nn.Module):
    """
    Simplified Tacotron2
    """

    def __init__(self):
        super().__init__()

        ############################################
        # Text Encoder
        ############################################

        self.embedding = CharacterEmbedding()
        self.encoder = Encoder()

        ############################################
        # Decoder
        ############################################

        self.prenet = Prenet()
        self.attention_rnn = AttentionRNN()
        self.attention = LocationSensitiveAttention()
        self.decoder_rnn = DecoderRNN()

        ############################################
        # Output Heads
        ############################################

        self.mel_projection = MelProjection()
        self.stop_token = StopTokenPredictor()
        self.postnet = PostNet()

    def initialize_decoder_states(
        self,
        encoder_outputs,
    ):
        """
        Initialize all decoder states.
        """

        batch_size = encoder_outputs.size(0)
        text_length = encoder_outputs.size(1)
        device = encoder_outputs.device

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
            Config.ENCODER_DIM,
            device=device
        )

        attention_weights = torch.zeros(
            batch_size,
            text_length,
            device=device
        )

        previous_mel = torch.zeros(
            batch_size,
            Config.N_MELS,
            device=device
        )

        return (
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            context,
            attention_weights,
            previous_mel
        )

    def forward(
        self,
        text,
        target_mels=None,
    ):
        """
        Forward pass.
        """

        ############################################
        # Encoder
        ############################################

        embeddings = self.embedding(text)

        encoder_outputs = self.encoder(
            embeddings
        )

        (
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            context,
            attention_weights,
            previous_mel
        ) = self.initialize_decoder_states(
            encoder_outputs
        )

        mel_outputs = []
        stop_outputs = []

        ############################################
        # Decoder Loop
        ############################################

        if target_mels is not None:
            max_steps = target_mels.size(2)
        else:
            max_steps = Config.MAX_DECODER_STEPS

        for step in range(max_steps):

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
                decoder_hidden,
                context
            )

            stop_prediction = self.stop_token(
                decoder_hidden,
                context
            )

            mel_outputs.append(
                mel_frame.unsqueeze(2)
            )

            stop_outputs.append(
                stop_prediction
            )

            ############################################
            # Teacher Forcing
            ############################################

            if target_mels is not None:

                previous_mel = target_mels[:, :, step]

            else:

                previous_mel = mel_frame

                if torch.sigmoid(stop_prediction).item() > Config.STOP_THRESHOLD:
                    break

        ############################################
        # Stack Decoder Outputs
        ############################################

        mel_outputs = torch.cat(
            mel_outputs,
            dim=2
        )

        stop_outputs = torch.cat(
            stop_outputs,
            dim=1
        )

        ############################################
        # PostNet Refinement
        ############################################

        mel_outputs_postnet = self.postnet(
            mel_outputs
        )

        mel_outputs_postnet = (
            mel_outputs
            + mel_outputs_postnet
        )

        ############################################
        # Return Outputs
        ############################################

        return (
            mel_outputs,
            mel_outputs_postnet,
            stop_outputs,
            attention_weights
        )