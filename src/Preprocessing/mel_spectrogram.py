import librosa
import numpy as np


def generate_mel_spectrogram(
    audio,
    sample_rate,
    n_fft=1024,
    hop_length=256,
    n_mels=80
):
    """
    Convert waveform into a Mel spectrogram.
    """

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db