import librosa


def load_audio(audio_path: str):
    """
    Load a WAV file without changing its sampling rate.

    Args:
        audio_path: Path to the audio file.

    Returns:
        tuple:
            audio (numpy.ndarray)
            sample_rate (int)
    """
    audio, sample_rate = librosa.load(audio_path, sr=None)

    return audio, sample_rate