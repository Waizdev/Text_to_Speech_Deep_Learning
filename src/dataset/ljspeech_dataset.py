import os

import pandas as pd
from torch.utils.data import Dataset

from Preprocessing.audio_loader import load_audio
from Preprocessing.mel_spectrogram import generate_mel_spectrogram


class LJSpeechDataset(Dataset):
    """
    Custom PyTorch Dataset for the LJSpeech dataset.
    """

    def __init__(self, dataset_path):
        """
        Args:
            dataset_path (str): Path to the LJSpeech dataset folder.
        """
        self.dataset_path = dataset_path

        metadata_path = os.path.join(
            dataset_path,
            "metadata.csv"
        )

        self.metadata = pd.read_csv(
            metadata_path,
            sep="|",
            header=None,
            names=[
                "id",
                "text",
                "normalized_text"
            ]
        )

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.metadata)

    def __getitem__(self, index):
        """
        Returns one training sample.
        """

        row = self.metadata.iloc[index]

        file_id = row["id"]
        text = row["normalized_text"]

        audio_path = os.path.join(
            self.dataset_path,
            "wavs",
            f"{file_id}.wav"
        )

        audio, sample_rate = load_audio(audio_path)

        mel_spectrogram = generate_mel_spectrogram(
            audio,
            sample_rate
        )

        sample = {
            "id": file_id,
            "text": text,
            "mel": mel_spectrogram
        }

        return sample