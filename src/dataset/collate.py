import torch
import torch.nn.functional as F


def ljspeech_collate(batch):
    """
    Collate function that dynamically pads mel spectrograms.
    """

    ids = []
    texts = []
    mels = []

    for sample in batch:
        ids.append(sample["id"])
        texts.append(sample["text"])

        mel = torch.tensor(
            sample["mel"],
            dtype=torch.float32
        )

        mels.append(mel)

    # Find longest mel in the batch
    max_length = max(mel.shape[1] for mel in mels)

    padded_mels = []

    for mel in mels:
        padding = max_length - mel.shape[1]

        padded_mel = F.pad(
            mel,
            (0, padding),
            mode="constant",
            value=0
        )

        padded_mels.append(padded_mel)

    padded_mels = torch.stack(padded_mels)

    return {
        "id": ids,
        "text": texts,
        "mel": padded_mels
    }