import torch
import torch.nn.functional as F


def ljspeech_collate(batch):
    """
    Collate function that dynamically pads text sequences
    and mel spectrograms.
    """

    ids = []
    texts = []

    text_sequences = []
    mels = []

    for sample in batch:

        ids.append(sample["id"])
        texts.append(sample["text"])

        text_sequences.append(
            torch.tensor(
                sample["text_sequence"],
                dtype=torch.long
            )
        )

        mels.append(
            torch.tensor(
                sample["mel"],
                dtype=torch.float32
            )
        )

    # ----------------------------
    # Pad Text
    # ----------------------------

    max_text_length = max(
        seq.size(0)
        for seq in text_sequences
    )

    padded_text = []

    for seq in text_sequences:

        padding = max_text_length - seq.size(0)

        seq = F.pad(
            seq,
            (0, padding),
            value=0
        )

        padded_text.append(seq)

    padded_text = torch.stack(padded_text)

    # ----------------------------
    # Pad Mel Spectrogram
    # ----------------------------

    max_mel_length = max(
        mel.size(1)
        for mel in mels
    )

    padded_mels = []

    for mel in mels:

        padding = max_mel_length - mel.size(1)

        mel = F.pad(
            mel,
            (0, padding),
            value=0
        )

        padded_mels.append(mel)

    padded_mels = torch.stack(padded_mels)

    return {
        "id": ids,
        "text": texts,
        "text_sequence": padded_text,
        "mel": padded_mels
    }