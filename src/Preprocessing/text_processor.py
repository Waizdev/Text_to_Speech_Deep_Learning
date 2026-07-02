import string


class TextProcessor:
    """
    Handles text normalization and tokenization.
    """

    def __init__(self):

        self.characters = (
            string.ascii_lowercase +
            "!'(),-.:;? "
        )

        self.char_to_id = {
            char: idx + 1
            for idx, char in enumerate(self.characters)
        }

        self.id_to_char = {
            idx: char
            for char, idx in self.char_to_id.items()
        }

    def normalize(self, text):
        """
        Convert text to lowercase.
        """

        return text.lower()

    def text_to_sequence(self, text):
        """
        Convert text into integer IDs.
        """

        text = self.normalize(text)

        sequence = [
            self.char_to_id[char]
            for char in text
            if char in self.char_to_id
        ]

        return sequence

    def sequence_to_text(self, sequence):
        """
        Convert integer IDs back to text.
        """

        return "".join(
            self.id_to_char[idx]
            for idx in sequence
        )