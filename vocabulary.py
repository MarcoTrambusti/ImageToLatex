from collections import Counter


class Vocabulary:
    """
    Vocabulary class to map tokens to numerical indices and vice versa.

    Attributes:
        itos (dict): Mapping from indices to tokens (int → str).
        stoi (dict): Mapping from tokens to indices (str → int).
        threshold (int): Minimum frequency required for a token to be included.
    """

    def __init__(self):
        """
        Initializes the vocabulary with special tokens:
            <pad>: padding token
            <sos>: start-of-sequence token
            <eos>: end-of-sequence token
            <unk>: unknown token for rare or unseen words
        """
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.threshold = 1  # Minimum frequency to include a word in the vocabulary

    def build_vocabulary(self, sentence_list):
        """
        Builds vocabulary from a list of tokenized sentences.

        Args:
            sentence_list (List[List[str]]): List of tokenized sentences.

        Steps:
            1. Count frequencies of all tokens in the dataset.
            2. Add tokens to the vocabulary if their frequency exceeds the threshold.
        """
        frequencies = Counter()
        idx = 4  # Start indexing new words after the special tokens

        # Count word frequencies, skipping special tokens
        for sentence in sentence_list:
            for word in sentence:
                if word not in ["<pad>", "<sos>", "<eos>", "<unk>"]:
                    frequencies[word] += 1

        # Add words meeting the frequency threshold to the vocabulary
        for word, freq in frequencies.items():
            if freq >= self.threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        """
        Converts a list of tokens into a list of numerical indices.

        Args:
            text (List[str]): List of tokens.

        Returns:
            List[int]: Corresponding indices for each token.
            Unseen tokens are mapped to the <unk> index.
        """
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in text]
