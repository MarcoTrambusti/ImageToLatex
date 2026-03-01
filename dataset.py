from torch.utils.data import Dataset
from PIL import Image
import io
import torch

class Im2LatexDataset(Dataset):
    """
    Custom PyTorch Dataset.

    This class provides a way to access paired data consisting of images
    and corresponding sequences of tokens. Each sample returned by the
    dataset is a tuple (image_tensor, token_tensor).
    """

    def __init__(self, df, vocab, tokenized_formulas, transform=None):
        """
        Initializes the dataset with data and preprocessing information.

        Args:
            df: DataFrame containing image data.
            vocab: Vocabulary object with a `numericalize` method for converting tokens to indices.
            tokenized_formulas: List of tokenized sequences corresponding to each image.
            transform: Optional transformations to apply to images (e.g., resizing, normalization).
        """
        self.df = df
        self.vocab = vocab
        self.transform = transform
        self.formulas = tokenized_formulas

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        This is required for PyTorch DataLoader to determine dataset size.
        """
        return len(self.df)

    def __getitem__(self, index):
        """
        Retrieves a single sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            image (torch.Tensor): Image converted to a tensor and optionally transformed.
            tokens_idx (torch.Tensor): Sequence of token indices as a tensor.
        """
        # Load image bytes from the DataFrame and convert to PIL Image
        img_bytes = self.df.iloc[index]["image"]["bytes"]
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Retrieve tokenized sequence for this sample
        tokens = self.formulas[index]

        # Convert tokens to numerical indices using the vocabulary
        tokens_idx = self.vocab.numericalize(tokens)

        # Convert list of indices to a PyTorch tensor
        tokens_idx = torch.tensor(tokens_idx, dtype=torch.long)

        return image, tokens_idx
