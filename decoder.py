import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

class TransformerDecoder(nn.Module):
    """
    Custom Transformer decoder module.

    This module implements a multi-layer Transformer decoder. It takes
    a sequence of token indices and a memory tensor as input, applies
    embedding and positional encoding, passes the sequence through
    stacked Transformer decoder layers, and outputs logits over the vocabulary.
    """

    def __init__(self, vocab_size, d_model, nhead, num_layers):
        """
        Initializes the decoder.

        Args:
            vocab_size (int): Size of the vocabulary (number of unique tokens).
            d_model (int): Dimension of the embedding and hidden layers.
            nhead (int): Number of attention heads in the multi-head attention mechanism.
            num_layers (int): Number of TransformerDecoder layers to stack.
        """
        super().__init__()

        # Token embedding layer converts indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Learnable positional encoding to provide sequence order information
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))

        # Single Transformer decoder layer
        layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)

        # Stacked Transformer decoder
        self.decoder = nn.TransformerDecoder(layer, num_layers)

        # Linear layer to project decoder outputs to vocabulary logits
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask):
        """
        Performs a forward pass through the decoder.

        Args:
            tgt (torch.Tensor): Target sequence tensor of shape (batch_size, seq_len)
                                containing token indices.
            memory (torch.Tensor): Encoder output tensor of shape (batch_size, mem_len, d_model)
                                   used as context for the decoder.
            tgt_mask (torch.Tensor): Mask to prevent attention to future tokens in the sequence.

        Returns:
            torch.Tensor: Logits over the vocabulary of shape (batch_size, seq_len, vocab_size).
        """
        # Apply token embedding and add positional encoding
        tgt_emb = self.embedding(tgt) + self.pos_encoding[:, :tgt.size(1), :]

        # Pass through the stacked Transformer decoder layers
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)

        # Project hidden states to vocabulary logits
        return self.fc_out(output)