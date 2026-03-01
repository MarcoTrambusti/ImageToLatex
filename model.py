import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from tqdm import tqdm
from encoder import CnnEncoder
from decoder import TransformerDecoder

class Im2LatexModel(nn.Module):
    """
    Complete encoder-decoder model using a CNN encoder and a Transformer decoder.

    The model integrates:
        - CnnEncoder: extracts features from input images.
        - TransformerDecoder: generates sequences from encoder features.
        - Training utilities: loss function, optimizer, training loop.
        - Prediction methods: greedy decoding and beam search.
    """

    def __init__(self, vocab, d_model=256, nhead=8, num_layers=4, lr=5e-5):
        """
        Initializes the model components and training utilities.

        Args:
            vocab: Vocabulary object containing stoi (str→index) and itos (index→str).
            d_model (int): Dimension of the encoder output and decoder embeddings.
            nhead (int): Number of attention heads in the Transformer decoder.
            num_layers (int): Number of Transformer decoder layers.
            lr (float): Learning rate for the optimizer.
        """
        super().__init__()

        self.vocab = vocab
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encoder extracts feature maps from images
        self.encoder = CnnEncoder(d_model)

        # Decoder generates sequences from encoder memory
        self.decoder = TransformerDecoder(len(vocab.stoi), d_model, nhead, num_layers)

        # Move model to device
        self.to(self.device)

        # Loss function (ignore padding index 0)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Optimizer for model parameters
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, images, tgt_tokens):
        """
        Forward pass through encoder and decoder.

        Args:
            images (torch.Tensor): Input image tensor.
            tgt_tokens (torch.Tensor): Target token sequences for decoding.

        Returns:
            torch.Tensor: Decoder output logits over vocabulary.
        """
        # Extract features from images
        memory = self.encoder(images)

        # Generate causal mask for decoder to prevent attending to future tokens
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_tokens.size(1)
        ).to(self.device)

        # Pass through decoder
        return self.decoder(tgt_tokens, memory, tgt_mask)

    def train_model(self, train_loader=None, epochs=None, val_loader=None):
        """
        Custom training loop for the model.

        Args:
            train_loader: PyTorch DataLoader for training data.
            epochs: Number of epochs to train.
            val_loader: Optional DataLoader for validation data.

        Returns:
            history (dict): Training and validation loss per epoch.
        """
        if train_loader is None:
            return super().train()

        train_losses = []
        val_losses = []

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            super().train()  # Set model to training mode
            total_train_loss = 0

            train_bar = tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False
            )

            for imgs, captions in train_bar:
                imgs = imgs.to(self.device)
                captions = captions.to(self.device)

                # Shift tokens: input vs expected
                tgt_input = captions[:, :-1]
                tgt_expected = captions[:, 1:]

                if tgt_input.size(1) == 0:
                    continue

                # Forward pass
                preds = self.forward(imgs, tgt_input)

                # Compute loss
                loss = self.criterion(
                    preds.reshape(-1, preds.shape[-1]), tgt_expected.reshape(-1)
                )

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optimizer.step()

                total_train_loss += loss.item()
                train_bar.set_postfix(loss=loss.item())

            avg_train_loss = total_train_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)

            # Validation loop if provided
            if val_loader is not None:
                super().eval()  # Set model to evaluation mode
                total_val_loss = 0

                val_bar = tqdm(
                    val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False
                )

                with torch.no_grad():
                    for imgs, captions in val_bar:
                        imgs = imgs.to(self.device)
                        captions = captions.to(self.device)

                        tgt_input = captions[:, :-1]
                        tgt_expected = captions[:, 1:]

                        preds = self.forward(imgs, tgt_input)

                        loss = self.criterion(
                            preds.reshape(-1, preds.shape[-1]), tgt_expected.reshape(-1)
                        )

                        total_val_loss += loss.item()
                        val_bar.set_postfix(val_loss=loss.item())

                avg_val_loss = total_val_loss / len(val_loader)
                history["val_loss"].append(avg_val_loss)

                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Train: {avg_train_loss:.4f} | "
                    f"Val: {avg_val_loss:.4f}"
                )
            else:
                print(f"Epoch [{epoch+1}/{epochs}] Train: {avg_train_loss:.4f}")

        return history

    def predict(self, img, max_len=150):
        """
        Greedy decoding of a single input image.

        Args:
            img (torch.Tensor): Input image tensor.
            max_len (int): Maximum output sequence length.

        Returns:
            List[str]: Predicted token sequence as strings.
        """
        super().eval()
        img = img.to(self.device)

        with torch.no_grad():
            memory = self.encoder(img)

            # Start-of-sequence token
            ys = torch.full(
                (1, 1), self.vocab.stoi["<sos>"], dtype=torch.long, device=self.device
            )

            # Generate tokens one by one
            for _ in range(max_len):
                tgt_mask = torch.triu(
                    torch.ones(ys.size(1), ys.size(1), device=self.device)
                    * float("-inf"),
                    diagonal=1,
                )

                out = self.decoder(ys, memory, tgt_mask)
                next_token = out[:, -1, :].argmax(dim=-1).item()

                ys = torch.cat(
                    [ys, torch.tensor([[next_token]], device=self.device)], dim=1
                )

                if next_token == self.vocab.stoi["<eos>"]:
                    break

        # Convert indices to tokens
        return [self.vocab.itos[idx.item()] for idx in ys[0]]

    def predict_beam_search(self, img, k=3, max_len=150):
        """
        Beam search decoding for a single input image.

        Args:
            img (torch.Tensor): Input image tensor.
            k (int): Beam width.
            max_len (int): Maximum output sequence length.

        Returns:
            List[str]: Predicted token sequence using beam search.
        """
        super().eval()
        device = img.device
        vocab_size = len(self.vocab.stoi)

        with torch.no_grad():
            memory = self.encoder(img)
            beams = [(0.0, [self.vocab.stoi["<sos>"]])]

            for _ in range(max_len):
                new_beams = []
                for score, seq in beams:
                    if seq[-1] == self.vocab.stoi["<eos>"]:
                        new_beams.append((score, seq))
                        continue

                    ys = torch.tensor([seq], device=device)
                    sz = ys.size(1)
                    tgt_mask = torch.triu(
                        torch.ones(sz, sz, device=device) * float("-inf"), diagonal=1
                    )

                    out = self.decoder(ys, memory, tgt_mask=tgt_mask)
                    log_probs = torch.log_softmax(out[:, -1, :], dim=-1).squeeze(0)

                    topk_probs, topk_idx = log_probs.topk(k)
                    for i in range(k):
                        new_beams.append(
                            (score + topk_probs[i].item(), seq + [topk_idx[i].item()])
                        )

                # Keep top-k beams
                beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:k]

                # Stop if all beams ended with EOS
                if all(s[-1] == self.vocab.stoi["<eos>"] for _, s in beams):
                    break

            best_seq = beams[0][1]
            return [self.vocab.itos[idx] for idx in best_seq]

    def save(self, path):
        """Save model state dictionary to the specified path."""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model state dictionary from the specified path."""
        self.load_state_dict(torch.load(path, map_location=self.device))