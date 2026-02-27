# Standard imports
import sys
print(sys.executable)

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import pandas as pd
from collections import Counter
import io
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn as nn

from pylatexenc.latexwalker import LatexWalker, LatexCharsNode, LatexMacroNode, LatexGroupNode

torch.cuda.is_available()

print("loading ...")
base_url = "hf://datasets/yuntian-deng/im2latex-100k/"
splits = {
    'train': 'data/train-00000-of-00001-93885635ef7c6898.parquet', 
    'test': 'data/test-00000-of-00001-fce261550cd3f5db.parquet', 
    'val': 'data/val-00000-of-00001-3f88ebb0c1272ccf.parquet'
}

train_df = pd.read_parquet(base_url + splits["train"])
val_df   = pd.read_parquet(base_url + splits["val"])
test_df  = pd.read_parquet(base_url + splits["test"])
print(train_df.size)

print("First 5 records: \n", train_df.head())
print("First 5 records formulas: \n", train_df.head().formula)

from collections import Counter

class Vocabulary:
    def __init__(self):
        # Mapping from index to token (word)
        # itos stands for "index to string" and is used to convert numerical predictions back to words.
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        # Mapping from token to index
        # stoi stands for "string to index" and is used to convert words to numerical values for model input.
        self.stoi = {v: k for k, v in self.itos.items()}
        # Minimum frequency threshold for including a word in the vocabulary.
        # Words that appear less than this threshold will be treated as "<unk>" (unknown).
        self.threshold = 1

    def build_vocabulary(self, sentence_list):
        """
        Builds vocabulary from a list of sentences.
        
        Theoretical explanation:
        - NLP models (like RNNs, Transformers) cannot process raw text; they require numerical input.
        - Vocabulary maps each word to a unique index.
        - Special tokens:
            <pad> : padding token to make sequences of equal length
            <sos> : start-of-sequence token
            <eos> : end-of-sequence token
            <unk> : token for unknown or rare words
        - Thresholding removes rare words to limit vocabulary size and reduce noise.
        """
        frequencies = Counter()  # Counter to store word frequencies
        idx = 4  # Start indexing new words from 4 (0-3 reserved for special tokens)

        # Count frequency of each word in the sentences
        for sentence in sentence_list:
            for word in sentence:
                # Skip special tokens; they are already in the vocabulary
                if word not in ["<pad>", "<sos>", "<eos>", "<unk>"]:
                    frequencies[word] += 1

        # Add words that meet the frequency threshold to the vocabulary
        for word, freq in frequencies.items():
            if freq >= self.threshold:
                self.stoi[word] = idx  # Map word -> index
                self.itos[idx] = word  # Map index -> word
                idx += 1

    def numericalize(self, text):
        """
        Converts a list of tokens into a list of numerical indices based on the vocabulary.
        
        Theoretical explanation:
        - Models cannot process strings directly; text must be converted to numbers.
        - Unknown words (words not in vocabulary) are mapped to "<unk>" index.
        - This process is called "numericalization" or "tokenization".
        """
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in text]

def parse_nodes(nodes):
    """
    Recursively parses a list of LaTeX nodes and flattens them into individual tokens.
    
    Theoretical explanation:
    - LaTeX formulas are represented as a tree of nodes (chars, macros, groups).
    - To feed them into an NLP model, we need a **flat sequence of tokens**.
    - This function converts the hierarchical LaTeX structure into a linear sequence.
    """
    flat_tokens = []

    for node in nodes:
        if node is None: 
            continue  # Skip empty nodes

        if node.isNodeType(LatexCharsNode):
            # Simple characters like letters, numbers, operators (a, b, =, +, etc.)
            # Each character can be treated as a separate token
            for c in node.chars:
                if not c.isspace():  # Ignore whitespace
                    flat_tokens.append(c)

        elif node.isNodeType(LatexMacroNode):
            # LaTeX commands like \frac, \alpha, \cal, etc.
            m_name = node.macroname
            token = "\\" + m_name if m_name else "\\\\"  # prepend backslash for macro
            if token.strip() == "\\":  # ignore empty backslashes
                continue
            flat_tokens.append(token)

        elif node.isNodeType(LatexGroupNode):
            # Groups enclosed in { ... } 
            # We treat { and } as tokens and recursively parse the contents
            flat_tokens.append("{")
            flat_tokens.extend(parse_nodes(node.nodelist))  # recursive call
            flat_tokens.append("}")

    return flat_tokens


def get_final_tokens(formula):
    """
    Converts a LaTeX formula string into a flat list of tokens, adding start and end tokens.
    
    Theoretical explanation:
    - NLP models (e.g., sequence models) require start-of-sequence (<sos>) and end-of-sequence (<eos>) tokens.
    - LatexWalker parses the formula into a structured tree of nodes.
    - parse_nodes flattens the tree into individual character and macro tokens.
    """
    walker = LatexWalker(rf"{formula}")  # Parse LaTeX formula into a syntax tree

    try:
        # Extract nodes from the formula
        (nodes, pos, len_) = walker.get_latex_nodes()
    except:
        # If parsing fails, return empty token list
        return []

    # Add <sos> at the start and <eos> at the end for sequence modeling
    return ["<sos>"] + parse_nodes(nodes) + ["<eos>"]

class Im2LatexDataset(Dataset):
    """
    PyTorch Dataset for the Im2Latex task.
    
    Theoretical explanation:
    - Each sample consists of an image of a formula and its corresponding LaTeX tokens.
    - Images must be preprocessed (resized, normalized) for neural network input.
    - LaTeX formulas are tokenized and numericalized to feed the model.
    """
    def __init__(self, df, vocab, tokenized_formulas, transform=None):
        """
        Parameters:
        - df: DataFrame containing images (as bytes) and other metadata
        - vocab: Vocabulary object to convert LaTeX tokens to indices
        - tokenized_formulas: list of tokenized LaTeX formulas corresponding to each image
        - transform: optional image transformations (resize, normalize, etc.)
        """
        self.df = df
        self.vocab = vocab
        self.transform = transform
        self.formulas = tokenized_formulas

    def __len__(self):
        # Returns the total number of samples
        return len(self.df)

    def __getitem__(self, index):
        """
        Returns:
        - image: preprocessed image tensor
        - tokens_idx: numericalized LaTeX formula as a tensor of indices
        """
        # Load image from bytes and convert to RGB
        img_bytes = self.df.iloc[index]['image']['bytes']
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Get tokenized formula and convert to numerical indices
        tokens = self.formulas[index]
        tokens_idx = self.vocab.numericalize(tokens)
        tokens_idx = torch.tensor(tokens_idx, dtype=torch.long)  # Convert to PyTorch tensor
        
        return image, tokens_idx
    

# Image transformations
# Theoretical explanation:
# - Resize ensures all images have the same dimensions (required for batch processing)
# - ToTensor converts PIL images to PyTorch tensors
# - Normalize scales pixel values to [-1, 1] (helps with model training stability)
transform = T.Compose([
    T.Resize((64, 320)),  # Resize to Height=64, Width=320
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))  # Normalize to mean=0.5, std=0.5
])


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    
    Theoretical explanation:
    - In a batch, formulas can have variable lengths.
    - To create a single tensor, we pad shorter sequences with zeros.
    - Returns a batch of images and a padded batch of token sequences.
    """
    imgs, captions = zip(*batch)  # Unzip images and token sequences
    imgs = torch.stack(imgs)       # Stack images into a batch tensor

    # Get lengths of each caption
    lengths = [len(cap) for cap in captions]
    max_len = max(lengths)         # Maximum length in the batch

    # Initialize a padded tensor of shape (batch_size, max_len)
    padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded[i, :len(cap)] = cap  # Copy actual token indices, pad the rest with 0 (<pad>)

    return imgs, padded

class CnnEncoder(nn.Module):
    """
    CNN-based encoder for Im2Latex (image-to-LaTeX) task.

    Theoretical explanation:
    - CNN extracts spatial features from the input image.
    - The output of the CNN is a feature map representing visual information of the formula.
    - This feature map is then flattened into a sequence for the Transformer decoder.
    - The Transformer requires input in the shape [Batch, Sequence Length, Feature Dimension].
    """
    def __init__(self, d_model):
        """
        Parameters:
        - d_model: the dimension of the feature vectors that will be fed into the Transformer
        """
        super().__init__()

        # Simple CNN architecture (inspired by a reduced ResNet)
        # Each Conv2d layer extracts features, BatchNorm stabilizes training,
        # ReLU adds non-linearity, MaxPool reduces spatial dimensions
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, d_model, 3, 1, 1), nn.BatchNorm2d(d_model), nn.ReLU(), nn.MaxPool2d((2, 2))
        )

    def forward(self, x):
        """
        Forward pass:
        - x: input image tensor of shape [Batch, 3, Height, Width]
        - features: output tensor of shape [Batch, Sequence Length, Feature Dimension]
        
        Theoretical explanation:
        - CNN produces a feature map of shape [Batch, d_model, H', W'].
        - We flatten the spatial dimensions (H'*W') to create a sequence for the Transformer.
        - Permute the tensor to shape [Batch, Sequence Length, d_model] to match the expected input of the Transformer decoder.
        """
        # Apply convolutional layers
        features = self.conv(x)  # Shape: [Batch, d_model, 4, 20] (example for 64x320 input)

        # Flatten spatial dimensions (H*W) and transpose to [Batch, Seq_len, d_model]
        features = features.flatten(2).permute(0, 2, 1)  # 4*20 = 80 sequence length

        return features

class TransformerDecoder(nn.Module):
    """
    Transformer-based decoder for Im2Latex (image-to-LaTeX) task.

    Theoretical explanation:
    - Takes as input the CNN features (memory) and previously generated LaTeX tokens.
    - Produces a probability distribution over the vocabulary for the next token at each position.
    - Uses self-attention to model dependencies between previously generated tokens.
    - Uses cross-attention to focus on relevant parts of the image features (memory).
    """
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        """
        Parameters:
        - vocab_size: number of tokens in the vocabulary
        - d_model: embedding dimension / feature dimension
        - nhead: number of attention heads
        - num_layers: number of Transformer decoder layers
        """
        super().__init__()

        # Token embedding: converts token indices into dense vectors
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding: adds information about the position of tokens in the sequence
        # Here we use a learnable positional encoding instead of sinusoidal
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))  # [1, max_seq_len, d_model]

        # Standard Transformer decoder layer
        # - Self-attention: considers previous tokens in the sequence
        # - Multi-head attention: allows model to focus on multiple aspects simultaneously
        # - Feed-forward: adds non-linearity and transforms features
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output linear layer: maps decoder output to vocabulary probabilities
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask):
        """
        Forward pass:
        - tgt: target token indices [Batch, Seq_Len] (input sequence so far)
        - memory: CNN image features [Batch, 80, d_model] (encoder output)
        - tgt_mask: mask for self-attention to prevent looking at future tokens

        Theoretical explanation:
        - Embedding: convert tokens to vectors
        - Add positional encoding: provides order information to the model
        - TransformerDecoder:
            - Self-attention: models dependencies between previous LaTeX tokens
            - Cross-attention: attends to image features to generate relevant tokens
        - Output: linear projection to vocabulary probabilities
        """
        # Embed target tokens and add positional encodings
        tgt_emb = self.embedding(tgt) + self.pos_encoding[:, :tgt.size(1), :]

        # Apply Transformer decoder
        output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)

        # Map output to vocabulary logits
        return self.fc_out(output)

class Im2LatexModel(nn.Module):
    """
    Full Image-to-LaTeX model combining a CNN encoder and a Transformer decoder.

    Theoretical explanation:
    - This is an encoder-decoder architecture, commonly used for sequence generation tasks.
    - Encoder (CNN) extracts spatial features from an image.
    - Decoder (Transformer) generates LaTeX tokens sequentially, attending to the image features.
    - During training, we feed the ground-truth tokens; during inference, we generate tokens autoregressively.
    """
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
        """
        Parameters:
        - vocab_size: number of tokens in the LaTeX vocabulary
        - d_model: feature dimension for CNN output and Transformer embeddings
        - nhead: number of attention heads in the Transformer
        - num_layers: number of Transformer decoder layers
        """
        super().__init__()
        
        # 1. ENCODER (CNN)
        # Takes an image of shape [Batch, 3, 64, 320]
        # Returns visual features [Batch, Seq_len=80, d_model]
        self.encoder_cnn = CnnEncoder(d_model) 
        
        # 2. DECODER (Transformer)
        # Takes LaTeX token sequence and CNN features (memory)
        # Predicts the next token at each sequence position
        self.decoder_transformer = TransformerDecoder(vocab_size, d_model, nhead, num_layers)

    def forward(self, src_img, tgt_tokens):
        """
        Forward pass:
        - src_img: input images [Batch, 3, 64, 320]
        - tgt_tokens: token indices for target LaTeX formulas [Batch, Seq_Len]
        - Returns logits over the vocabulary for each position: [Batch, Seq_Len, Vocab_Size]
        """

        # A. Visual feature extraction (Encoder)
        # The CNN converts images into a sequence of feature vectors
        memory = self.encoder_cnn(src_img)  # Output shape: [Batch, 80, d_model]
        
        # B. Create target mask for the Decoder
        # Prevents the decoder from attending to future tokens (causal masking)
        device = tgt_tokens.device
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_tokens.size(1)
        ).to(device)
        
        # C. Generate output via Transformer decoder
        # The decoder uses previous tokens (tgt_tokens) and CNN memory
        output = self.decoder_transformer(tgt_tokens, memory, tgt_mask)
        
        # Output shape: [Batch, Seq_Len, Vocab_Size], ready for softmax
        return output
 # =====================
# MINI DATASET DEBUG
# =====================

# Optional: sample small subset for debugging / fast runs
# train_df = train_df.sample(1000, random_state=2000)
# val_df   = val_df.sample(100, random_state=2000)

# Print dataset sizes
print(f"TRAIN SIZE: {train_df.size}")
print(f"TEST SIZE: {val_df.size}")

# TOKENIZATION
print("Tokenizing training formulas...")
# Convert LaTeX strings into token sequences with <sos>/<eos>
tokenized_train = [get_final_tokens(f) for f in train_df.formula.values]
tokenized_val = [get_final_tokens(f) for f in val_df.formula.values]

# Track max sequence length for reference
lengths = [len(f) for f in tokenized_train]
print(f"Maximum sequence length in training set: {max(lengths)}")

# VOCABULARY BUILDING
# Create vocabulary from training tokens (defines final vocab size)
vocab = Vocabulary()
vocab.build_vocabulary(tokenized_train)
new_vocab_size = len(vocab.stoi)
print(f"New vocabulary size: {new_vocab_size}") 

# CREATE DATASETS AND DATALOADERS
train_dataset = Im2LatexDataset(train_df, vocab, tokenized_train, transform=transform)
val_dataset   = Im2LatexDataset(val_df, vocab, tokenized_val, transform=transform)

# DataLoader batches sequences, pads to max length in batch
# Use small batch_size if CPU or limited GPU memory
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# SANITY CHECK: ensure token indices do not exceed vocab size
max_found = 0
for i, (imgs, caps) in enumerate(train_loader):
    if i > 100: break # Check only first 100 batches
    if caps.max() >= new_vocab_size:
        print(f"!!! ERROR IN BATCH {i} !!! Max index: {caps.max()}")
        break
    max_found = max(max_found, caps.max().item())
print(f"Sanity check done. Max index: {max_found} / Vocab size: {new_vocab_size}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# MODEL INITIALIZATION
model = Im2LatexModel(vocab_size=new_vocab_size).to(device)

# Optimizer and Loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore <pad> in loss


def clean_formula(tensor, vocab):
    """
    Decodes a tensor of token indices into a LaTeX string.
    Removes special tokens (<pad>, <sos>, <eos>).
    """
    if isinstance(tensor, torch.Tensor):
        tokens = [vocab.itos[idx.item()] for idx in tensor]
    else:
        tokens = tensor
    
    tokens = [t for t in tokens if t not in ("<pad>", "<sos>", "<eos>")]
    formula = " ".join(tokens)
    
    return formula.strip()

def train_epoch(model, dataloader, optimizer, criterion):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0
    
    for i, (imgs, captions) in enumerate(dataloader):
        imgs, captions = imgs.to(device), captions.to(device)
        
        # Prepare input and target sequences for decoder
        # tgt_input: <sos> to penultimate token
        # tgt_expected: second token to <eos>
        tgt_input = captions[:, :-1]
        tgt_expected = captions[:, 1:]
        
        # Skip empty sequences
        if tgt_input.size(1) == 0: 
            continue

        # Forward pass
        preds = model(imgs, tgt_input)  # [Batch, Seq, Vocab]
        
        # Compute loss: flatten for CrossEntropy
        loss = criterion(preds.reshape(-1, preds.shape[-1]), tgt_expected.reshape(-1))
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()

        if i % 100 == 0:
           print(f"Batch {i}/{len(dataloader)} - Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    """
    Evaluates model on validation set.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, captions in dataloader:
            imgs, captions = imgs.to(device), captions.to(device)
            tgt_input = captions[:, :-1]
            tgt_expected = captions[:, 1:]
            
            preds = model(imgs, tgt_input)
            loss = criterion(preds.reshape(-1, preds.shape[-1]), tgt_expected.reshape(-1))
            total_loss += loss.item()
            
    return total_loss / len(dataloader)


def predict(model, img, vocab, max_len=150):
    """
    Generates LaTeX token sequence from a single image using greedy decoding.
    """
    model.eval()
    device = img.device
    vocab_size = len(vocab.stoi)

    with torch.no_grad():
        # Encode image
        memory = model.encoder_cnn(img)  # [1, Seq_len, d_model]

        # Initialize sequence with <sos>
        ys = torch.full((1, 1), vocab.stoi["<sos>"], dtype=torch.long, device=device)

        for _ in range(max_len):
            # Create causal mask
            tgt_mask = torch.triu(
                torch.ones(ys.size(1), ys.size(1), device=device) * float('-inf'),
                diagonal=1
            )

            # Decoder forward
            out = model.decoder_transformer(ys, memory, tgt_mask=tgt_mask)  # [1, seq_len, vocab_size]

            # Greedy selection: pick most probable token
            next_word = out[:, -1, :].argmax(dim=-1).item()
            next_word = max(0, min(next_word, vocab_size - 1))

            # Append to sequence
            ys = torch.cat([ys, torch.tensor([[next_word]], device=device)], dim=1)

            # Stop if <eos> generated
            if next_word == vocab.stoi["<eos>"]:
                break

        # Convert indices to tokens
        tokens = [vocab.itos[idx.item()] for idx in ys[0]]
        return tokens

num_epochs = 12
log_file = "training_log.log"

with open(log_file, "w") as f:
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)

        log_line = f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        print(log_line)
        f.write(log_line + "\n")

    # Example prediction after training
    example_img, example_tgt = next(iter(val_loader))
    example_img = example_img[0].unsqueeze(0).to(device)

    pred_tokens = predict(model, example_img, vocab)
    prediction = clean_formula(pred_tokens, vocab)
    real_formula = clean_formula(example_tgt[0], vocab)
    
    final_example = f"\nFinal Example:\nReal: {real_formula}\nPred: {prediction}\n"
    print(final_example)
    f.write(final_example)

def predict_beam_search(model, img, vocab, k=3, max_len=150):
    """
    Beam search decoding for more accurate predictions.
    
    - Keeps k best sequences at each step
    - Explores multiple candidate sequences simultaneously
    """
    model.eval()
    device = img.device
    vocab_size = len(vocab.stoi)
    
    with torch.no_grad():
        # Encode image
        memory = model.encoder_cnn(img)

        # Initialize beams: (log_score, sequence of indices)
        beams = [(0.0, [vocab.stoi["<sos>"]])]

        for _ in range(max_len):
            new_beams = []
            for score, seq in beams:
                if seq[-1] == vocab.stoi["<eos>"]:
                    new_beams.append((score, seq))
                    continue

                ys = torch.tensor([seq], device=device)
                sz = ys.size(1)
                tgt_mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
                
                out = model.decoder_transformer(ys, memory, tgt_mask=tgt_mask)
                log_probs = torch.log_softmax(out[:, -1, :], dim=-1).squeeze(0)

                topk_probs, topk_idx = log_probs.topk(k)
                for i in range(k):
                    new_beams.append((score + topk_probs[i].item(), seq + [topk_idx[i].item()]))
            
            # Keep top k beams
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:k]
            if all(s[-1] == vocab.stoi["<eos>"] for _, s in beams):
                break
        
        # Return best sequence
        best_seq = beams[0][1]
        return [vocab.itos[idx] for idx in best_seq]

# Tokenize test set and create DataLoader
print("Tokenizing test set...")
tokenized_test = [get_final_tokens(f) for f in test_df.formula.values]
test_dataset = Im2LatexDataset(test_df, vocab, tokenized_test, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Compare greedy vs beam for an example image
example_img, example_tgt = next(iter(test_loader))
example_img = example_img.to(device)

print("--- GREEDY VS BEAM COMPARISON ---")
greedy_tokens = predict(model, example_img, vocab)
beam_tokens = predict_beam_search(model, example_img, vocab, k=3)

print(f"REAL:   {clean_formula(example_tgt[0], vocab)}")
print(f"GREEDY: {clean_formula(greedy_tokens, vocab)}")
print(f"BEAM:   {clean_formula(beam_tokens, vocab)}")

def evaluate_official_metrics(model, dataloader, vocab, num_samples=100, k=3):
    """
    Computes standard metrics on test set:
    - Exact Match (EM)
    - BLEU Score
    - Normalized Edit Distance
    """
    model.eval()
    em_count = 0
    bleu_scores = []
    edit_distances = []
    smoothie = SmoothingFunction().method4

    print(f"Evaluating on {num_samples} samples...")

    with torch.no_grad():
        for i, (img, tgt) in enumerate(dataloader):
            if i >= num_samples: 
                break
            img = img.to(device)
            
            # Beam search generation
            pred_tokens = predict_beam_search(model, img, vocab, k=k)
            pred_str = clean_formula(pred_tokens, vocab)
            real_str = clean_formula(tgt[0], vocab)

            # Exact Match
            if pred_str.replace(" ", "") == real_str.replace(" ", ""):
                em_count += 1

            # BLEU Score
            p_tokens = pred_str.split()
            r_tokens = real_str.split()
            b_score = sentence_bleu([r_tokens], p_tokens, smoothing_function=smoothie)
            bleu_scores.append(b_score)

            # Normalized Edit Distance
            max_len = max(len(pred_str), len(real_str))
            norm_ed = 1 - (edit_distance(pred_str, real_str) / max_len) if max_len > 0 else 1.0
            edit_distances.append(norm_ed)

    # Print final report
    print("\n" + "="*40)
    print(f"FINAL TEST SET REPORT")
    print(f"Exact Match (EM):     {em_count/num_samples*100:.2f}%")
    print(f"BLEU Score:           {np.mean(bleu_scores):.4f}")
    print(f"Edit Distance (Text): {np.mean(edit_distances)*100:.2f}%")
    print("="*40)

# Run official evaluation
evaluate_official_metrics(model, test_loader, vocab, num_samples=100)