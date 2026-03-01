import torch
from torch.utils.data import DataLoader
from utils import (
    get_final_tokens,
    collate_fn,
    evaluate_official_metrics,
    clean_formula
)
from model import Im2LatexModel
import pandas as pd
import torchvision.transforms as T
from vocabulary import Vocabulary
from dataset import Im2LatexDataset
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_URL = "hf://datasets/yuntian-deng/im2latex-100k/"
SPLITS = {
    "train": "data/train-00000-of-00001-93885635ef7c6898.parquet",
    "val": "data/val-00000-of-00001-3f88ebb0c1272ccf.parquet",
    "test": "data/test-00000-of-00001-fce261550cd3f5db.parquet",
}

IMAGE_SIZE = (64, 320)
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 1
EPOCHS = 20
NUM_TEST_SAMPLES = 100

NORMALIZE_MEAN = (0.5,)
NORMALIZE_STD = (0.5,)

def main():
    print(f"Using device: {device}\n")
    print("Loading datasets...")

    train_df = pd.read_parquet(BASE_URL + SPLITS["train"])
    val_df = pd.read_parquet(BASE_URL + SPLITS["val"])
    test_df = pd.read_parquet(BASE_URL + SPLITS["test"])

    print("Datasets loaded successfully.\n")
    print(f"Train dataset size: {train_df.shape}")
    print(f"Validation dataset size: {val_df.shape}")
    print(f"Test dataset size: {test_df.shape}\n")

    transform = T.Compose(
        [
            T.Resize(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )

    print("Building vocabulary from training data...")
    tokenized_train = [get_final_tokens(f) for f in train_df.formula.values]
    tokenized_val = [get_final_tokens(f) for f in val_df.formula.values]

    vocab = Vocabulary()
    vocab.build_vocabulary(tokenized_train)

    print("Preparing datasets and data loaders...")
    train_dataset = Im2LatexDataset(train_df, vocab, tokenized_train, transform=transform)
    val_dataset = Im2LatexDataset(val_df, vocab, tokenized_val, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=False, collate_fn=collate_fn
    )
    print("Data loaders ready.\n")

    print("Initializing model...")
    model = Im2LatexModel(vocab=vocab)
    print("Starting training...\n")
    history = model.train_model(train_loader=train_loader, epochs=EPOCHS, val_loader=val_loader)
    print("Training completed.\n")

    print("Drawing learning curve...\n")

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)

    plt.savefig("learning_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Learning curve saved.\n")

    print("Preparing test set for evaluation...")
    tokenized_test = [get_final_tokens(f) for f in test_df.formula.values]
    test_dataset = Im2LatexDataset(test_df, vocab, tokenized_test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)
    print("Test dataset ready.\n")

    print(f"Evaluating model on test set with official metrics (sample of {NUM_TEST_SAMPLES})...")
    evaluate_official_metrics(model, test_loader, vocab, num_samples=NUM_TEST_SAMPLES)
    print("Evaluation completed.")


if __name__ == "__main__":
    main()