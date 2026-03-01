import torch
from torch.utils.data import DataLoader
from utils import (
    get_final_tokens,
    collate_fn,
    evaluate_official_metrics,
    clean_formula,
    predict_beam_search,
)
from model import Im2LatexModel
import sys
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.transforms as T
from vocabulary import Vocabulary
from model import Im2LatexModel
from Im2LatexDataset import Im2LatexDataset


def main():
    print("loading ...")
    base_url = "hf://datasets/yuntian-deng/im2latex-100k/"
    splits = {
        "train": "data/train-00000-of-00001-93885635ef7c6898.parquet",
        "test": "data/test-00000-of-00001-fce261550cd3f5db.parquet",
        "val": "data/val-00000-of-00001-3f88ebb0c1272ccf.parquet",
    }

    train_df = pd.read_parquet(base_url + splits["train"])
    val_df = pd.read_parquet(base_url + splits["val"])
    test_df = pd.read_parquet(base_url + splits["test"])
    print(train_df.size)

    print("First 5 records: \n", train_df.head())
    print("First 5 records formulas: \n", train_df.head().formula)

    transform = T.Compose(
        [T.Resize((64, 320)), T.ToTensor(), T.Normalize((0.5,), (0.5,))]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Building vocabulary...")
    tokenized_train = [get_final_tokens(f) for f in train_df.formula.values]
    tokenized_val = [get_final_tokens(f) for f in val_df.formula.values]

    vocab = Vocabulary()
    vocab.build_vocabulary(tokenized_train)

    train_dataset = Im2LatexDataset(
        train_df, vocab, tokenized_train, transform=transform
    )
    val_dataset = Im2LatexDataset(val_df, vocab, tokenized_val, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    print("Initializing model...")
    model = Im2LatexModel(vocab=vocab, device=device)

    model.train_model(train_loader=train_loader, epochs=5, val_loader=val_loader)

    example_img, _ = next(iter(val_loader))
    example_img = example_img[0].unsqueeze(0)

    tokens = model.predict(example_img)

    print("Prediction:", " ".join(tokens))

    # 1. Prepara i token del test set
    print("Tokenizing test set...")
    tokenized_test = [get_final_tokens(f) for f in test_df.formula.values]
    test_dataset = Im2LatexDataset(test_df, vocab, tokenized_test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 2. Prendi un'immagine di esempio dal test set
    example_img, example_tgt = next(iter(test_loader))
    example_img = example_img.to(device)

    # 3. Confronto Greedy vs Beam
    print("--- CONFRONTO SUL TEST SET ---")
    greedy_tokens = model.predict(example_img)
    beam_tokens = model.predict_beam_search(example_img)

    print(f"REAL:   {clean_formula(example_tgt[0], vocab)}")
    print(f"GREEDY: {clean_formula(greedy_tokens, vocab)}")
    print(f"BEAM:   {clean_formula(beam_tokens, vocab)}")

    evaluate_official_metrics(model, test_loader, vocab, num_samples=100)


if __name__ == "__main__":
    main()
