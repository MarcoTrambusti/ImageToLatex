import numpy as np
import torch
from pylatexenc.latexwalker import (
    LatexWalker,
    LatexCharsNode,
    LatexMacroNode,
    LatexGroupNode,
)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.metrics import edit_distance
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(batch):
    imgs, captions = zip(*batch)
    imgs = torch.stack(imgs)

    lengths = [len(cap) for cap in captions]
    max_len = max(lengths)

    padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded[i, : len(cap)] = cap

    return imgs, padded


def parse_nodes(nodes):
    flat_tokens = []
    for node in nodes:
        if node is None:
            continue

        if node.isNodeType(LatexCharsNode):

            for c in node.chars:
                if not c.isspace():
                    flat_tokens.append(c)

        elif node.isNodeType(LatexMacroNode):
            m_name = node.macroname
            token = "\\" + m_name if m_name else "\\\\"
            if token.strip() == "\\":
                continue
            flat_tokens.append(token)

        elif node.isNodeType(LatexGroupNode):
            flat_tokens.append("{")
            flat_tokens.extend(parse_nodes(node.nodelist))
            flat_tokens.append("}")
    return flat_tokens


def get_final_tokens(formula):
    walker = LatexWalker(rf"{formula}")
    try:
        (nodes, pos, len_) = walker.get_latex_nodes()
    except:
        return []

    return ["<sos>"] + parse_nodes(nodes) + ["<eos>"]


def evaluate_official_metrics(
    model,
    dataloader,
    vocab,
    num_samples=None,
    k=3,
    output_file_beam="evaluation_results_beam.txt",
    output_file_greedy="evaluation_results_greedy.txt",
):
    model.eval()

    dataset_size = len(dataloader.dataset) if hasattr(dataloader, "dataset") else None
    if dataset_size is None:
        samples_to_evaluate = (
            num_samples if num_samples is not None else "all available"
        )
    elif num_samples is None:
        samples_to_evaluate = dataset_size
    else:
        samples_to_evaluate = min(num_samples, dataset_size)
    print(f"Evaluation samples: {samples_to_evaluate}")

    em_count = 0
    bleu_scores = []
    edit_distances = []
    evaluated_samples = 0

    smoothie = SmoothingFunction().method4

    with torch.no_grad():
        for i, (img, tgt) in enumerate(dataloader):
            if num_samples is not None and i >= num_samples:
                break

            img = img.to(device)

            pred_tokens = model.predict_beam_search(img, k=k)

            pred_str = clean_formula(pred_tokens, vocab)
            real_str = clean_formula(tgt[0], vocab)

            if pred_str.replace(" ", "") == real_str.replace(" ", ""):
                em_count += 1

            p_tokens = pred_str.split()
            r_tokens = real_str.split()
            b_score = sentence_bleu([r_tokens], p_tokens, smoothing_function=smoothie)
            bleu_scores.append(b_score)

            max_len = max(len(pred_str), len(real_str))
            if max_len > 0:
                ed = edit_distance(pred_str, real_str)
                norm_ed = 1 - (ed / max_len)
            else:
                norm_ed = 1.0
            edit_distances.append(norm_ed)
            evaluated_samples += 1

    em_pct = (em_count / evaluated_samples * 100) if evaluated_samples > 0 else 0.0
    bleu_mean = float(np.mean(bleu_scores)) if bleu_scores else 0.0
    edit_mean = (float(np.mean(edit_distances)) * 100) if edit_distances else 0.0

    report = (
        "\n"
        + "=" * 40
        + "\n"
        + "FINAL TEST SET REPORT Beam Search\n"
        + f"Exact Match (EM):     {em_pct:.2f}%\n"
        + f"BLEU Score:           {bleu_mean:.4f}\n"
        + f"Edit Distance (Text): {edit_mean:.2f}%\n"
        + "=" * 40
        + "\n"
    )

    with open(output_file_beam, "w") as f:
        f.write(report)

    em_count = 0
    bleu_scores = []
    edit_distances = []
    evaluated_samples = 0

    with torch.no_grad():
        for i, (img, tgt) in enumerate(dataloader):
            if num_samples is not None and i >= num_samples:
                break

            img = img.to(device)

            pred_tokens = model.predict(img)

            pred_str = clean_formula(pred_tokens, vocab)
            real_str = clean_formula(tgt[0], vocab)

            if pred_str.replace(" ", "") == real_str.replace(" ", ""):
                em_count += 1

            p_tokens = pred_str.split()
            r_tokens = real_str.split()
            b_score = sentence_bleu([r_tokens], p_tokens, smoothing_function=smoothie)
            bleu_scores.append(b_score)

            max_len = max(len(pred_str), len(real_str))
            if max_len > 0:
                ed = edit_distance(pred_str, real_str)
                norm_ed = 1 - (ed / max_len)
            else:
                norm_ed = 1.0
            edit_distances.append(norm_ed)
            evaluated_samples += 1

    em_pct = (em_count / evaluated_samples * 100) if evaluated_samples > 0 else 0.0
    bleu_mean = float(np.mean(bleu_scores)) if bleu_scores else 0.0
    edit_mean = (float(np.mean(edit_distances)) * 100) if edit_distances else 0.0

    report = (
        "\n"
        + "=" * 40
        + "\n"
        + "FINAL TEST SET REPORT Greedy\n"
        + f"Exact Match (EM):     {em_pct:.2f}%\n"
        + f"BLEU Score:           {bleu_mean:.4f}\n"
        + f"Edit Distance (Text): {edit_mean:.2f}%\n"
        + "=" * 40
        + "\n"
    )

    with open(output_file_greedy, "w") as f:
        f.write(report)


def clean_formula(tensor, vocab):
    if isinstance(tensor, torch.Tensor):
        tokens = [vocab.itos[idx.item()] for idx in tensor]
    else:
        tokens = tensor

    tokens = [t for t in tokens if t not in ("<pad>", "<sos>", "<eos>")]

    formula = " ".join(tokens)

    return formula.strip()
