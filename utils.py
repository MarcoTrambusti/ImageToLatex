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


def evaluate_official_metrics(model, dataloader, vocab, num_samples=100, k=3, output_file="evaluation_results.txt"):
    model.eval()

    em_count = 0
    bleu_scores = []
    edit_distances = []

    smoothie = SmoothingFunction().method4

    with torch.no_grad():
        for i, (img, tgt) in enumerate(dataloader):
            if i >= num_samples:
                break

            img = img.to(device)

            pred_tokens = model.predict_beam_search(img, k=k)

            pred_str = clean_formula(pred_tokens, vocab)
            real_str = clean_formula(tgt[0], vocab)

            pred_str = canonicalize_latex(pred_str)
            real_str = canonicalize_latex(real_str)

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

    report = (
        "\n" + "=" * 40 + "\n" +
        "FINAL TEST SET REPORT\n" +
        f"Exact Match (EM):     {em_count/num_samples*100:.2f}%\n" +
        f"BLEU Score:           {np.mean(bleu_scores):.4f}\n" +
        f"Edit Distance (Text): {np.mean(edit_distances)*100:.2f}%\n" +
        "=" * 40 + "\n"
    )

    with open(output_file, "w") as f:
        f.write(report)

def clean_formula(tensor, vocab):
    if isinstance(tensor, torch.Tensor):
        tokens = [vocab.itos[idx.item()] for idx in tensor]
    else:
        tokens = tensor

    tokens = [t for t in tokens if t not in ("<pad>", "<sos>", "<eos>")]

    formula = " ".join(tokens)

    return formula.strip()


def canonicalize_latex(s: str) -> str:
    if not isinstance(s, str):
        return ""

    # =====================================================
    # 1. Normalizzazioni macro equivalenti
    # =====================================================
    replacements = {
        r"\\ldots": "...",
        r"\\cdots": "...",
        r"\\dots": "...",
        r"\\cdot": "*",
        r"\\times": "*",
        r"\\,": "",
        r"\\;": "",
        r"\\!": "",
        r"\\:": "",
        r"\\quad": "",
        r"\\qquad": "",
        r"\\displaystyle": "",
        r"\\textstyle": "",
        r"\\left": "",
        r"\\right": "",
    }

    for k, v in replacements.items():
        s = re.sub(k, v, s)

    # =====================================================
    # 2. Normalizza \frac12 → \frac{1}{2}
    # =====================================================
    s = re.sub(r"\\frac\s*([^\{\s])\s*([^\{\s])", r"\\frac{\1}{\2}", s)

    # =====================================================
    # 3. Rimuovi graffe inutili: {a} → a
    # (solo se contengono un singolo token semplice)
    # =====================================================
    s = re.sub(r"\{([a-zA-Z0-9])\}", r"\1", s)

    # =====================================================
    # 4. Normalizza spazi
    # =====================================================
    s = re.sub(r"\s+", " ", s)

    # Rimuove spazi attorno a simboli comuni
    s = re.sub(r"\s*([\+\-\=\*/\^\_\(\)\[\]\{\}])\s*", r"\1", s)

    # =====================================================
    # 5. Rimuove doppi apici di graffe inutili
    # {{x}} → {x}
    # =====================================================
    while re.search(r"\{\{([^{}]+)\}\}", s):
        s = re.sub(r"\{\{([^{}]+)\}\}", r"{\1}", s)

    # =====================================================
    # 6. Normalizza punti di sospensione
    # =====================================================
    s = s.replace(". . .", "...")
    s = re.sub(r"\.\s*\.\s*\.", "...", s)

    # =====================================================
    # 7. Normalizza operatori multipli
    # =====================================================
    s = s.replace("**", "*")
    s = s.replace("--", "-")

    # =====================================================
    # 8. Rimuovi spazi residui
    # =====================================================
    s = s.strip()

    return s
