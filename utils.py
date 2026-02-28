import numpy as np
import torch
from pylatexenc.latexwalker import LatexWalker, LatexCharsNode, LatexMacroNode, LatexGroupNode
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.metrics import edit_distance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    imgs, captions = zip(*batch)
    imgs = torch.stack(imgs)

    lengths = [len(cap) for cap in captions]
    max_len = max(lengths)

    padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded[i, :len(cap)] = cap

    return imgs, padded

def parse_nodes(nodes):
        flat_tokens = []
        for node in nodes:
            if node is None: continue

            if node.isNodeType(LatexCharsNode):
                # Caratteri semplici (a, b, =, +, ecc.)
                # Li dividiamo ulteriormente per carattere singolo se necessario
                for c in node.chars:
                    if not c.isspace(): flat_tokens.append(c)
                    
            elif node.isNodeType(LatexMacroNode):
                # Comandi tipo \frac, \alpha, \cal
                m_name = node.macroname
                token = "\\" + m_name if m_name else "\\\\"
                if token.strip() == "\\": continue
                flat_tokens.append(token)
                
            elif node.isNodeType(LatexGroupNode):
                # Gruppi tra { ... }, li apriamo ricorsivamente
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

def predict_beam_search(model, img, vocab, k=3, max_len=150):
    model.eval()
    device = img.device
    vocab_size = len(vocab.stoi)
    
    with torch.no_grad():
        # 1. Encode immagine
        memory = model.encoder_cnn(img) 

        # 2. Inizializza il fascio: (punteggio_log, sequenza_indici)
        # Partiamo con <sos> e punteggio 0
        beams = [(0.0, [vocab.stoi["<sos>"]])]

        for _ in range(max_len):
            new_beams = []
            for score, seq in beams:
                # Se la sequenza è già finita, la manteniamo così com'è
                if seq[-1] == vocab.stoi["<eos>"]:
                    new_beams.append((score, seq))
                    continue
                
                # Decoder forward per l'ultimo stato della sequenza
                ys = torch.tensor([seq], device=device)
                sz = ys.size(1)
                tgt_mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
                
                out = model.decoder_transformer(ys, memory, tgt_mask=tgt_mask)
                
                # Prendi le log-probabilità dell'ultimo token
                log_probs = torch.log_softmax(out[:, -1, :], dim=-1).squeeze(0)
                
                # Prendi i k migliori candidati per questo ramo
                topk_probs, topk_idx = log_probs.topk(k)
                
                for i in range(k):
                    new_beams.append((score + topk_probs[i].item(), seq + [topk_idx[i].item()]))
            
            # Seleziona i k migliori in assoluto tra tutti i rami espansi
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:k]
            
            # Se tutti i rami hanno incontrato <eos>, abbiamo finito
            if all(s[-1] == vocab.stoi["<eos>"] for _, s in beams):
                break
        
        # Restituisci la sequenza migliore (la prima della lista ordinata)
        best_seq = beams[0][1]
        return [vocab.itos[idx] for idx in best_seq]
    

def evaluate_official_metrics(model, dataloader, vocab, num_samples=100, k=3):
    model.eval()
    
    em_count = 0
    bleu_scores = []
    edit_distances = []
    
    smoothie = SmoothingFunction().method4
    print(f"Valutazione ufficiale su {num_samples} campioni...")

    with torch.no_grad():
        for i, (img, tgt) in enumerate(dataloader):
            if i >= num_samples: break
            
            img = img.to(device)
            
            # Generazione con Beam Search (più vicina ai paper)
            pred_tokens = predict_beam_search(model, img, vocab, k=k)
            
            # Pulizia stringhe
            pred_str = clean_formula(pred_tokens, vocab)
            real_str = clean_formula(tgt[0], vocab) # tgt è un batch di 1
            
            # 1. Exact Match (EM)
            if pred_str.replace(" ", "") == real_str.replace(" ", ""):
                em_count += 1
            
            # 2. BLEU Score
            p_tokens = pred_str.split()
            r_tokens = real_str.split()
            b_score = sentence_bleu([r_tokens], p_tokens, smoothing_function=smoothie)
            bleu_scores.append(b_score)
            
            # 3. Edit Distance (Text) - Normalizzata
            max_len = max(len(pred_str), len(real_str))
            if max_len > 0:
                ed = edit_distance(pred_str, real_str)
                # Molti paper riportano (1 - d/max_len)
                norm_ed = 1 - (ed / max_len)
            else:
                norm_ed = 1.0
            edit_distances.append(norm_ed)

    print("\n" + "="*40)
    print(f"REPORT FINALE SUL TEST SET")
    print(f"Exact Match (EM):     {em_count/num_samples*100:.2f}%")
    print(f"BLEU Score:           {np.mean(bleu_scores):.4f}")
    print(f"Edit Distance (Text): {np.mean(edit_distances)*100:.2f}%")
    print("="*40)

def clean_formula(tensor, vocab):
    """
    Decodifica un tensor di token in stringa LaTeX.
    """
    if isinstance(tensor, torch.Tensor):
        tokens = [vocab.itos[idx.item()] for idx in tensor]
    else:
        tokens = tensor
    
    # Rimuovi token speciali
    tokens = [t for t in tokens if t not in ("<pad>", "<sos>", "<eos>")]
    
    # Unisci in stringa
    formula = " ".join(tokens)
    
    return formula.strip()