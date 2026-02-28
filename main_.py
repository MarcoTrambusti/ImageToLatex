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
# Standard Pytorch imports (note the aliases).
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


transform = T.Compose([
    T.Resize((64, 320)), # Dimensioni immagini del dataset (Height, Width)
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,)) 
])

def collate_fn(batch):
    imgs, captions = zip(*batch)
    imgs = torch.stack(imgs)

    lengths = [len(cap) for cap in captions]
    max_len = max(lengths)

    padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded[i, :len(cap)] = cap

    return imgs, padded


# dataLoader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# =====================
# MINI DATASET DEBUG
# =====================
# train_df = train_df.sample(1000, random_state=2000)
# val_df   = val_df.sample(100, random_state=2000)
print(f"TRAIN SIZE: {train_df.size}")
print(f"TEST SIZE: {val_df.size}")

# 1. GENERAZIONE TOKEN (Senza Dataset per ora)
print("Tokenizing training formulas...")
tokenized_train = [get_final_tokens(f) for f in train_df.formula.values]
tokenized_val = [get_final_tokens(f) for f in val_df.formula.values]
lengths = [len(f) for f in tokenized_train]
print(f"Lunghezza massima trovata nel Train: {max(lengths)}")

# 2. COSTRUZIONE VOCABOLARIO (Questo definisce la size finale)
vocab = Vocabulary()
vocab.build_vocabulary(tokenized_train)
new_vocab_size = len(vocab.stoi)
print(f"Nuova dimensione vocabolario: {new_vocab_size}") 

# 3. CREAZIONE DATASET E LOADER (Ora usano il vocab aggiornato)
train_dataset = Im2LatexDataset(train_df, vocab, tokenized_train, transform=transform)
val_dataset   = Im2LatexDataset(val_df, vocab, tokenized_val, transform=transform)

# Importante: batch_size piccolo se sei su CPU o GPU limitata
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 4. SANITY CHECK (Ora deve passare per forza)
max_found = 0
for i, (imgs, caps) in enumerate(train_loader):
    if i > 100: break # Controlla i primi 100 batch
    if caps.max() >= new_vocab_size:
        print(f"!!! ERRORE AL BATCH {i} !!! Max idx: {caps.max()}")
        break
    max_found = max(max_found, caps.max().item())
print(f"Controllo finito. Max index: {max_found} / Vocab Size: {new_vocab_size}")


# =====================
# MODEL / DEVICE
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# 5. INIZIALIZZAZIONE MODELLO
model = Im2LatexModel(vocab_size=new_vocab_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=0)


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

# =====================
# TRAIN & EVAL FUNCTIONS
# =====================
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    
    for imgs, captions in dataloader:
        imgs, captions = imgs.to(device), captions.to(device)
        
        # Prepariamo input e target: 
        # tgt_input: da <sos> a penultimo token
        # tgt_expected: da secondo token a <eos>
        tgt_input = captions[:, :-1]
        tgt_expected = captions[:, 1:]
        
        # 2. Controllo di sicurezza sulle dimensioni
        if tgt_input.size(1) == 0: continue

        # Forward
        preds = model(imgs, tgt_input) # [B, Seq, Vocab]
        
        # Calcolo Loss (Flatten per CrossEntropy)
        loss = criterion(preds.reshape(-1, preds.shape[-1]), tgt_expected.reshape(-1))
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()

        # Clip degli gradienti (evita esplosioni che possono causare errori CUDA)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()

        if i % 100 == 0:
           print(f"Batch {i}/{len(dataloader)} - Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
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
    Genera una sequenza di token LaTeX dall'immagine.

    img: [1, 3, H, W] già su device
    vocab: oggetto Vocabulary
    max_len: lunghezza massima generata
    return_tokens: se True restituisce lista di token, altrimenti stringa joinata
    """
    model.eval()
    device = img.device
    vocab_size = len(vocab.stoi)

    with torch.no_grad():
        # 1️⃣ Encode dell'immagine
        memory = model.encoder_cnn(img)  # [B, Seq, d_model]

        # 2️⃣ Inizializza sequenza con <sos>
        ys = torch.full((1, 1), vocab.stoi["<sos>"], dtype=torch.long, device=device)

        for _ in range(max_len):
            # Maschera triangolare
            tgt_mask = torch.triu(
                torch.ones(ys.size(1), ys.size(1), device=device) * float('-inf'),
                diagonal=1
            )

            # 3️⃣ Decoder autoregressivo
            out = model.decoder_transformer(ys, memory, tgt_mask=tgt_mask)  # [1, seq_len, vocab_size]

            # 4️⃣ Greedy: prendi il token più probabile
            next_word = out[:, -1, :].argmax(dim=-1).item()
            next_word = max(0, min(next_word, vocab_size - 1))

            # 5️⃣ Aggiungi alla sequenza
            ys = torch.cat([ys, torch.tensor([[next_word]], device=device)], dim=1)

            # 6️⃣ Stop se <eos>
            if next_word == vocab.stoi["<eos>"]:
                break

        # 7️⃣ Converti in token (lista di stringhe)
        tokens = [vocab.itos[idx.item()] for idx in ys[0]]

        return tokens


# ===================== LOOP DI TRAIN =====================
num_epochs = 12

log_file = "training_log.log"

with open(log_file, "w") as f:
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)

        log_line = f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        print(log_line)
        f.write(log_line + "\n")

    # ===================== ESEMPIO FINALE =====================
    example_img, example_tgt = next(iter(val_loader))
    example_img = example_img[0].unsqueeze(0).to(device)

    pred_tokens = predict(model, example_img, vocab)
    prediction = clean_formula(pred_tokens, vocab)  # usa la funzione clean_formula
    real_formula = clean_formula(example_tgt[0], vocab)
    
    final_example = f"\nEsempio finale:\nReal: {real_formula}\nPred: {prediction}\n"
    print(final_example)
    f.write(final_example)


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
greedy_tokens = predict(model, example_img, vocab) # La tua vecchia funzione
beam_tokens = predict_beam_search(model, example_img, vocab, k=3)

print(f"REAL:   {clean_formula(example_tgt[0], vocab)}")
print(f"GREEDY: {clean_formula(greedy_tokens, vocab)}")
print(f"BEAM:   {clean_formula(beam_tokens, vocab)}")


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.metrics import edit_distance

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

# Esegui il test finale
evaluate_official_metrics(model, test_loader, vocab, num_samples=100)
