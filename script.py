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

class Vocabulary:
    def __init__(self):
        # Token speciali con ID fissi
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.threshold = 1  # Frequenza minima per includere un token

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4  # Start dopo i token speciali

        for sentence in sentence_list:
            for word in sentence:
                if word not in ["<pad>", "<sos>", "<eos>", "<unk>"]:
                    frequencies[word] += 1

        for word, freq in frequencies.items():
            if freq >= self.threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        # Converte token in indici, usa <unk> se non trovato
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in text]

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


# 1. Istanzia e costruisci il vocabolario sui tuoi token
# tokenized = [get_final_tokens(formula) for formula in train_df.formula.values]
# vocab = Vocabulary()
# vocab.build_vocabulary(tokenized)

# # 2. Esempio di conversione della tua formula (record 1)
# example_indices = vocab.numericalize(tokenized[0])

# print(f"formula originale: {test_df.formula.values[0]}")
# print(f"Token originali: {tokenized[0]}")
# print(f"Indici numerici: {example_indices}")
# print(f"Dimensione del vocabolario: {len(vocab.stoi)}")

class Im2LatexDataset(Dataset):
    def __init__(self, df, vocab, tokenized_formulas, transform = None):
        self.df = df
        self.vocab = vocab
        self.transform = transform
        self.formulas= tokenized_formulas
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_bytes = self.df.iloc[index]['image']['bytes']
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        tokens = self.formulas[index]
        tokens_idx = self.vocab.numericalize(tokens)
        tokens_idx = torch.tensor(tokens_idx, dtype=torch.long)
        
        return image, tokens_idx
    

# Trasformazioni: Resize fisso (importante!) e normalizzazione
transform = T.Compose([
    T.Resize((64, 320)), # Dimensioni immagini del dataset (Height, Width)
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,)) 
])

# dataset = Im2LatexDataset(test_df, vocab, transform=transform)

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

class CnnEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        # Una CNN semplice (stile ResNet ridotta)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, d_model, 3, 1, 1), nn.BatchNorm2d(d_model), nn.ReLU(), nn.MaxPool2d((2, 2))
        )

    
    def forward(self, x):
        # Output: [Batch, d_model, 4, 20]
        features = self.conv(x)
        # Lo "appiattiamo" per il Transformer: [Batch, 80, d_model]
        # 80 è la lunghezza della sequenza visiva (4*20)
        features = features.flatten(2).permute(0, 2, 1)
        return features

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1,1000, d_model)) # Positional enoding

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

    
    def forward(self, tgt, memory, tgt_mask):
        # tgt: token Latex [Batch, Seq_Len]
        # memory: feature della CNN [Batch, 80, d_model]

        tgt_emb = self.embedding(tgt) + self.pos_encoding[:, :tgt.size(1), :]

        output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)

        return self.fc_out(output)

class Im2LatexModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        
        # 1. ENCODER (CNN)
        # Prende l'immagine [B, 3, 64, 320] -> restituisce feature [B, 80, d_model]
        self.encoder_cnn = CnnEncoder(d_model) 
        
        # 2. DECODER (Transformer)
        # Prende i token LaTeX e le feature della CNN -> predice il prossimo token
        self.decoder_transformer = TransformerDecoder(vocab_size, d_model, nhead, num_layers)

    def forward(self, src_img, tgt_tokens):
        # A. Estrazione feature visive (Memory)
        # src_img shape: [Batch, 3, 64, 320]
        memory = self.encoder_cnn(src_img) # Output: [Batch, 80, d_model]
        
        # B. Creazione maschera per il Decoder
        # Impedisce di guardare i token futuri nella sequenza LaTeX
        device = tgt_tokens.device
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tokens.size(1)).to(device)
        
        # C. Generazione output tramite il Decoder
        # tgt_tokens: [Batch, Seq_Len]
        output = self.decoder_transformer(tgt_tokens, memory, tgt_mask)
        
        return output # [Batch, Seq_Len, Vocab_Size]

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


example_img, example_tgt = next(iter(val_loader))
example_img = example_img[0].unsqueeze(0).to(device)

pred_tokens = predict(model, example_img, vocab)
prediction = clean_formula(pred_tokens, vocab)  # usa la funzione clean_formula

real_formula = clean_formula(example_tgt[0], vocab)

print(f"\nEsempio finale:\nReal: {real_formula}\nPred: {prediction}\n")

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
