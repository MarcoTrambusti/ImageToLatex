import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class CnnEncoder(nn.Module):
    def __init__(self, d_model=256, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            nn.ReLU(inplace=True),
            resnet.maxpool
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.layer4 = resnet.layer4
        self.layer4[0].conv1.stride = (1, 1)
        if self.layer4[0].downsample is not None:
            self.layer4[0].downsample[0].stride = (1, 1)

        self.proj = nn.Conv2d(512, d_model, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.proj(x)

        x = x.flatten(2)
        x = x.permute(0, 2, 1)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))

        layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask):
        tgt_emb = self.embedding(tgt) + self.pos_encoding[:, :tgt.size(1), :]
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.fc_out(output)

class Im2LatexModel(nn.Module):

    def __init__(self, vocab, d_model=256, nhead=8, num_layers=4, lr=1e-4, device=None):
        super().__init__()

        self.vocab = vocab
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.encoder = CnnEncoder(d_model)
        self.decoder = TransformerDecoder(
            len(vocab.stoi), d_model, nhead, num_layers
        )

        self.to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, images, tgt_tokens):
        memory = self.encoder(images)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_tokens.size(1)
        ).to(self.device)

        return self.decoder(tgt_tokens, memory, tgt_mask)

    def train_model(self, train_loader=None, epochs=None, val_loader=None):
        """
        Se chiamato senza argomenti → comportamento standard PyTorch.
        Se chiamato con train_loader e epochs → esegue il training loop.
        """

        # Caso standard PyTorch
        if train_loader is None:
            return super().train()

        # Training loop
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):

            super().train()
            total_train_loss = 0

            for imgs, captions in train_loader:

                imgs = imgs.to(self.device)
                captions = captions.to(self.device)

                tgt_input = captions[:, :-1]
                tgt_expected = captions[:, 1:]

                if tgt_input.size(1) == 0:
                    continue

                preds = self.forward(imgs, tgt_input)

                loss = self.criterion(
                    preds.reshape(-1, preds.shape[-1]),
                    tgt_expected.reshape(-1)
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)

            # VALIDATION
            if val_loader is not None:
                super().eval()
                total_val_loss = 0

                with torch.no_grad():
                    for imgs, captions in val_loader:
                        imgs = imgs.to(self.device)
                        captions = captions.to(self.device)

                        tgt_input = captions[:, :-1]
                        tgt_expected = captions[:, 1:]

                        preds = self.forward(imgs, tgt_input)

                        loss = self.criterion(
                            preds.reshape(-1, preds.shape[-1]),
                            tgt_expected.reshape(-1)
                        )

                        total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(val_loader)
                history["val_loss"].append(avg_val_loss)

                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Train: {avg_train_loss:.4f} | "
                      f"Val: {avg_val_loss:.4f}")
            else:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Train: {avg_train_loss:.4f}")

        return history

    # =====================================================
    # PREDICT (GREEDY)
    # =====================================================
    def predict(self, img, max_len=150):

        super().eval()
        img = img.to(self.device)

        with torch.no_grad():

            memory = self.encoder(img)

            ys = torch.full(
                (1, 1),
                self.vocab.stoi["<sos>"],
                dtype=torch.long,
                device=self.device
            )

            for _ in range(max_len):

                tgt_mask = torch.triu(
                    torch.ones(ys.size(1), ys.size(1), device=self.device)
                    * float("-inf"),
                    diagonal=1
                )

                out = self.decoder(ys, memory, tgt_mask)

                next_token = out[:, -1, :].argmax(dim=-1).item()

                ys = torch.cat(
                    [ys, torch.tensor([[next_token]], device=self.device)],
                    dim=1
                )

                if next_token == self.vocab.stoi["<eos>"]:
                    break

        return [self.vocab.itos[idx.item()] for idx in ys[0]]

    def predict_beam_search(self, img, k=3, max_len=150):
        super().eval()
        device = img.device
        vocab_size = len(self.vocab.stoi)
        
        with torch.no_grad():
            # 1. Encode immagine
            memory = self.encoder(img) 

            # 2. Inizializza il fascio: (punteggio_log, sequenza_indici)
            # Partiamo con <sos> e punteggio 0
            beams = [(0.0, [self.vocab.stoi["<sos>"]])]

            for _ in range(max_len):
                new_beams = []
                for score, seq in beams:
                    # Se la sequenza è già finita, la manteniamo così com'è
                    if seq[-1] == self.vocab.stoi["<eos>"]:
                        new_beams.append((score, seq))
                        continue
                    
                    # Decoder forward per l'ultimo stato della sequenza
                    ys = torch.tensor([seq], device=device)
                    sz = ys.size(1)
                    tgt_mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
                    
                    out = self.decoder(ys, memory, tgt_mask=tgt_mask)
                    
                    # Prendi le log-probabilità dell'ultimo token
                    log_probs = torch.log_softmax(out[:, -1, :], dim=-1).squeeze(0)
                    
                    # Prendi i k migliori candidati per questo ramo
                    topk_probs, topk_idx = log_probs.topk(k)
                    
                    for i in range(k):
                        new_beams.append((score + topk_probs[i].item(), seq + [topk_idx[i].item()]))
                
                # Seleziona i k migliori in assoluto tra tutti i rami espansi
                beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:k]
                
                # Se tutti i rami hanno incontrato <eos>, abbiamo finito
                if all(s[-1] == self.vocab.stoi["<eos>"] for _, s in beams):
                    break
            
            # Restituisci la sequenza migliore (la prima della lista ordinata)
            best_seq = beams[0][1]
            return [self.vocab.itos[idx] for idx in best_seq]


    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))