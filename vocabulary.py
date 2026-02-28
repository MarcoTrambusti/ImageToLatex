from collections import Counter

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