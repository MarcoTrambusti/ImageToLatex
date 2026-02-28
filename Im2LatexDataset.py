class Im2LatexDataset(Dataset):
    def __init__(self, df, vocab, tokenized_formulas, transform = None):
        self.df = df
        self.vocab = vocab
        self.transform = transform
        self.formulas= tokenized_formulas
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_bytes = self.df.iloc[index]['image.bytes']
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        tokens = self.formulas[index]
        tokens_idx = self.vocab.numericalize(tokens)
        tokens_idx = torch.tensor(tokens_idx, dtype=torch.long)
        
        return image, tokens_idx
