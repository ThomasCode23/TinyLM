import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_length=512):
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        self.data = []
        for text in texts:
            tokens = self.tokenizer.encode_as_ids(text.strip()) + [self.tokenizer.piece_to_id('<eos>')]
            self.data.extend(tokens)

    def __len__(self):
        return max(0, len(self.data) - self.seq_length)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_length], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_length+1], dtype=torch.long)
        return x, y
