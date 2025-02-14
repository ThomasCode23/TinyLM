import torch
import torch.nn as nn
from models.transformer import TransformerBlock
from models.positional_encoding import PositionalEncoding

class TinyLLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=8, dim_feedforward=2048, dropout=0.1):
        super(TinyLLM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        seq_len = x.size(1)

        # Generate causal mask
        # attn_mask shape: (seq_len, seq_len)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        x = self.ln_final(x)
        logits = self.output(x)
        return logits
