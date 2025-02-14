"""
tiny_llm.py

This script defines TinyLLM, a small decoder-only transformer model designed for language modeling tasks.  
It follows a GPT-style architecture with stacked transformer blocks and causal self-attention.  

Key Components:
- **Token Embeddings:** Converts input token IDs into dense vector representations.  
- **Positional Encoding:** Injects position information into token embeddings using sine/cosine functions.  
- **Stacked Transformer Blocks:** Implements a series of custom transformer layers (`TransformerBlock`)  
  that include multi-head self-attention and a feedforward network.  
- **Final Linear Layer:** Maps hidden states to vocabulary logits for next-token prediction.  

Key Features:
- **Decoder-only architecture** (no encoder, only self-attention layers).  
- **Causal self-attention** (uses an upper-triangular mask to prevent future token access).  
- **Lightweight and efficient** for small-scale language modeling tasks.  

This model is used in `train.py`, where it gets trained on text data.
"""


import torch
import torch.nn as nn
from models.transformer import TransformerBlock, PositionalEncoding

class TinyLLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        seq_len = x.size(1)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        x = self.ln_final(x)
        logits = self.output(x)
        return logits
