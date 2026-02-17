from decoder.tokenizer import BPE_Tokenizer
from decoder.positionalEncoder import LearnedPositionalEncoding
from .MHA import MultiHeadAttention
from torch import Tensor
from torch import nn
import torch

class TextDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int, n_layers: int, n_heads: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = LearnedPositionalEncoding(max_len, d_model)
        self.blocks = nn.ModuleList([MultiHeadAttention(d_model, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.mask = None

    def forward(self, x: Tensor, cross_x: Tensor):
        x = self.token_emb(x) + self.pos_enc(x)
        x = x.to(cross_x.dtype)
        for block in self.blocks:
            x = block(x, cross_x, cross_x, mask=self.mask)
        x = self.ln(x)
        logits = x @ torch.transpose(self.token_emb.weight.to(x.dtype), 0, 1)
        return logits