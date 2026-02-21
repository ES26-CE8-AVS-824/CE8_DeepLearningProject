from mini_whisper.decoder.positionalEncoder import LearnedPositionalEncoding
from mini_whisper.transformer.MHA import MultiHeadAttention
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
        print(f'pos emb shape: {self.pos_enc(x)[0:0+x.shape[-1]].shape}, token shape: {self.token_emb(x).shape}')  # Debugging statement to check shapes
        x = self.token_emb(x) + self.pos_enc(x)[0:0+x.shape[-1]] 
        x = x.to(cross_x.dtype)
        for block in self.blocks:
            x = block(x, cross_x=cross_x, mask=self.mask)[0]  # Use cross-attention, take only the output (not qk)
        x = self.ln(x)
        logits = x @ torch.transpose(self.token_emb.weight.to(x.dtype), 0, 1)
        return logits
    
if __name__ == "__main__":
    # And now to decode the results
    from mini_whisper.tokenizer.tokenizer import BPE_Tokenizer
    from mini_whisper.decoder.textDecoder import TextDecoder
    from torch.nn import functional as F
    
    BATCH_SIZE = 16
    D_MODEL = 128
    random_seed = 42
    torch.manual_seed(random_seed)

    z = torch.randn(BATCH_SIZE, 1500, D_MODEL)  # Simulated encoder output with batch size 16 and sequence length 50


    tokenizer = BPE_Tokenizer()
    tokenizer.load_merges('mini_whisper/decoder/merges.txt')
    seqs = [tokenizer.encode("hello world"),
            tokenizer.encode("this is a test")]          

    txt = torch.tensor(seqs, dtype=torch.long)          
    B, T = txt.shape                                    
    max_len = 1500

    # 1) Pad last dim from 3 -> 20 with zeros
    pad_len = max_len - T
    txt_padded = F.pad(txt, (0, pad_len), value=0)      # [2, 20]

    # 2) Expand batch dim from 2 -> 16 by repeating
    repeats = (BATCH_SIZE + B - 1) // B               # ceil(16 / 2) = 8
    txt_big = txt_padded.repeat(repeats, 1)[:BATCH_SIZE]  # [16, 20]

    decoder = TextDecoder(vocab_size=10000, d_model=D_MODEL, max_len=1500, n_layers=3, n_heads=4)
    with torch.no_grad():
        print(f'x: {txt_big.shape}, z: {z.shape}')  # [16, seq_len, d_model]
        logits = decoder(txt_big, z)
        print(f'logits shape: {logits.shape}')  # Should be [16, seq_len, vocab_size]
        print(F.softmax(logits, dim=-1).shape)  # Should also be [16, seq_len, vocab_size]
        print(F.softmax(logits, dim=-1)[0, 0, :10])  # Print probabilities of first 10 tokens for the first position in the first batch item
        greedy_token = torch.argmax(F.softmax(logits, dim=-1), dim=-1)  # [16, seq_len]
        print(f'greedy token shape: {greedy_token.shape}')  # Should be [16, seq_len]
        print(f'greedy token for first position in first batch item: {greedy_token[0, 0]}')
        list(map(lambda i: print(f'greedy token for position {i} in first batch item: {tokenizer.decode([greedy_token[0, i].item()])}'), range(20)))