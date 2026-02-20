import math
import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int, vocab_size: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Max len = iters + 256 (initial vocab size), d_model = 128/256 (toy example)
        self.positional_embeddings = nn.Embedding(max_len, d_model)
                
    def forward(self, x, device="cpu"):
        # x: (batch, seq_len, d_model)    
        batch_size, seq_len, d = x.size()
        sq = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.positional_embeddings(sq)
        return x + pos_emb
    
    
class ToyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int, n_layers: int, n_heads: int, device="cpu"):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = LearnedPositionalEncoding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.token_emb(input_ids) * math.sqrt(self.token_emb.embedding_dim)
        x = self.pos_enc(x, device=input_ids.device)
        x = self.encoder(x)
        x = self.lm_head(x)

        last_token_logits = x[:, -1, :] 
        probs = torch.softmax(last_token_logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1)
        return x, probs, next_token

        
if __name__ == "__main__":
    from mini_whisper.tokenizer import tokenizer

    Tokenizer = tokenizer.BPE_Tokenizer()
    Tokenizer.load_merges('decoder/merges.txt')

    txt = Tokenizer.encode('A text corpus (plural: corpora) is a large, structured, and typically digital collection of written or spoken language samples used for linguistic research, language modeling, and AI training. These datasets allow researchers to analyze word frequencies, syntax, and usage patterns to understand language structure and evolution. ')
    print("Encoded text:", txt, len(txt))
    #split every 64 tokens
    chunks = [txt[i:i+64] for i in range(0, len(txt), 64)]
    chunks = chunks[:-1] if len(chunks[-1]) < 64 else chunks
    print(chunks)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    embedding_layer = LearnedPositionalEncoding(max_len=1000+256, d_model=128).to(device)
    transformer = ToyTransformer(vocab_size=1000+256, d_model=128, max_len=1000+256, n_layers=2, n_heads=4, device=device).to(device)
    x_indicies = torch.tensor(chunks).to(device)
    output, probs, next_token = transformer(x_indicies)
    print("Output shape:", output.shape)  # Should be (batch_size, seq_len, vocab_size)
    print("Probs shape:", probs.shape)
    print("Next token:", next_token)



    decoded_tokens = Tokenizer.decode(next_token.cpu().numpy())
    print("Decoded next token:", decoded_tokens)

    
        