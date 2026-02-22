import math
import torch
import torch.nn as nn

from mini_whisper.tokenizer.tokenizer import BPE_Tokenizer

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int, vocab_size: int = 1000):
        super().__init__()
        self.positional_embeddings = nn.Embedding(max_len, d_model)
                
    def forward(self, x, device="cpu"):
        # x: (batch, seq_len)    
        batch_size, seq_len = x.size()
        positions = torch.arange(seq_len, device=self.positional_embeddings.weight.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)  # [16, 1500]
        return self.positional_embeddings(positions)  # [16, 1500, 128]
    
    
class ToyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int, n_layers: int, n_heads: int, device="cpu"):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = LearnedPositionalEncoding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.pos_enc(input_ids, device=input_ids.device)
        x = self.token_emb(input_ids) * math.sqrt(self.token_emb.embedding_dim)

        x = self.encoder(x)
        x = self.lm_head(x)

        last_token_logits = x[:, -1, :] 
        probs = torch.softmax(last_token_logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1)
        return x, probs, next_token

        
if __name__ == "__main__":
    from mini_whisper.tokenizer.tokenizer import BPE_Tokenizer

    Tokenizer = BPE_Tokenizer()
    Tokenizer.load_merges('mini_whisper/decoder/merges.txt')

    txt = Tokenizer.encode('A text corpus (plural: corpora) is a large, structured, and typically digital collection of written or spoken language samples used for linguistic research, language modeling, and AI training. These datasets allow researchers to analyze word frequencies, syntax, and usage patterns to understand language structure and evolution. ')
    print("Encoded text:", txt, len(txt))
    #split every 64 tokens
    chunks = [txt[i:i+64] for i in range(0, len(txt), 64)]
    chunks = chunks[:-1] if len(chunks[-1]) < 64 else chunks
    print(chunks)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    embedding_layer = LearnedPositionalEncoding(max_len=10000+256, d_model=128).to(device)
    transformer = ToyTransformer(vocab_size=10000+256, d_model=128, max_len=1000+256, n_layers=2, n_heads=4, device=device).to(device)
    x_indicies = torch.tensor(chunks).to(device)
    print("Input shape:", x_indicies.shape)  # Should be (batch_size, seq_len)
    output, probs, next_token = transformer(x_indicies)
    print("Output shape:", output.shape)  # Should be (batch_size, seq_len, vocab_size)
    print("Probs shape:", probs.shape)
    print("Next token:", next_token)



    decoded_tokens = Tokenizer.decode(next_token.cpu().numpy())
    print("Decoded next token:", decoded_tokens)

    
        