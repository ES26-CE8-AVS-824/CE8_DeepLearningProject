import torch.nn.functional as F
from torch import nn, Tensor


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module that supports both self-attention and cross-attention.
    ------
    Input:
    - x: (batch_size, seq_len, n_state) - the input sequence for self-attention
    - cross_x: (batch_size, cross_seq_len, n_state) - the input sequence for cross-attention (optional)
    - mask: (batch_size, seq_len, seq_len) - attention mask (optional)
     Output:
    - output: (batch_size, seq_len, n_state) - the result of the attention mechanism
    """
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(d_model, d_model) # In whisper, they have created their own Linear class representation, not sure if needed
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, x: Tensor, cross_x: Tensor=None, mask: Tensor=None):
        q = self.query(x)
        # if cross attention, use cross_x for k and v, otherwise use x
        k = self.key(cross_x if cross_x is not None else x)
        v = self.value(cross_x if cross_x is not None else x)
        
        wv, qk = self.qkv_attention(q, k, v, mask)
        
        return self.out(wv), qk
    
    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor=None):
        n_batch, seq_len, d_model = q.shape
        scale = (d_model // self.n_head) ** -0.25 # Scaling factor
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) # (batch_size, n_head, seq_len, n_batch)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        
        qk = (q * scale) @ (k * scale).transpose(-1, -2) # scaled dot product of q and k
        if mask is not None:
            qk = qk + mask[:seq_len, :seq_len] # Applies attention mask
        qk = qk.float() # Cast to f32 for numerical stability during softmax

        w = F.softmax(qk, dim=-1).to(q.dtype) # Convert back to original dtype after softmax
        out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2) # Compute weighted sum of values and reshape back to (batch_size, seq_len, d_model)
        qk = qk.detach() # Detach qk to prevent gradients from flowing back through the attention scores
        
        return out, qk
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, cross_attention: bool=False, dropout: float=0.0):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_head)
        self.attn_ln = nn.LayerNorm(d_model)
        
        self.cross_attention = MultiHeadAttention(d_model, n_head) if cross_attention else None
        self.cross_attention_ln = nn.LayerNorm(d_model) if cross_attention else None
        
        n_mlp = d_model * 4  # In whisper at least, the feedforward network has an inner dimension 4 times the model dimension
        self.ff = nn.Sequential(
            nn.Linear(d_model, n_mlp),
            nn.GELU(),
            nn.Linear(n_mlp, d_model)
        )
        self.ff_ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor, cross_x: Tensor=None, mask: Tensor=None):
        x = x  + self.attn(self.attn_ln(x), mask=mask)[0] # Self-attention with residual connection
        if self.cross_attention is not None and cross_x is not None:
            x = x + self.cross_attention(self.cross_attention_ln(x), cross_x=cross_x, mask=mask)[0] # Cross-attention with residual connection
        x = x + self.ff(self.ff_ln(x)) # Feedforward network with residual connection
        return x