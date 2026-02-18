import torch
import torch.nn.functional as F
from torch import nn, Tensor


class SimpleMultiHeadAttention(nn.Module):
    """
    Simplified Multi-Head Attention with clear mathematical formulas.
    Supports both self-attention and cross-attention.

    Self-Attention (cross_x=None):
        Q = xW_q, K = xW_k, V = xW_v

    Cross-Attention (cross_x provided):
        Q = xW_q, K = cross_xW_k, V = cross_xW_v

    Math:
        Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
        MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W_o
    """

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head  # Dimension per head

        # Linear projections: Q = xW_q, K = xW_k, V = xW_v
        # d_model for both input and output dimensions of Q, K and V but in theory we could have
        # nn.Linear(d_model, d_state) for (Q,K,V) and nn.Linear(d_state, d_model) for (O)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, cross_x: Tensor = None, mask: Tensor = None):
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            cross_x: Optional cross-attention source (batch, source_len, d_model)
                     If None, performs self-attention
                     If provided, performs cross-attention (Q from x, K,V from cross_x)
            mask: Optional attention mask (batch, n_head, seq_len, seq_len or source_len)
        """
        # x.shape = (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape

        # Linear projections:
        # Self-attention: Q = xW_q, K = xW_k, V = xW_v
        # Cross-attention: Q = xW_q, K = cross_xW_k, V = cross_xW_v
        Q = self.query(x)  # Query always comes from x: (batch, seq_len, d_model)

        # Keys and values come from cross_x if provided, otherwise from x
        source = cross_x if cross_x is not None else x
        K = self.key(source)    # (batch, source_len, d_model)
        V = self.value(source)  # (batch, source_len, d_model)

        # Split into multiple heads: (batch, seq_len, d_model) -> (batch, n_head, seq_len, d_head)
        source_len = K.shape[1]  # Source length (could be different from seq_len in cross-attention)
        Q = Q.reshape(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        K = K.reshape(batch_size, source_len, self.n_head, self.d_head).transpose(1, 2)
        V = V.reshape(batch_size, source_len, self.n_head, self.d_head).transpose(1, 2)

        # Scaled dot-product attention: QK^T / sqrt(d_k)
        # K^T shape: (batch, n_head, d_head, source_len)
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, n_head, seq_len, source_len)
        scores = scores / (self.d_head ** 0.5)  # Scale by sqrt(d_k)

        # Apply mask if provided (for padding or causal masking)
        if mask is not None:
            scores = scores + mask  # Add mask (typically large negative values)

        # Attention weights: softmax(QK^T / sqrt(d_k))
        attn_weights = F.softmax(scores, dim=-1)  # (batch, n_head, seq_len, source_len)

        # Apply attention to values: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
        # V shape: (batch, n_head, source_len, d_head)
        attn_output = torch.matmul(attn_weights, V)  # (batch, n_head, seq_len, d_head)

        # Concatenate heads: (batch, n_head, seq_len, d_head) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

        # Final linear projection: MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W_o
        output = self.out(attn_output)  # (batch, seq_len, d_model)

        return output


class SimpleTransformerBlock(nn.Module):
    """
    Simplified Transformer Block with Pre-LayerNorm architecture.
    Supports optional cross-attention for encoder-decoder architectures.

    Architecture:
    1. x = x + SelfAttention(LayerNorm(x))
    2. x = x + CrossAttention(LayerNorm(x), encoder_output)  [optional]
    3. x = x + FeedForward(LayerNorm(x))
    """

    def __init__(self, d_model: int, n_head: int, d_ff: int = None, dropout: float = 0.0,
                 cross_attention: bool = False):
        super().__init__()
        # Default feedforward dimension is 4x the model dimension
        if d_ff is None:
            d_ff = d_model * 4

        # Self-attention sublayer
        self.attention = SimpleMultiHeadAttention(d_model, n_head)
        self.attn_norm = nn.LayerNorm(d_model)

        # Optional cross-attention sublayer (for decoder blocks)
        self.cross_attention = None
        self.cross_attn_norm = None
        if cross_attention:
            self.cross_attention = SimpleMultiHeadAttention(d_model, n_head)
            self.cross_attn_norm = nn.LayerNorm(d_model)

        # Fully connected sublayer: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),  # First projection: expand to d_ff
            nn.GELU(),  # Non-linear activation
            nn.Linear(d_ff, d_model)  # Second projection: back to d_model
        )
        self.ff_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, cross_x: Tensor = None, mask: Tensor = None,
                cross_mask: Tensor = None):
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            cross_x: Optional encoder output for cross-attention (batch, encoder_len, d_model)
            mask: Optional mask for self-attention
            cross_mask: Optional mask for cross-attention
        """
        # Self-attention with residual: x = x + Attention(LayerNorm(x))
        attn_out = self.attention(self.attn_norm(x), mask=mask)
        x = x + attn_out  # Residual connection

        # Cross-attention with residual (if enabled): x = x + CrossAttention(LayerNorm(x), cross_x)
        if self.cross_attention is not None and cross_x is not None:
            cross_attn_out = self.cross_attention(self.cross_attn_norm(x), cross_x=cross_x,
                                                   mask=cross_mask)
            x = x + cross_attn_out  # Residual connection

        # Feedforward with residual: x = x + FFN(LayerNorm(x))
        ff_out = self.feedforward(self.ff_norm(x))
        x = x + ff_out  # Residual connection

        return x


# Example usage
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 2
    seq_len = 10
    encoder_len = 15
    d_model = 512
    n_head = 8

    print("=" * 60)
    print("Testing Simple Multi-Head Attention")
    print("=" * 60)

    # Test 1: Self-Attention
    print("\n1. Self-Attention (Encoder)")
    x = torch.randn(batch_size, seq_len, d_model)
    encoder_block = SimpleTransformerBlock(d_model, n_head, cross_attention=False)
    encoder_output = encoder_block(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {encoder_output.shape}")

    # Test 2: Cross-Attention
    print("\n2. Cross-Attention (Decoder)")
    decoder_input = torch.randn(batch_size, seq_len, d_model)
    encoder_output = torch.randn(batch_size, encoder_len, d_model)
    decoder_block = SimpleTransformerBlock(d_model, n_head, cross_attention=True)
    decoder_output = decoder_block(decoder_input, cross_x=encoder_output)
    print(f"   Decoder input shape: {decoder_input.shape}")
    print(f"   Encoder output shape: {encoder_output.shape}")
    print(f"   Decoder output shape: {decoder_output.shape}")

    # Test 3: Just Multi-Head Attention
    print("\n3. Standalone Multi-Head Attention")
    mha = SimpleMultiHeadAttention(d_model, n_head)

    # Self-attention
    self_attn_out = mha(x)
    print(f"   Self-attention output: {self_attn_out.shape}")

    # Cross-attention
    cross_attn_out = mha(decoder_input, cross_x=encoder_output)
    print(f"   Cross-attention output: {cross_attn_out.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
