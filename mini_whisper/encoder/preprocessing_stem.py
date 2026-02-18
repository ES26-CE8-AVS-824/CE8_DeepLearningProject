import torch
from torch import nn

from mini_whisper.audio import compute_log_mel_spectrogram, normalize


def sinusoidal_positional_embedding(
        token_sequence_length: int,
        token_embedding_dim: int,
) -> torch.Tensor:
    """
    Generate sinusoidal positional embeddings/encodings for a sequence of tokens.
    The embedding dimension must be even, as the formula uses pairs of dimensions for sine and cosine.
    (from "Attention Is All You Need" paper, section 3.5 - https://arxiv.org/pdf/1706.03762;
    additional resources: https://kazemnejad.com/blog/transformer_architecture_positional_encoding/,
    https://pub.aimind.so/creating-sinusoidal-positional-embedding-from-scratch-in-pytorch-98c49e153d6)

    :param token_sequence_length: Length of the token sequence (T)
    :param token_embedding_dim:  Dimensionality of the token embeddings (d_model), must be even
    :return: Tensor of shape (token_sequence_length, token_embedding_dim) containing the sinusoidal positional embeddings
    """
    if token_embedding_dim % 2 != 0:
        raise ValueError(f"Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={token_embedding_dim})")

    T = token_sequence_length
    d = token_embedding_dim

    # Create indices for positions (shape=(T,)) and unsqueeze to shape=(T, 1) for broadcasting
    pos = torch.arange(0, T).unsqueeze_(1)
    # Create an empty tensor of shape (T, d) to hold the positional embeddings
    emb = torch.zeros(T, d)
    # Calculate the denominators 10000^(2i/d) for the sine/cosine computations; i.e.
    # the scaling factors for each embedding dimension index
    div_term = torch.pow(10000, 2*torch.arange(0, d//2) / d)  # shape=(d//2,)

    # Finally, embedding[i, 2k  ] = sin(pos[i] / div_term[k])
    # and      embedding[i, 2k+1] = cos(pos[i] / div_term[k])
    emb[:, 0::2] = torch.sin(pos * div_term)
    emb[:, 1::2] = torch.cos(pos * div_term)
    return emb  # (T, d) = (token_sequence_length, token_embedding_dim)


class AudioEncoderStem(nn.Module):
    """
    Two conv layers (filter width 3, GELU) where the second
    has stride 2, followed by sinusoidal positional embeddings.
    Input:  (B, n_mel, T)
    Output: (B, T', d_model)
    """
    def __init__(self, n_mels: int = 80, d_model: int = 512):
        """
        :param n_mels:  80 as the default as per the paper
        :param d_model: 512 as the default in the base model (tiny=384, small=768, medium=1024, large=1280)
        """
        super().__init__()
        # "a small stem consisting of two convolution layers with a filter width of 3"
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1, stride=1)
        # "where the second convolution layer has a stride of two"
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2)
        # "and the GELU activation function (Hendrycks & Gimpel, 2016)"
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_mels, T)
        x = self.gelu(self.conv1(x))   # (B, d_model, T)
        x = self.gelu(self.conv2(x))   # (B, d_model, T') where T' = T // 2

        # Rearrange to (B, T', d_model) for transformer
        x = x.permute(0, 2, 1)

        # Add sinusoidal positional embeddings
        seq_len = x.shape[1]
        d_model = x.shape[2]
        pe = sinusoidal_positional_embedding(seq_len, d_model).to(x.device)
        x = x + torch.unsqueeze(pe, dim=0)  # shape=(B, T', d_model) + (T', d_model) -> (B, T', d_model)

        return x


# --- Quick sanity check ---
if __name__ == "__main__":
    # Simulate a 30-second audio clip
    waveform = torch.randn(1, 16000 * 30)
    log_mel_spectrogram = compute_log_mel_spectrogram(waveform)   # (1, 80, T)
    log_mel_spectrogram = normalize(log_mel_spectrogram)
    print("Log-mel shape:", log_mel_spectrogram.shape)            # e.g. (1, 80, 3000)

    stem = AudioEncoderStem(n_mels=80, d_model=512)
    out = stem(log_mel_spectrogram)
    print("Stem output shape:", out.shape)
