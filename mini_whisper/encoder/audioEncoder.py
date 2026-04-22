import torch
from torch import nn

from mini_whisper.encoder.preprocessing_stem import AudioEncoderStem
from mini_whisper.transformer.MHA_simple import SimpleTransformerBlock


class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int = 80, d_model: int = 512, n_heads: int = 8, n_layers: int = 4, dropout: float = 0.0):
        super().__init__()
        self.stem = AudioEncoderStem(n_mels, d_model)
        self.transformer_encoder_blocks = nn.ModuleList([
            SimpleTransformerBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, n_mels, T)
        x, input_lengths = self.stem(x, input_lengths)  # (B, T', d_model)
        for block in self.transformer_encoder_blocks:
            x = block(x)  # (B, T', d_model)
        return x, input_lengths  # (B, T', d_model), (B,) or None
