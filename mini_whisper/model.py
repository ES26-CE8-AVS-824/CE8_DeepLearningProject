"""
Complete Mini-Whisper model combining encoder and decoder.
"""

import torch
import torch.nn as nn

from mini_whisper.encoder.audioEncoder import AudioEncoder
from mini_whisper.decoder.textDecoder import TextDecoder



class MiniWhisper(nn.Module):
    """
    Complete Mini-Whisper model: Audio Encoder + Text Decoder.
    """

    def __init__(
        self,
        vocab_size: int,
        n_mels: int = 80,
        d_model: int = 512,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.0,
        max_text_len: int = 448,
    ):
        """
        Args:
            vocab_size: Size of the vocabulary
            n_mels: Number of mel frequency bins
            d_model: Model dimension
            n_encoder_layers: Number of encoder transformer blocks
            n_decoder_layers: Number of decoder transformer blocks
            n_heads: Number of attention heads
            dropout: Dropout rate
            max_text_len: Maximum text sequence length
        """
        super().__init__()

        self.encoder = AudioEncoder(
            n_mels=n_mels,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            dropout=dropout,
        )

        self.decoder = TextDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_text_len,
            n_layers=n_decoder_layers,
            n_heads=n_heads,
        )

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_text_len = max_text_len
        self.n_mel_bins = n_mels
        

    def forward(
        self,
        mel: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            mel: (batch, n_mels, seq_len) - Log-mel spectrograms
            tokens: (batch, seq_len) - Text token IDs

        Returns:
            (batch, seq_len, vocab_size) - Logits over vocabulary
        """
        # Encode audio: (B, n_mels, T) -> (B, T', d_model)
        z = self.encoder(mel)

        # Decode to text: (B, seq_len) -> (B, seq_len, vocab_size)
        logits = self.decoder(tokens, z)  # Using encoder output as cross-attention input

        return logits

    def encode_audio(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to features (useful for inference).

        Args:
            mel: (batch, n_mels, time) - Log-mel spectrogram

        Returns:
            (batch, seq_len, d_model) - Encoded audio features
        """
        return self.encoder(mel)

    def decode_tokens(self, tokens: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Decode tokens given encoder output (useful for inference).

        Args:
            tokens: (batch, seq_len) - Text token IDs
            audio_features: (batch, enc_seq_len, d_model) - Encoded audio features

        Returns:
            (batch, seq_len, vocab_size) - Logits over vocabulary
        """
        return self.decoder(tokens, audio_features)


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    n_mels = 80
    time_frames = 3000
    seq_len = 20
    vocab_size = 51864  # GPT-2 vocab size + special tokens

    # Create model
    model = MiniWhisper(
        vocab_size=vocab_size,
        n_mels=n_mels,
        d_model=512,
        n_encoder_layers=4,
        n_decoder_layers=4,
        n_heads=8,
    )

    # Test forward pass
    mel = torch.randn(batch_size, n_mels, time_frames)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits = model(mel, tokens)

    print(f"Input mel shape: {mel.shape}")
    print(f"Input tokens shape: {tokens.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

