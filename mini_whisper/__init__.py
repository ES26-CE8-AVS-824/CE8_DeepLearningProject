"""
Mini-Whisper: A simplified implementation of OpenAI's Whisper model.

This package provides:
- Audio preprocessing utilities (resampling, mel-spectrograms, normalization)
- Data loading from LibriSpeech dataset
- Encoder and decoder components for speech-to-text transcription
- Complete MiniWhisper model
- Tokenizer wrapper using tiktoken
- Training and inference utilities
"""

# Version info
__version__ = "0.1.0"
__author__ = "Your Name"

# Audio preprocessing
from .audio import (
    load_and_resample,
    pad_or_trim,
    compute_log_mel_spectrogram,
    normalize,
    SAMPLE_RATE,
    N_SAMPLES,
    N_FFT,
    STRIDE,
    N_MEL_BINS,
)

# Data loading
from .data_loading import (
    LibriSpeechAudioPreprocessingDataLoader,
)

# Encoder components
from .encoder.preprocessing_stem import (
    AudioEncoderStem,
    sinusoidal_positional_embedding,
)
from .encoder.audioEncoder import (
    AudioEncoder,
)

# Decoder components
from .decoder.textDecoder import (
    TextDecoder,
)
from .decoder.positionalEncoder import (
    LearnedPositionalEncoding,
)

# Transformer components
from .transformer.MHA import (
    MultiHeadAttention,
    TransformerBlock,
)
from .transformer.MHA_simple import (
    SimpleMultiHeadAttention,
    SimpleTransformerBlock,
)

# Complete model
from .model import (
    MiniWhisper,
)

# Define what gets imported with "from mini_whisper import *"
__all__ = [
    # Audio processing
    "load_and_resample",
    "pad_or_trim",
    "compute_log_mel_spectrogram",
    "normalize",
    "SAMPLE_RATE",
    "N_SAMPLES",
    "N_FFT",
    "STRIDE",
    "N_MEL_BINS",
    # Data loading
    "LibriSpeechAudioPreprocessingDataLoader",
    # Encoder
    "AudioEncoderStem",
    "sinusoidal_positional_embedding",
    "AudioEncoder",
    # Decoder
    "TextDecoder",
    "LearnedPositionalEncoding",
    # Transformer
    "MultiHeadAttention",
    "TransformerBlock",
    "SimpleMultiHeadAttention",
    "SimpleTransformerBlock",
    # Complete model
    "MiniWhisper",
]



