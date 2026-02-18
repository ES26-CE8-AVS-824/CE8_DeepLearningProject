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

# Model
from .model import (
    MiniWhisper,
    AudioEncoder,
)

# Tokenizer
from .tokenizer_wrapper import (
    WhisperTokenizer,
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
    # Model
    "MiniWhisper",
    "AudioEncoder",
    # Tokenizer
    "WhisperTokenizer",
]

