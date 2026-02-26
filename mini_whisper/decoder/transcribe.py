from mini_whisper.audio import *
import numpy as np
import torch

from mini_whisper.decoder import *

def transcribe(model, audio, **kwargs):
    """transcribe a given audio file with a model

    Args:
        model (_type_): Mini_whisper model
        audio (_type_): waveform audio file
    """
    N_FRAMES = N_SAMPLES // STRIDE
    
    mel_spectrogram = compute_log_mel_spectrogram(audio, SAMPLE_RATE, N_MEL_BINS, N_FFT, STRIDE)
    mel_spectrogram = normalize(mel_spectrogram)
    
    content_frames = mel_spectrogram.shape[-1] - N_FRAMES
    content_duration = float(content_frames * STRIDE / SAMPLE_RATE)
    print(content_duration)
    
    

if __name__ == '__main__':
    audio = torchaudio.load('data/LibriSpeech/dev-clean/174/50561/174-50561-0000.flac')[0]
    print(audio)
    transcribe('x', audio)
    