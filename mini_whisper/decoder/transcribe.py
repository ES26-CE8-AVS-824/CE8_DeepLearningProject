from mini_whisper.audio import *
import numpy as np
import torch
from transformers import WhisperTokenizer
from mini_whisper.decoder import *

def transcribe(model, audio, tokenizer, DEVICE):
    """transcribe a given audio file with a model

    Args:
        model (_type_): Mini_whisper model
        audio (_type_): Mel spectrogram
        tokenizer (_type_): Whisper tokenizer
        DEVICE (_type_): Device to put data on
    """
    SOS = tokenizer.encode('') # Start of sentence (<SOS><TRANSCRIBE><EOS>)
    EOS = SOS[-1]

    decoder_input = torch.tensor([SOS[:-1]]).to(DEVICE)
    encoder_output = model.encoder(audio)
    max_length = 100
    for step in range(max_length):
        logits = model.decoder(decoder_input, encoder_output)
        next_token = logits.argmax(-1)[:, -1:]  # greedy on last token
        decoder_input = torch.cat([decoder_input, next_token], dim=1)
        if step == 0:
            decoder_input = decoder_input[:, 1:]

        if next_token == EOS:
            break
    print(decoder_input.shape)
    return decoder_input
    
    