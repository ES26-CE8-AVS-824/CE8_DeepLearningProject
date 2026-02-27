from mini_whisper.audio import *
import numpy as np
import torch
from transformers import WhisperTokenizer
from mini_whisper.decoder import *

def transcribe(model, audio, **kwargs):
    """transcribe a given audio file with a model

    Args:
        model (_type_): Mini_whisper model
        audio (_type_): Mel spectrogram
    """
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base") # Have this an an input
    SOS = tokenizer.encode('') # Start of sentence
    EOS = SOS[-1]

    decoder_input = torch.tensor(SOS[:-1])
    max_length = 100
    for step in range(max_length):
        logits = model.decoder(decoder_input, encoder_output)
        next_token = logits.argmax(-1)[:, -1:]  # greedy on last token
        decoder_input = torch.cat([decoder_input, next_token], dim=1)

        if next_token == EOS:
            break

    return decoder_input
    

if __name__ == '__main__':
    audio = torchaudio.load('data/LibriSpeech/dev-clean/174/50561/174-50561-0000.flac')[0]
    model, dataloader, loss_fn, optimizer, tokenizer, DEVICE = load_model(split='train-clean-100')

    transcribe('x', audio)
    