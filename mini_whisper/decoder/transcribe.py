from mini_whisper.audio import *
import numpy as np
import torch
from transformers import WhisperTokenizer
from mini_whisper.decoder import *
from typing import List, Tuple


def transcribe(model, audio, tokenizer, DEVICE, beam_size: int, max_length: int):
    """transcribe a given audio file with a model

    Args:
        model (_type_): Mini_whisper model
        audio (_type_): Mel spectrogram
        tokenizer (_type_): Whisper tokenizer
        DEVICE (_type_): Device to put data on
    """

    SOS = tokenizer.encode('')  # Start of sentence (<SOS><TRANSCRIBE><EOS>)
    if len(SOS) == 0:  # fallback if encode('') returns nothing
        SOS = [tokenizer.sot] if hasattr(tokenizer, 'sot') else [50258]
    EOS = SOS[-1]

    prompt = SOS[:-1] if len(SOS) > 1 else SOS
    encoder_output = model.encoder(audio)  # Encoder output (computed once)

    hypotheses: List[Tuple[List[int], float]] = [(prompt.copy(), 0.0)]  # Each hypothesis = (list_of_token_ids, cumulative_log_probability)
    finished: List[Tuple[List[int], float]] = []

    # decoding loop
    for step in range(max_length):
        if not hypotheses:
            break

        new_hypotheses = []

        for seq, score in hypotheses:
            input_ids = torch.tensor([seq], dtype=torch.long, device=DEVICE)  # tensor (shape [1, current_length])
            logits = model.decoder(input_ids, encoder_output)  # [1, seq_len, vocab]
            next_logits = logits[0, -1, :]

            log_probs = torch.log_softmax(next_logits, dim=-1)  # log-probabilities conversion
            topk_logp, topk_ids = torch.topk(log_probs, k=beam_size * 2)  # Get top-k candidates (more than beam_size for diversity)

            for logp, token_id in zip(topk_logp, topk_ids):
                new_seq = seq + [token_id.item()]
                new_score = score + logp.item()

                if token_id.item() == EOS:
                    finished.append((new_seq, new_score))
                else:
                    new_hypotheses.append((new_seq, new_score))

        if new_hypotheses:  # Only keeping the best `beam_size` active hypotheses
            new_hypotheses.sort(key=lambda x: x[1], reverse=True)  # highest score first
            hypotheses = new_hypotheses[:beam_size]
        else:
            hypotheses = []

    finished.extend(hypotheses)  # If any still-active hypotheses
    if not finished:
        best_seq = prompt
        best_score = 0.0
    else:
        best_seq, best_score = max(finished, key=lambda x: x[1])  # Selecting hypothesis with the highest total log-probability
    print(f"Best sequence: {best_seq} with score {best_score:.4f}")  # Remove when its working
    print(f"Decoded {len(best_seq)} tokens (beam_size={beam_size})")

    # Same shape as before:
    return torch.tensor([best_seq], device=DEVICE)
