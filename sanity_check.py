import numpy as np
import torch

from transformers import WhisperTokenizer

CONFIG = {
    "total_epochs": 1,
    "warmup_epochs": 1,
    "batch_size_train": 32,
    "batch_size_val": 16,
    # where 28539 is the number of files, 10 is the number of epochs wanted and 16 is the batch size
    # Hardcoded for now but TODO remove this as a global
    "num_files": 28539,
    "max_len": 50,
}


def convert_transcripts_to_targets(transcripts, tokenizer, max_len, prefix_token_ids=None):
    BATCH_SIZE = len(transcripts)

    if prefix_token_ids is not None:
        prefix = [prefix_token_ids]
    else:
        prefix = tokenizer.encode("")[:-1]  # Remove the EOS token from the prefix

    # Calculate encoded lengths before tokenization then add tokens
    encoded_batch = [tokenizer.encode(t, add_special_tokens=False) for t in transcripts]
    seqs = [prefix + t + [tokenizer.eos_token_id] for t in encoded_batch]

    # Pad all sequences to the same length (max_len)
    txt_padded = torch.full((BATCH_SIZE, max_len), tokenizer.pad_token_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        txt_padded[i, :len(s)] = torch.tensor(s[:max_len], dtype=torch.long)

    # Calculate transcript lengths (capped at max_len)
    transcript_len = torch.tensor([min(len(s), max_len) for s in seqs], dtype=torch.long)

    return txt_padded, transcript_len


def load_model():
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")
    return tokenizer


def main():
    tokenizer = load_model()
    transcripts = [
        "hello world",
        "this is a test",
        "whisper tokenizer is working",
    ]
    targets, transcript_len = convert_transcripts_to_targets(transcripts, tokenizer,
                                             max_len=CONFIG["max_len"])
    print("Targets:", targets)

    for i, (seq, l) in enumerate(zip(targets, transcript_len)):
        print(f"\nTranscript {i}: '{transcripts[i]}'; len={l}")
        for token_id in seq.tolist():
            print(f"  {token_id:>6} -> {repr(tokenizer.decode([token_id]))}")


if __name__ == "__main__":
    main()
