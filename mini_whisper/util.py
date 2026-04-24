from collections import defaultdict
import torch
import torch.nn as nn


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())


def parameter_breakdown(model: nn.Module):
    breakdown = defaultdict(int)

    #print("All param names:")
    #for name, param in model.named_parameters():
    #    print(name)
    #print()


    for name, param in model.named_parameters():
        n = param.numel()

        # ---- Custom grouping logic ----
        if name.startswith("encoder"):
            if "stem" in name:
                breakdown["encoder.stem"] += n
            elif "pos" in name:
                breakdown["encoder.positional_embedding"] += n
            elif "attn" in name or "attention" in name:
                breakdown["encoder.attention"] += n
            elif "mlp" in name or "ff" in name or "feedforward" in name:
                breakdown["encoder.mlp"] += n
            elif "ln" in name or "norm" in name:
                breakdown["encoder.layernorm"] += n
            else:
                breakdown["encoder.other"] += n
                print(f"encoder.OTHER: {name}")

        elif name.startswith("decoder"):
            if "token_embedding" in name or "token_emb" in name:
                breakdown["decoder.token_embedding"] += n
            elif "pos" in name:
                breakdown["decoder.positional_embedding"] += n
            elif "attn" in name or "attention" in name:
                breakdown["decoder.attention"] += n
            elif "mlp" in name or "ff" in name or "feedforward" in name:
                breakdown["decoder.mlp"] += n
            elif "ln" in name or "norm" in name:
                breakdown["decoder.layernorm"] += n
            else:
                breakdown["decoder.other"] += n
                print(f"decoder.OTHER: {name}")

        else:
            breakdown["other"] += n

    return breakdown


def print_param_breakdown(model: nn.Module):
    total = count_parameters(model)
    breakdown = parameter_breakdown(model)

    print(f"\nTotal parameters: {total:,}\n")

    for k, v in sorted(breakdown.items()):
        pct = 100 * v / total
        print(f"{k:35s} {v:12,d}  ({pct:6.2f}%)")


def convert_transcripts_to_targets(transcripts, tokenizer, max_len, prefix_token_ids=None):
    BATCH_SIZE = len(transcripts)

    if prefix_token_ids is not None:
        prefix = prefix_token_ids
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
