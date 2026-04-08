from collections import defaultdict

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
