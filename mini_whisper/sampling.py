import math
import torch
from dataclasses import dataclass
from typing import Literal


@dataclass
class SamplingConfig:
    """
    Controls the decoder input strategy during training.

    mode:
        "teacher_forcing"   – standard training, ground-truth tokens only (default).
        "greedy"            – vanilla scheduled sampling: per-step probability of
                              replacing a ground-truth token with the model's
                              greedy prediction, controlled by a decay schedule.
        "confidence_aware"  – paper-style: use the first-pass softmax confidence
                              to decide per-position whether to keep ground-truth,
                              use the predicted token, or inject a random token
                              from the same sequence (target denoising).

    decay_mode (used only by "greedy"):
        "linear"            – p_gold = max(eps, 1 - k * step)
        "exponential"       – p_gold = k ^ step
        "inverse_sigmoid"   – p_gold = k / (k + exp(step / k))   [paper default]

    decay_k:
        Shape / rate of the decay curve.
        •  linear:           slope magnitude  (e.g. 1e-5 means p_gold hits 0 at ~100k steps)
        •  exponential:      base < 1         (e.g. 0.9999 per step)
        •  inverse_sigmoid:  k ≥ 1, larger k  → slower decay  (e.g. 10 000)

    decay_eps:
        Floor on p_gold for the linear schedule; prevents it from going below
        this value. Ignored by the other schedules.

    t_golden (confidence_aware only):
        Confidence threshold above which the predicted token replaces ground truth.
        Paper default: 0.9

    t_rand (confidence_aware only):
        Confidence threshold above which a random target token replaces ground truth
        (target denoising). Must be > t_golden. Set to 1.1 to disable denoising.
        Paper default: 0.95
    """
    mode: Literal["teacher_forcing", "greedy", "confidence_aware"] = "teacher_forcing"

    # --- greedy decay ---
    decay_mode: Literal["linear", "exponential", "inverse_sigmoid"] = "inverse_sigmoid"
    decay_k: float = 10_000.0
    decay_eps: float = 0.0

    # --- confidence-aware thresholds ---
    t_golden: float = 0.9
    t_rand: float = 0.95

    # protect fixed decoder prefix tokens, e.g. SOT + no-timestamps
    protected_prefix_len: int = 2


def _gold_token_probability(step: int, cfg: SamplingConfig) -> float:
    """
    Returns p_gold: probability of keeping a ground-truth token.
    1.0 = pure teacher forcing, 0.0 = always use the model's own output.
    Used only by the "greedy" mode.
    """
    k = cfg.decay_k
    if cfg.decay_mode == "linear":
        return max(cfg.decay_eps, 1.0 - k * step)
    elif cfg.decay_mode == "exponential":
        return k ** step
    elif cfg.decay_mode == "inverse_sigmoid":
        return k / (k + math.exp(step / k))
    else:
        raise ValueError(f"Unknown decay_mode: {cfg.decay_mode!r}")


def _disable_prefix_positions(mask: torch.Tensor, prefix_len: int) -> torch.Tensor:
    """
    Sets the first `prefix_len` positions to False in a (B, T) boolean mask.
    """
    if prefix_len > 0:
        prefix_len = min(prefix_len, mask.size(1))
        mask[:, :prefix_len] = False
    return mask


def apply_scheduled_sampling(
        model: torch.nn.Module,
        log_mels: torch.Tensor,
        tgt_in: torch.Tensor,
        global_step: int,
        device: torch.device,
        cfg: SamplingConfig,
) -> tuple[torch.Tensor, dict]:
    """
    Optionally replaces tokens in `tgt_in` using the model's first-pass predictions.

    Returns
    -------
    tgt_in_modified : Tensor  – decoder input for the second (gradient) pass.
    stats           : dict    – diagnostic values for logging.

    Implementation notes
    --------------------
    • The first forward pass runs under torch.no_grad() so it does NOT interact
      with DDP's gradient hooks; only the second pass (called by the caller with
      .backward()) triggers gradient synchronization.
    • `model` may be a DDP-wrapped module – both passes call through the wrapper
      normally, which is fine.
    """
    if cfg.mode == "teacher_forcing":
        return tgt_in, {}

    # ---- First pass (no gradients) ----------------------------------------
    with torch.no_grad():
        logits_first = model(log_mels, tgt_in)  # (B, T, vocab)

    B, T, _ = logits_first.shape
    prefix_len = min(cfg.protected_prefix_len, T)

    # ---- Greedy scheduled sampling ----------------------------------------
    if cfg.mode == "greedy":
        p_gold = _gold_token_probability(global_step, cfg)

        preds = logits_first.argmax(dim=-1)  # (B, T)
        predicted_tokens = torch.cat([tgt_in[:, :1], preds[:, :-1]], dim=1)  # (B, T)

        # Bernoulli mask: True where we replace GT with predicted token
        use_predicted = torch.rand(B, T, device=device) >= p_gold
        use_predicted = _disable_prefix_positions(use_predicted, prefix_len)

        tgt_in_modified = torch.where(use_predicted, predicted_tokens, tgt_in)
        replaced_frac = use_predicted[:, prefix_len:].float().mean().item() if T > prefix_len else 0.0

        stats = {
            "ss/p_gold": p_gold,
            "ss/replaced_frac": replaced_frac,
            "ss/protected_prefix_len": prefix_len,
        }
        return tgt_in_modified, stats

    # ---- Confidence-aware scheduled sampling ------------------------------
    elif cfg.mode == "confidence_aware":
        # Confidence = max probability of the softmax distribution at each position.
        # Using the predicted translation probability (PTP) approach from the paper
        # (Eq. 3): conf(t) = P(y_t | y_<t, X, θ)
        probs = torch.softmax(logits_first.float(), dim=-1)  # (B, T, vocab)
        confidence, preds = probs.max(dim=-1)  # both (B, T)
        predicted_tokens = torch.cat([tgt_in[:, :1], preds[:, :-1]], dim=1)        # (B, T) shifted

        # Random tokens: uniformly sample a position from the same sequence
        # to simulate wordy / wrong-order perturbations (3.3 in the paper)
        rand_idx = torch.randint(0, T, (B, T), device=device)
        random_tokens = tgt_in.gather(1, rand_idx)  # (B, T)

        # Three-way selection (Eq. 7 from the paper):
        #   conf ≤ t_golden  →  keep ground-truth (golden)
        #   t_golden < conf ≤ t_rand  →  use predicted token
        #   conf > t_rand    →  inject random token (target denoising)
        use_predicted = confidence > cfg.t_golden
        use_random = confidence > cfg.t_rand

        use_predicted = _disable_prefix_positions(use_predicted, prefix_len)
        use_random = _disable_prefix_positions(use_random, prefix_len)

        tgt_in_modified = tgt_in.clone()
        tgt_in_modified = torch.where(use_predicted, predicted_tokens, tgt_in_modified)
        tgt_in_modified = torch.where(use_random, random_tokens, tgt_in_modified)

        pred_frac = use_predicted[:, prefix_len:].float().mean().item() if T > prefix_len else 0.0
        rand_frac = use_random[:, prefix_len:].float().mean().item() if T > prefix_len else 0.0
        mean_conf = confidence[:, prefix_len:].mean().item() if T > prefix_len else confidence.mean().item()

        stats = {
            "ss/mean_confidence": mean_conf,
            "ss/pred_replaced_frac": pred_frac,
            "ss/rand_replaced_frac": rand_frac,
        }
        return tgt_in_modified, stats

    else:
        raise ValueError(f"Unknown sampling mode: {cfg.mode!r}")
