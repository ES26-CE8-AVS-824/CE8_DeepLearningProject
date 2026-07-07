import torch
import torchlens as tl

from train import CONFIG, load_model


def main(
        ckpt_path: str
):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    model, loss_fn, optimizer, scheduler, tokenizer = load_model(
        D_MODEL=CONFIG["d_model"],
        device=device,
        distributed=False,
        kwargs={
            "warmup": True,
            "adam_init_lr": CONFIG["adam_init_lr"]
        }
    )
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    model.eval()

    B = 1
    # Encoder input: log-mel spectrogram (B, n_mels, T)
    inp = torch.randn(B, CONFIG["n_mel_bins"], 3000, device=device)
    # Decoder input: token ids (B, max_len - 1), using pad_token_id as dummy
    tgt_in = torch.full(
        (B, CONFIG["max_len"] - 1),
        fill_value=tokenizer.pad_token_id,
        dtype=torch.long,
        device=device
    )

    model_history = tl.log_forward_pass(
        model, (inp, tgt_in),
        layers_to_save='all',
        vis_mode='rolled',
        vis_outpath="mini-whisper_arch_graph",
        vis_save_only=True,
        vis_fileformat="png"
    )


if __name__ == "__main__":
    main(
        ckpt_path="ckpts/best_before_val_gain.pth"
    )
