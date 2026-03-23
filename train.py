import datetime
import torch
import wandb
from mini_whisper import *
from mini_whisper.model import MiniWhisper
from mini_whisper.decoder import transcribe
from eval.wer import jiwer_wer
from transformers import WhisperTokenizer

CONFIG = {
    "total_epochs": 20,
    "warmup_epochs": 10,
    "batch_size": 16,
    # where 28539 is the number of files, 10 is the number of epochs wanted and 16 is the batch size
    "num_files": 28539,
    "max_len": 448,
    "adam_init_lr": 3e-4,
    "adam_betas": (0.9, 0.98),
    "n_mel_bins": 80,
}

# Hardcoded for now but TODO remove this as a global
CONFIG["num_warmup_steps"] = CONFIG["warmup_epochs"] * CONFIG["num_files"] // CONFIG["batch_size"]

def validate(model, dataloader, tokenizer, device, num_batches=None, epoch=None, step=None):
    model.eval()
    total_wer, total_examples = 0.0, 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if num_batches is not None and i >= num_batches:
                break

            log_mels = batch['log_mel']
            # log_mels = torch.nan_to_num(log_mels, nan=0.0, posinf=10.0, neginf=-10.0)
            # log_mels = torch.clamp(log_mels, -10, 10)

            # if log_mels.shape[1:] != (80, 3000):
            #     print(f"Skipping batch {i}: unexpected shape {log_mels.shape}")
            #     continue

            inp = log_mels.to(device)
            # targets_cpu = convert_transcripts_to_targets(batch['transcript'], tokenizer)
            # targets = targets_cpu.to(device, non_blocking=True)

            outputs = transcribe.transcribe(model, inp, tokenizer, device)
            decoded = tokenizer.decode(outputs)

            for ref, hyp in zip(batch['transcript'], decoded):
                wer = jiwer_wer(ref, hyp)
                total_wer += wer
                total_examples += 1
                print(f"[{total_examples}] WER: {wer:.4f}\n  REF: {ref}\n  HYP: {hyp}")

    mean_wer = total_wer / total_examples if total_examples > 0 else float('nan')
    print(f"\nValidation WER: {mean_wer:.4f} over {total_examples} examples")

    # Log a scalar per validation run / epoch
    wandb.log(
        {
            "val/wer": mean_wer,
            "val/num_examples": total_examples,
            "val/epoch": epoch,
        },
        step=step,
    )

    return mean_wer


def train(model, dataloader, val_dataloader, optimizer, scheduler, loss_fn, tokenizer, DEVICE, epochs=1):
    global_step = 0
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(dataloader):
            log_mels = batch['log_mel']  # (B, n_mels, T)
            # log_mels = torch.nan_to_num(log_mels, nan=0.0, posinf=10.0, neginf=-10.0)
            # log_mels = torch.clamp(log_mels, -10, 10)

            # if log_mels.shape != (log_mels.shape[0], 80, 3000):
            #     print(f"Skipping bad batch {i} shape {log_mels.shape}")
            #     continue

            log_mels = log_mels.to(DEVICE)

            targets_cpu = convert_transcripts_to_targets(batch['transcript'], tokenizer)
            targets = targets_cpu.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            tgt_in = targets[:, :-1]
            tgt_out = targets[:, 1:]

            outputs = model(log_mels, tgt_in)
            loss = loss_fn(outputs.reshape(-1, outputs.size(-1)),
                           tgt_out.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/epoch": epoch
                },
                step=global_step,
            )
            global_step += 1

            if i % 100 == 0:
                print(f'Epoch {epoch + 1}, Batch {i}, Loss: {float(loss.item())}')

            del log_mels, targets, outputs
            torch.cuda.memory.empty_cache()
            scheduler.step()

        # Run validation and log at the same global_step
        if val_dataloader is not None:
            validate(model, val_dataloader, tokenizer, DEVICE, epoch=epoch, step=global_step)
            model.train()

        # Save the model with name date and epoch count
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        model_name = f"ckpts/model_{ts}_epoch-{epoch + 1}.pth"
        torch.save(model.state_dict(), model_name)
        wandb.save(model_name)


def convert_transcripts_to_targets(transcripts, tokenizer):
    max_len = CONFIG["max_len"]
    BATCH_SIZE = len(transcripts)

    prefix = [
        tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
        tokenizer.convert_tokens_to_ids("<|en|>"),
        tokenizer.convert_tokens_to_ids("<|transcribe|>"),
        tokenizer.convert_tokens_to_ids("<|notimestamps|>"),
    ]

    seqs = [prefix + tokenizer.encode(t, add_special_tokens=False) + [tokenizer.eos_token_id] for t in transcripts]

    # Pad all sequences to the same length (max_len)
    txt_padded = torch.full((BATCH_SIZE, max_len), tokenizer.pad_token_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        txt_padded[i, :len(s)] = torch.tensor(s[:max_len], dtype=torch.long)

    return txt_padded


def load_libriSpeech(split,
                     download_dataset=True,
                     batch_size=16,
                     shuffle=True,
                     num_workers=4,
                     n_mel_bins=80,
                     ) -> LibriSpeechAudioPreprocessingDataLoader:
    DATA_DIR = "./data"
    FOLDER_IN_ARCHIVE = "LibriSpeech"
    dataloader = LibriSpeechAudioPreprocessingDataLoader(
        split=split,
        root_dir=DATA_DIR,
        folder_in_archive=FOLDER_IN_ARCHIVE,
        download_dataset=download_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        n_mel_bins=n_mel_bins
    )

    print(f"\nCreating DataLoader for: {DATA_DIR}")
    print(f"Batch size: {batch_size}")
    return dataloader

def linear_warmup_lambda(current_step: int):
# current_step starts at 0
    if current_step >= CONFIG["num_warmup_steps"]:
        return 1.0  # stay at peak_lr
    return float(current_step + 1) / float(max(1, CONFIG["num_warmup_steps"]))

def load_model(N_MELS=80, D_MODEL=128, N_HEADS=4, N_LAYERS=4, MAX_LEN=448, kwargs={}):
    """
    Loads a MiniWhisper model with the specified hyperparameters.

    Args:
        split (str): The split of the LibriSpeech dataset to load. Defaults to "dev-clean".
        BATCH_SIZE (int): The batch of audio files to use as a batch size for the model. Defaults to 16.
        N_MELS (int): The number of mel bins to use. Defaults to 80.
        D_MODEL (int): The dimensionality of the model. Defaults to 128.
        N_HEADS (int): The number of attention heads. Defaults to 4.
        N_LAYERS (int): The number of encoder and decoder layers. Defaults to 4.
        MAX_LEN (int): The maximum length of the text to generate. Defaults to 448.
    Returns:
        model (MiniWhisper): The loaded MiniWhisper model.
        dataloader (DataLoader): The DataLoader for the dataset.
        tokenizer (WhisperTokenizer): The tokenizer for the dataset.
        optimizer (Adam): The optimizer for the model.
        criterion (CrossEntropyLoss): The loss function for the model.
        DEVICE (torch.device): The device to use for training.
    """

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_ENCODER_LAYERS = N_LAYERS
    N_DECODER_LAYERS = N_LAYERS
    MAX_TEXT_LEN = MAX_LEN

    print("=" * 60)
    print("Mini-Whisper Training - Data Loading & Preprocessing")
    print("=" * 60)

    print(f'\nLoading tokenizer...')
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")

    print(f'\nLoading model...')
    model = MiniWhisper(
        vocab_size=tokenizer.vocab_size + 1000,
        n_mels=N_MELS,
        d_model=D_MODEL,
        n_encoder_layers=N_ENCODER_LAYERS,
        n_decoder_layers=N_DECODER_LAYERS,
        n_heads=N_HEADS,
        max_text_len=MAX_TEXT_LEN
    ).to(DEVICE)

    # For warmup actually the peak lr, can be used for annealing as a base also
    adam_init_lr = 1e-4 if kwargs.get("adam_init_lr") is None else kwargs["adam_init_lr"]
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=adam_init_lr, betas=CONFIG["adam_betas"], eps=1e-9)

    if kwargs.get("warmup") is not None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=linear_warmup_lambda
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=CONFIG["num_warmup_steps"],
            eta_min=adam_init_lr*0.1 #10% of initial lr
        )


    return model, loss_fn, optimizer, scheduler, tokenizer, DEVICE


def main(mode: str = "eval"):
    model, loss_fn, optimizer, scheduler, tokenizer, DEVICE = load_model(kwargs={"warmup": True, "adam_init_lr": CONFIG["adam_init_lr"]})

    run = wandb.init(
        project="mini-whisper",
        entity="mini-whisper",
        config=CONFIG,
        name=f"mini-whisper-{mode}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )

    run.watch(model, log="all", log_freq=100)

    if mode == "train":
        train_dataloader = load_libriSpeech('train-clean-100',
                                            batch_size=CONFIG["batch_size"],
                                            n_mel_bins=CONFIG["n_mel_bins"])
        val_dataloader = load_libriSpeech('test-clean', batch_size=1,
                                          n_mel_bins=CONFIG["n_mel_bins"])
        train(model, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn, tokenizer, DEVICE, epochs=CONFIG["total_epochs"])
    elif mode == "eval":
        ckpt_path = "ckpts/model_2026-03-18"
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state)
        val_dataloader = load_libriSpeech('test-clean', batch_size=1, n_mel_bins=CONFIG["n_mel_bins"])
        validate(model, val_dataloader, tokenizer, DEVICE)
    run.finish()


if __name__ == "__main__":
    main(mode="train")
