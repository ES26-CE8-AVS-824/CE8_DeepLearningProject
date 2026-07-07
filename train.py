import datetime
import os
import re
import torch
import torch.distributed as dist
import torch.nn.functional as F

import wandb
from utils.config_parser import get_config as get_config_parser_config
from mini_whisper import *
from mini_whisper.model import MiniWhisper
from mini_whisper.decoder import transcribe
from mini_whisper.encoder.ctc_head import CTCHead
from mini_whisper.sampling import SamplingConfig, apply_scheduled_sampling
from eval.wer import wer_calc
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    WhisperTokenizer,
    get_cosine_with_min_lr_schedule_with_warmup_lr_rate,
)

CONFIG = get_config_parser_config("CONFIG_REGULARIZATION")


def validate(
    model,
    dataloader,
    tokenizer,
    device,
    loss_fn,
    num_batches=None,
    epoch=None,
    step=None,
    is_main=True,
):
    model.eval()
    total_wer, total_examples = 0.0, 0
    val_loss = 0.0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if num_batches is not None and i >= num_batches:
                break

            inp_log_mels = batch["log_mel"].to(device)
            targets = batch["tokenized_transcript"].to(device)
            mel_lengths = torch.tensor(batch["mel_length"]).to(device)

            tgt_in = targets[:, :-1]
            tgt_out = targets[:, 1:]

            outputs = model(inp_log_mels, tgt_in)
            loss = loss_fn(outputs.reshape(-1, outputs.size(-1)), tgt_out.reshape(-1))
            val_loss += loss.item()

            raw = model.module if hasattr(model, "module") else model
            text_outputs = transcribe.transcribe(
                raw,
                inp_log_mels,
                mel_lengths,
                tokenizer,
                device,
                max_new_tokens=100,
                beam_width=5,
                length_penalty=0.6,
            )
            for i, txt in enumerate(text_outputs):
                decoded = tokenizer.decode(txt)
                ref = batch["transcript"][i]
                wer = wer_calc(ref, decoded)
                total_wer += wer
                print(
                    f"[{total_examples}] WER beam: {wer:.2%}\n  REF: {ref}\n  HYP: {decoded}\n"
                )
                total_examples += 1

    mean_wer = total_wer / total_examples if total_examples > 0 else float("nan")
    mean_val = val_loss / total_examples if total_examples > 0 else float("nan")
    print(f"\nValidation WER: {mean_wer:.2%} over {total_examples} examples")
    print(f"Validation Loss: {mean_val:.4f}")

    # Log a scalar per validation run / epoch
    if is_main and wandb.run is not None:
        wandb.log(
            {
                "val/wer": mean_wer,
                "val/num_examples": total_examples,
                "val/epoch": epoch,
                "val/loss": mean_val,
                "val/perplexity": torch.exp(torch.tensor(mean_val)),
            },
            step=step,
        )

    return mean_wer


def train(
    model,
    dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    loss_fn,
    tokenizer,
    DEVICE,
    epochs=1,
    start_epoch=0,
    is_main=True,
    ctc_head=None,
    loss_lambda=0.5,
    sampling_cfg: SamplingConfig = None,
):
    if sampling_cfg is None:
        sampling_cfg = SamplingConfig()  # defaults to teacher_forcing

    steps_per_epoch = len(dataloader)
    global_step = start_epoch * steps_per_epoch

    if is_main:
        if sampling_cfg.mode == "greedy":
            print(
                f"  decay={sampling_cfg.decay_mode!r}  k={sampling_cfg.decay_k}", end=""
            )
        elif sampling_cfg.mode == "confidence_aware":
            print(
                f"  t_golden={sampling_cfg.t_golden}  t_rand={sampling_cfg.t_rand}",
                end="",
            )
        print()

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()

        inner = model.module if hasattr(model, "module") else model

        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        for i, batch in enumerate(dataloader):
            log_mels = batch["log_mel"].to(DEVICE)  # (B, n_mels, T)
            targets = batch["tokenized_transcript"].to(DEVICE)

            tgt_in = targets[:, :-1]  # [BOS, t1, t2, ..., t_{n-1}]
            tgt_out = targets[:, 1:]  # [t1,  t2, t3, ..., t_n    ]

            optimizer.zero_grad()

            # Apply scheduled sampling (if given)
            tgt_in_modified, ss_stats = apply_scheduled_sampling(
                inner, log_mels, tgt_in, global_step, DEVICE, sampling_cfg
            )

            # Get features and mel lengths just from the encoder, to be used for both CTC and decoder loss
            z, mel_lengths = inner.encoder(
                log_mels, torch.tensor(batch["mel_length"]).to(DEVICE)
            )  # Pass input lengths for proper handling in the encoder
            # ... and use this also as the main pass
            logits = inner.decoder(tgt_in_modified, z)
            loss_cr = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            if ctc_head is not None:
                ctc_softmax_logits = ctc_head(z)
                loss_ctc = F.ctc_loss(
                    ctc_softmax_logits,
                    tgt_out,
                    mel_lengths,
                    batch["transcript_length"].to(DEVICE),
                    blank=tokenizer.pad_token_id,
                    reduction="mean",
                    zero_infinity=True,
                )
                effective_lambda = (
                    scheduler.get_last_lr()[0] / CONFIG.adam_init_lr * loss_lambda
                )  # Scale lambda with scheduler
                loss = effective_lambda * loss_ctc + (1 - effective_lambda) * loss_cr
            else:
                loss = loss_cr
                loss_ctc = None  # BUG FIX 2: was referenced in wandb.log but could be undefined

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if is_main and wandb.run is not None:
                log_dict = {
                    "train/grad_norm": grad_norm,
                    #                   "train/loss": loss.item(),
                    "train/loss_cr": loss_cr.item(),
                    "train/epoch": epoch,
                    **ss_stats,
                }
                if ctc_head is not None:
                    log_dict["train/loss_ctc"] = loss_ctc.item()
                    wandb.log(log_dict, step=global_step)
            global_step += 1

            if i % 100 == 0:
                print(f"Epoch {epoch + 1}, Batch {i}, Loss: {float(loss.item())}")

            del (
                log_mels,
                targets,
                tgt_in,
                tgt_out,
                tgt_in_modified,
                z,
                logits,
                loss,
                loss_cr,
            )
            if ctc_head is not None:
                del ctc_softmax_logits, loss_ctc
            # torch.cuda.memory.empty_cache()
            scheduler.step()

        # Run validation and log at the same global_step each 5th epoch
        if (epoch + 1) % 5 == 0:
            if val_dataloader is not None:
                validate(
                    model,
                    val_dataloader,
                    tokenizer,
                    DEVICE,
                    loss_fn,
                    epoch=epoch,
                    step=global_step,
                    is_main=is_main,
                )
                model.train()

            # Save the model with name date and epoch count
            if is_main:
                ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
                model_name = f"ckpts/model_{ts}_epoch-{epoch + 1}.pth"
                inner = model.module if hasattr(model, "module") else model
                torch.save(inner.state_dict(), model_name)
                if wandb.run is not None:
                    wandb.save(model_name)


def load_libriSpeech(
    split,
    download_dataset=True,
    batch_size=16,
    shuffle=True,
    sampler=None,
    num_workers=4,
    n_mel_bins=80,
    rank: int = 0,
    world_size: int = 1,
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
        sampler=sampler,
        num_workers=num_workers,
        n_mel_bins=n_mel_bins,
        world_size=world_size,
        rank=rank,
    )

    print(f"\nCreating DataLoader for: {DATA_DIR}")
    print(f"Batch size: {batch_size}")
    return dataloader


def load_model(
    N_MELS=80,
    D_MODEL=128,
    N_HEADS=4,
    N_ENCODER_LAYERS=4,
    N_DECODER_LAYERS=4,
    MAX_LEN=448,
    local_rank: int = 0,
    device: str = "cuda",
    kwargs=None,
    distributed: bool = True,
):
    """
    Loads a MiniWhisper model with the specified hyperparameters.

    Args:
        split (str): The split of the LibriSpeech dataset to load. Defaults to "dev-clean".
        BATCH_SIZE (int): The batch of audio files to use as a batch size for the model. Defaults to 16.
        N_MELS (int): The number of mel bins to use. Defaults to 80.
        D_MODEL (int): The dimensionality of the model. Defaults to 128.
        N_HEADS (int): The number of attention heads. Defaults to 4.
        N_ENCODER_LAYERS (int): The number of encoder layers. Defaults to 4.
        N_DECODER_LAYERS (int): The number of decoder layers. Defaults to 4.
        MAX_LEN (int): The maximum length of the text to generate. Defaults to 448.
    Returns:
        model (MiniWhisper): The loaded MiniWhisper model.
        dataloader (DataLoader): The DataLoader for the dataset.
        tokenizer (WhisperTokenizer): The tokenizer for the dataset.
        optimizer (Adam): The optimizer for the model.
        criterion (CrossEntropyLoss): The loss function for the model.
        DEVICE (torch.device): The device to use for training.
    """

    MAX_TEXT_LEN = MAX_LEN

    print("=" * 60)
    print("Mini-Whisper Training - Data Loading & Preprocessing")
    print("=" * 60)

    print(f"\nLoading tokenizer...")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")

    kwargs = kwargs or {}

    print(f"\nLoading model...")
    model = MiniWhisper(
        vocab_size=len(tokenizer),  # tokenizer.vocab_size + 1000,
        n_mels=N_MELS,
        d_model=D_MODEL,
        n_encoder_layers=N_ENCODER_LAYERS,
        n_decoder_layers=N_DECODER_LAYERS,
        n_heads=N_HEADS,
        max_text_len=MAX_TEXT_LEN,
    ).to(device)

    if distributed:
        ddp_device_id = (
            device.index
            if isinstance(device, torch.device) and device.index is not None
            else local_rank
        )
        model = DDP(model, device_ids=[ddp_device_id], output_device=ddp_device_id)

    # For warmup actually the peak lr, can be used for annealing as a base also
    adam_init_lr = (
        1e-4 if kwargs.get("adam_init_lr") is None else kwargs["adam_init_lr"]
    )
    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id, label_smoothing=CONFIG.label_smoothing
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=adam_init_lr,
        betas=CONFIG.adam_betas,
        eps=1e-9,
        weight_decay=CONFIG.adamw_weight_decay,
    )

    # https://github.com/huggingface/transformers/blob/e5ad3946209fb96db5e9965b3eb67d69cc3749e0/src/transformers/optimization.py#L389
    scheduler = get_cosine_with_min_lr_schedule_with_warmup_lr_rate(
        optimizer,
        num_warmup_steps=CONFIG.num_warmup_steps,
        num_training_steps=CONFIG.num_training_steps,
        num_cycles=0.5,  # single half‑cosine
        min_lr_rate=0.1,  # decay to 10% of initial lr (= eta_min_ratio)
        warmup_lr_rate=None,  # initial lr: None to start at (step+1)/num_warmup_steps
    )

    return model, loss_fn, optimizer, scheduler, tokenizer


def run(
    mode: str = "eval",
    validate_during_training: bool = False,
    distributed: bool = False,
    load_from_ckpt_path: None | str = None,
    use_wandb: bool = True,
    sampling: None | SamplingConfig = None,
):
    # DDP init
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        is_main = local_rank == 0
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        world_size = 1
        is_main = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, loss_fn, optimizer, scheduler, tokenizer = load_model(
        D_MODEL=CONFIG.d_model,
        N_ENCODER_LAYERS=CONFIG.n_encoder_layers,
        N_DECODER_LAYERS=CONFIG.n_decoder_layers,
        local_rank=local_rank,
        device=device,
        distributed=distributed,
        kwargs={"warmup": True, "adam_init_lr": CONFIG.adam_init_lr},
    )

    raw = model.module if hasattr(model, "module") else model
    start_epoch = 0
    if load_from_ckpt_path is not None:
        state = torch.load(load_from_ckpt_path, map_location=device)
        raw.load_state_dict(state)
        search = re.search(r"epoch-(\d+)", load_from_ckpt_path)
        if search:
            start_epoch = int(search.group(1))

    if is_main:
        print_param_breakdown(raw)

        CONFIG.ckpt_path = load_from_ckpt_path
        if use_wandb:
            run = wandb.init(
                project="mini-whisper",
                entity="mini-whisper",
                config=CONFIG,
                name=f"mini-whisper-{mode}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
            )

            run.watch(model, log="all", log_freq=100)

    if mode == "train":
        train_dataloader = load_libriSpeech(
            "train-clean-360",
            batch_size=CONFIG.batch_size_train,
            n_mel_bins=CONFIG.n_mel_bins,
            num_workers=CONFIG.num_workers_dataloader,
            sampler=DistributedSampler if distributed else None,
            shuffle=False if distributed else True,
            rank=local_rank,
            world_size=world_size,
        )

        if validate_during_training:
            val_dataloader = load_libriSpeech(
                "dev-clean",
                batch_size=CONFIG.batch_size_val,
                n_mel_bins=CONFIG.n_mel_bins,
                num_workers=CONFIG.num_workers_dataloader,
                sampler=DistributedSampler if distributed else None,
                shuffle=False,
                rank=local_rank,
                world_size=world_size,
            )
        else:
            val_dataloader = None

        if CONFIG.use_ctc_head:
            ctc_head = CTCHead(in_dim=CONFIG.d_model, vocab_size=len(tokenizer)).to(
                device
            )
            if distributed:
                ctc_head = DDP(
                    ctc_head, device_ids=[device.index], output_device=device.index
                )
        else:
            ctc_head = None

        train(
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            scheduler,
            loss_fn,
            tokenizer,
            device,
            epochs=CONFIG.total_epochs,
            start_epoch=start_epoch,
            is_main=is_main,
            ctc_head=ctc_head,
            sampling_cfg=sampling,
        )

    elif mode == "eval":
        val_dataloader = load_libriSpeech(
            "train-clean-360",
            batch_size=CONFIG.batch_size_val,
            n_mel_bins=CONFIG.n_mel_bins,
            num_workers=CONFIG.num_workers_dataloader,
            sampler=DistributedSampler if distributed else None,
            shuffle=False if distributed else True,
            rank=local_rank,
            world_size=world_size,
        )
        validate(model, val_dataloader, tokenizer, device, loss_fn, is_main=is_main)
    if distributed and dist.is_initialized():
        dist.destroy_process_group()
    if is_main:
        run.finish()


if __name__ == "__main__":
    sampling = SamplingConfig(
        mode="confidence_aware",
        t_golden=0.9,  # replace with predicted token above this confidence
        t_rand=0.95,  # additionally inject a random token above this confidence
    )
    run(
        mode="eval",
        validate_during_training=False,
        distributed=False,
        load_from_ckpt_path="ckpts/model_2026-04-19_01-34_epoch-270.pth",
        use_wandb=True,
    )
