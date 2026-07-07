import ast
import configparser
from dataclasses import dataclass, fields


@dataclass
class TrainingConfig:
    total_epochs: int
    warmup_epochs: int
    batch_size_train: int
    batch_size_val: int
    n_encoder_layers: int
    n_decoder_layers: int
    num_files: int
    max_len: int
    adam_init_lr: float
    adam_betas: tuple
    adamw_weight_decay: float
    n_mel_bins: int
    num_workers_dataloader: int
    d_model: int
    label_smoothing: float
    use_ctc_head: bool
    checkpoint_path: str
    num_warmup_steps: int
    num_training_steps: int


def auto_typed(section: configparser.SectionProxy) -> dict:
    """Convert every value in a config section to its inferred Python type."""
    result = {}
    for key, val in section.items():
        val = val.split("#")[0].strip()  # defensively strip trailing inline comments
        try:
            result[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            result[key] = val  # not a literal (e.g. a file path) -> keep as string
    return result


def load_config(section: configparser.SectionProxy) -> TrainingConfig:
    raw = auto_typed(section)
    valid_keys = {f.name for f in fields(TrainingConfig)}
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    filtered["num_warmup_steps"] = filtered["warmup_epochs"] * filtered["num_files"]
    filtered["num_training_steps"] = filtered["total_epochs"] * filtered["num_files"]
    return TrainingConfig(**filtered)


def get_config(name: str) -> TrainingConfig:
    config = configparser.ConfigParser()
    if not config.read("utils/configs.cfg"):
        raise FileNotFoundError("config.cfg not found")
    print(config.sections())
    return load_config(config["CONFIG_REGULARIZATION"])
