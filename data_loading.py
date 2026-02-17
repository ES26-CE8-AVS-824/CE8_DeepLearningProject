import json
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader

from audio import load_and_resample, pad_or_trim, compute_log_mel_spectrogram, normalize


def load_librispeech_flat(root_dir: str, save_to_json_path: str = None) -> list:
    """
    Load LibriSpeech dataset into a flat JSON structure.

    Args:
        root_dir: Path to LibriSpeech parent directory (e.g., 'data/LibriSpeech/dev-clean')
        save_to_json_path: Optional path to save output as JSON file

    Returns:
        List of utterance dictionaries

    Output format:
    [
        {
            "path": "relative/path/to/file.flac",
            "transcripts": {
                "original": "TRANSCRIPT TEXT",
                "mini-whisper": ""
            }
        },
        ...
    ]
    """
    root_path = Path(root_dir)
    dataset = []

    for speaker_dir in sorted(root_path.iterdir()):
        if not speaker_dir.is_dir():
            continue

        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue

            trans_files = list(chapter_dir.glob("*.trans.txt"))
            if not trans_files:
                continue

            trans_file = trans_files[0]

            # Parse transcripts
            transcripts = {}
            with open(trans_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        utterance_id, text = parts
                        transcripts[utterance_id] = text

            # Match FLAC files with transcripts
            for flac_file in sorted(chapter_dir.glob("*.flac")):
                utterance_id = flac_file.stem

                if utterance_id in transcripts:
                    dataset.append({
                        "path": str(flac_file.relative_to(root_path.parent)),
                        "transcripts": {
                            "original": transcripts[utterance_id],
                            "mini-whisper": ""
                        }
                    })

    if save_to_json_path:
        with open(save_to_json_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(dataset)} utterances to {save_to_json_path}")

    print(f"Loaded {len(dataset)} utterances from {root_dir}")
    return dataset


def load_librispeech_hierarchical(root_dir: str, save_to_json_path: str = None) -> dict:
    """
    Load LibriSpeech dataset preserving speaker/chapter hierarchy.

    Args:
        root_dir: Path to LibriSpeech parent directory (e.g., 'data/LibriSpeech/dev-clean')
        save_to_json_path: Optional path to save output as JSON file

    Returns:
        Dictionary with speaker/chapter hierarchy

    Output format:
    {
        "speaker_id": {
            "chapter_id": [
                {
                    "path": "relative/path/to/file.flac",
                    "transcripts": {
                        "original": "TRANSCRIPT TEXT",
                        "mini-whisper": ""
                    }
                },
                ...
            ]
        }
    }
    """
    root_path = Path(root_dir)
    dataset = {}

    for speaker_dir in sorted(root_path.iterdir()):
        if not speaker_dir.is_dir():
            continue

        speaker_id = speaker_dir.name
        dataset[speaker_id] = {}

        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue

            chapter_id = chapter_dir.name
            chapter_data = []

            trans_files = list(chapter_dir.glob("*.trans.txt"))
            if not trans_files:
                continue

            trans_file = trans_files[0]

            # Parse transcripts
            transcripts = {}
            with open(trans_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        utterance_id, text = parts
                        transcripts[utterance_id] = text

            # Match FLAC files with transcripts
            for flac_file in sorted(chapter_dir.glob("*.flac")):
                utterance_id = flac_file.stem

                if utterance_id in transcripts:
                    chapter_data.append({
                        "path": str(flac_file.relative_to(root_path.parent)),
                        "transcripts": {
                            "original": transcripts[utterance_id],
                            "mini-whisper": ""
                        }
                    })

            if chapter_data:
                dataset[speaker_id][chapter_id] = chapter_data

    total_utterances = sum(len(chapter) for speaker in dataset.values() for chapter in speaker.values())

    if save_to_json_path:
        with open(save_to_json_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Saved {total_utterances} utterances from {len(dataset)} speakers to {save_to_json_path}")

    print(f"Loaded {total_utterances} utterances from {len(dataset)} speakers from {root_dir}")
    return dataset


class LibriSpeechLocalDataset(Dataset):
    """
    PyTorch Dataset for LibriSpeech that loads and preprocesses local audio files on-the-fly
    or optionally preloads them into memory.

    Each item returns:
        - log_mel: Preprocessed log-mel spectrogram (n_mels, T)
        - transcript: Original transcript text
        - audio_path: Path to the audio file
    """

    def __init__(
        self,
        root_dir: str,
        preload_data: bool = False,
    ):
        """
        Args:
            root_dir: Path to LibriSpeech directory (e.g., 'data/LibriSpeech/dev-clean')
            preload_data: If True, load all audio into memory (faster but uses more RAM)
        """
        self.root_dir = Path(root_dir)
        self.base_path = self.root_dir.parent  # Go up to LibriSpeech parent
        self.preload_data = preload_data

        # Load dataset metadata
        self.data = load_librispeech_flat(str(root_dir))

        # Optionally preload all audio
        self.preloaded_mels = None
        if preload_data:
            print("Preloading all audio files into memory...")
            self.preloaded_mels = [self._load_and_preprocess(i) for i in range(len(self.data))]
            print(f"Preloaded {len(self.preloaded_mels)} audio files")

    def __len__(self) -> int:
        return len(self.data)

    def _load_and_preprocess(self, idx: int) -> torch.Tensor:
        """Load and preprocess a single audio file to log-mel spectrogram."""
        item = self.data[idx]
        audio_path = self.base_path / item['path']

        # Load and resample
        waveform = load_and_resample(str(audio_path))

        # Pad or trim to 30 seconds
        waveform = pad_or_trim(waveform)

        # Compute log-mel spectrogram
        log_mel = compute_log_mel_spectrogram(waveform)

        # Normalize
        log_mel = normalize(log_mel)

        # Remove batch dimension: (1, n_mels, T) -> (n_mels, T)
        log_mel = log_mel.squeeze(0)

        return log_mel

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset."""
        item = self.data[idx]

        # Get log-mel spectrogram (either from preloaded or load on-the-fly)
        if self.preloaded_mels is not None:
            log_mel = self.preloaded_mels[idx]
        else:
            log_mel = self._load_and_preprocess(idx)

        return {
            'log_mel': log_mel,
            'transcript': item['transcripts']['original'],
            'audio_path': item['path'],
        }


def create_librispeech_dataloader(
    root_dir: str,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    preload_data: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for LibriSpeech dataset.

    Args:
        root_dir: Path to LibriSpeech directory
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for data loading
        preload_data: Whether to preload all audio into memory

    Returns:
        PyTorch DataLoader
    """
    dataset = LibriSpeechLocalDataset(root_dir, preload_data=preload_data)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Speeds up data transfer to GPU
    )

    return dataloader

