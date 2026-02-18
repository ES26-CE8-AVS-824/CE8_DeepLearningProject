from typing import Dict, Any, List

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchaudio.datasets import LIBRISPEECH

from mini_whisper.audio import pad_or_trim, compute_log_mel_spectrogram, normalize, SAMPLE_RATE


class LibriSpeechAudioPreprocessingDataLoader(DataLoader):
    """
    Custom DataLoader that applies audio preprocessing (pad/trim, mel-spectrogram, normalization)
    to batches of audio data from LIBRISPEECH dataset.

    This DataLoader wraps any audio dataset and handles all preprocessing in the collate function,
    converting raw audio waveforms to preprocessed log-mel spectrograms.
    """

    dataset: Dataset
    batch_size: int | None
    num_workers: int
    pin_memory: bool

    def __init__(
            self,
            split: str = 'dev-clean',
            root_dir: str = "./data",
            folder_in_archive: str = "LibriSpeech",
            download_dataset: bool = True,
            batch_size: int = 16,
            shuffle: bool = True,
            num_workers: int = 2,
            target_sr: int = SAMPLE_RATE,
            **kwargs
    ):
        """
        Args:
            dataset: Any audio dataset (e.g., LIBRISPEECH) that returns (waveform, sample_rate, transcript, ...)
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle the dataset
            num_workers: Number of worker processes for data loading
            **kwargs: Additional arguments to pass to DataLoader
        """
        dataset = LIBRISPEECH(
            root=root_dir,
            url=split,
            folder_in_archive=folder_in_archive,
            download=download_dataset,
        )

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_and_preprocess,
            pin_memory=True,
            **kwargs
        )

        self.target_sr = target_sr

    def _collate_and_preprocess(self, batch: List[tuple]) -> Dict[str, Any]:
        """
        Collate function that preprocesses raw audio into log-mel spectrograms.

        Args:
            batch: List of tuples from LIBRISPEECH dataset, each containing:
                   (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)

        Returns:
            Dictionary with:
                - audio_path: List of audio paths
                - log_mel: Batched log-mel spectrograms (B, n_mels, T)
                - transcript: List of transcripts
        """
        audio_paths = []
        log_mels = []
        transcripts = []

        for item in batch:
            waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = item

            # Resample if needed
            if sample_rate != self.target_sr:
                waveform = torchaudio.functional.resample(waveform, sample_rate, self.target_sr)

            # Convert stereo to mono if needed
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Pad or trim to 30 seconds
            waveform = pad_or_trim(waveform)

            # Compute log-mel spectrogram
            log_mel = compute_log_mel_spectrogram(waveform)

            # Normalize
            log_mel = normalize(log_mel)

            # Remove batch dimension: (1, n_mels, T) -> (n_mels, T)
            log_mel = log_mel.squeeze(0)

            log_mels.append(log_mel)
            transcripts.append(transcript)
            audio_paths.append(f"{speaker_id}/{chapter_id}/{speaker_id}-{chapter_id}-{utterance_id:04d}.flac")

        # Stack all log-mels into a single batch tensor
        log_mels = torch.stack(log_mels, dim=0)  # (B, n_mels, T)

        return {
            'audio_path': audio_paths,
            'log_mel': log_mels,
            'transcript': transcripts,
        }
