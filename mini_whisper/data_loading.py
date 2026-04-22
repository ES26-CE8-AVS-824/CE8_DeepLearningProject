from typing import Dict, Any, List, Union

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchaudio.datasets import LIBRISPEECH
from transformers import WhisperTokenizer
from mini_whisper import N_MEL_BINS
from mini_whisper.audio import pad_or_trim, compute_log_mel_spectrogram, normalize, SAMPLE_RATE
from mini_whisper.util import convert_transcripts_to_targets as tokenize_transcripts


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
            sampler: Union[None, DistributedSampler] = None,
            num_workers: int = 2,
            target_sr: int = SAMPLE_RATE,
            n_mel_bins: int = N_MEL_BINS,
            rank: int = 0,
            world_size: int = 1,
            **kwargs
    ):
        """
        Args:
            split:
                Which LibriSpeech split to load (e.g., 'train-clean-100', 'dev-clean', 'test-clean')
            root_dir:
                Root directory where the dataset is stored or will be downloaded to
            folder_in_archive:
                Top-level folder in the archive (usually "LibriSpeech")
            download_dataset:
                Whether to download the dataset if not found
            batch_size:
                Batch size for DataLoader
            shuffle:
                Whether to shuffle the data each epoch
            num_workers:
                Number of subprocesses for data loading
            target_sr:
                Target sample rate for audio (default 16kHz)
            n_mel_bins:
                Number of Mel bins for spectrogram (default 80)
        """
        dataset = LIBRISPEECH(
            root=root_dir,
            url=split,
            folder_in_archive=folder_in_archive,
            download=download_dataset,
        )

        assert sampler is None or not shuffle, "Cannot use shuffle=True with an explicit sampler — sampler controls ordering"
        if sampler is not None:
            sampler = sampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=self._collate_and_preprocess,
            pin_memory=True,
            **kwargs
        )

        self.target_sr = target_sr
        self.n_mel_bins = n_mel_bins

        self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
        self.prefix = self.tokenizer.encode("")[:-1]  # Remove the EOS token from the prefix

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
        mel_lengths = []

        for item in batch:
            waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = item

            # Resample if needed
            if sample_rate != self.target_sr:
                waveform = torchaudio.functional.resample(waveform, sample_rate, self.target_sr)

            # Convert stereo to mono if needed
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            mel_length = compute_log_mel_spectrogram(waveform, n_mel_bins=self.n_mel_bins).shape[2]
            mel_length = min(mel_length, 3000)

            # Pad or trim to 30 seconds
            waveform = pad_or_trim(waveform)

            # Compute log-mel spectrogram
            log_mel = compute_log_mel_spectrogram(waveform, n_mel_bins=self.n_mel_bins)

            # Normalize
            log_mel = normalize(log_mel)

            # Remove batch dimension: (1, n_mels, T) -> (n_mels, T)
            log_mel = log_mel.squeeze(0)
            

            log_mels.append(log_mel)
            transcripts.append(transcript)
            mel_lengths.append(mel_length)
            audio_paths.append(f"{speaker_id}/{chapter_id}/{speaker_id}-{chapter_id}-{utterance_id:04d}.flac")

        # Stack all log-mels into a single batch tensor
        log_mels = torch.stack(log_mels, dim=0)  # (B, n_mels, T)
        # Tokenize transcripts to target token IDs, padding to max_len, 
        # max_len is curretly set to 224 which is the max text length used during training
        tokenized_transcripts, transcript_lengths = tokenize_transcripts(transcripts, self.tokenizer, max_len=224, prefix_token_ids=self.prefix)

        return {
            'audio_path': audio_paths,
            'log_mel': log_mels,
            'transcript': transcripts,
            'mel_length': mel_lengths,
            'tokenized_transcript': tokenized_transcripts,
            'transcript_length': transcript_lengths,
        }
