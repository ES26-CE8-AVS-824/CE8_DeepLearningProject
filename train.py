import torch

from data_loading import LibriSpeechAudioPreprocessingDataLoader
from encoder.preprocessing_stem import AudioEncoderStem


def main():

    # TODO: this is only a dummy script to test the data loading and preprocessing pipeline, and the encoder stem

    # Configuration
    SPLIT = "dev-clean"
    DATA_DIR = "./data"
    FOLDER_IN_ARCHIVE = "LibriSpeech"
    BATCH_SIZE = 16
    N_MELS = 80
    D_MODEL = 512

    print("=" * 60)
    print("Mini-Whisper Training - Data Loading & Preprocessing")
    print("=" * 60)

    # Create DataLoader
    print(f"\nCreating DataLoader for: {DATA_DIR}")
    print(f"Batch size: {BATCH_SIZE}")

    dataloader = LibriSpeechAudioPreprocessingDataLoader(
        split=SPLIT,
        root_dir=DATA_DIR,
        folder_in_archive=FOLDER_IN_ARCHIVE,
        download_dataset=True,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    print(f"\nDataLoader created with {len(dataloader.dataset)} samples")
    print(f"  Number of batches: {len(dataloader)}")

    # Initialize encoder stem
    print(f"\nInitializing AudioEncoderStem (n_mels={N_MELS}, d_model={D_MODEL})")
    stem = AudioEncoderStem(n_mels=N_MELS, d_model=D_MODEL)
    stem.eval()

    # Process first batch as a test
    print(f"\nProcessing first batch...")
    batch = next(iter(dataloader))

    log_mels = batch['log_mel']  # (B, n_mels, T)
    transcripts = batch['transcript']
    audio_paths = batch['audio_path']

    print(f"\nBatch contents:")
    print(f"  Log-mel shape: {log_mels.shape}")
    print(f"  Number of transcripts: {len(transcripts)}")

    print(f"\nFirst 3 samples in batch:")
    for i in range(min(3, len(transcripts))):
        print(f"  {i+1}. {audio_paths[i]}")
        print(f"     Transcript: {transcripts[i][:60]}...")

    # Process through encoder stem
    with torch.no_grad():
        batch_features = stem(log_mels)

    print(f"\nEncoder output shape: {batch_features.shape}")

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Total files in dataset: {len(dataloader.dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Total batches: {len(dataloader)}")
    print(f"\nInput shape (log-mel): {log_mels.shape}")
    print(f"  - Batch size: {log_mels.shape[0]}")
    print(f"  - Mel bins: {log_mels.shape[1]}")
    print(f"  - Time frames: {log_mels.shape[2]}")
    print(f"\nOutput shape (features): {batch_features.shape}")
    print(f"  - Batch size: {batch_features.shape[0]}")
    print(f"  - Sequence length: {batch_features.shape[1]}")
    print(f"  - Feature dimension: {batch_features.shape[2]}")
    print("\nDataLoader and preprocessing pipeline verified successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()


