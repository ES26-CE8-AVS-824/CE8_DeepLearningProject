import datetime
import torch
from mini_whisper import *
from mini_whisper.model import MiniWhisper
from mini_whisper.decoder import transcribe
from transformers import WhisperTokenizer
def validate(model, dataloader, tokenizer, DEVICE):
    with torch.no_grad():
        for i, batch in list(enumerate(dataloader))[:1]:
            log_mels = batch['log_mel']  # (B, n_mels, T)
            log_mels = torch.nan_to_num(log_mels, nan=0.0, posinf=10.0, neginf=-10.0)
            log_mels = torch.clamp(log_mels, -10, 10)
            
            if log_mels.shape != (log_mels.shape[0], 80, 3000):
                print(f"Skipping bad batch {i} shape {log_mels.shape}")
                continue
                
            inp = log_mels.to(DEVICE)
            targets_cpu = convert_transcripts_to_targets(batch['transcript'], tokenizer)
            targets = targets_cpu.to(DEVICE, non_blocking=True)
            
            output = transcribe.transcribe(model, inp)
            print(output.shape, output[0])
            target_texts = [x+'\n' for x in batch['transcript']]
            # print(f"Output: {output_text}")
        
        
def train(model, dataloader, optimizer, loss_fn, tokenizer, DEVICE, epochs=1):
    for epoch in range(epochs):
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
            outputs = model(log_mels, targets)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            loss.backward()
            optimizer.step()
                
            
            if i % 10 == 0:
                print(f'Batch {i}, Loss: {loss.item():.4f}')
                
            del log_mels, targets, outputs
            torch.cuda.memory.empty_cache()
            
    # Save the model with name date and epoch count
    torch.save(model.state_dict(), f"model_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{epoch}.pth")
    
def convert_transcripts_to_targets(transcripts, tokenizer):
    max_len = 448
    BATCH_SIZE = len(transcripts)
    
    seqs = list(map(lambda t: tokenizer.encode(t), transcripts)) 

    # Pad all sequences to the same length (max_len)
    txt_padded = torch.zeros(BATCH_SIZE, max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        txt_padded[i, :len(s)] = torch.tensor(s[:max_len], dtype=torch.long)

    return txt_padded

def load_libriSpeech(split, 
                     download_dataset=True, 
                     batch_size=16, 
                     shuffle=True, 
                     num_workers=4, 
                     n_mel_bins=80, 
                     ):
    
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

def load_model(split="dev-clean", BATCH_SIZE=16, N_MELS=80, D_MODEL=128, N_HEADS=4, N_LAYERS=4, MAX_LEN=448):
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

    dataloader = load_libriSpeech(split, batch_size=BATCH_SIZE, n_mel_bins=N_MELS)
    
    print(f'\nLoading tokenizer...')
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")
    
    print(f'\nLoading model...')
    model = MiniWhisper(
        vocab_size=tokenizer.vocab_size+1000,
        n_mels=N_MELS,
        d_model=D_MODEL,
        n_encoder_layers=N_ENCODER_LAYERS,
        n_decoder_layers=N_DECODER_LAYERS,
        n_heads=N_HEADS,
        max_text_len=MAX_TEXT_LEN
    ).to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    return model, dataloader, loss_fn, optimizer, tokenizer, DEVICE
    
    
if __name__ == "__main__":
    model, train_dataloader, loss_fn, optimizer, tokenizer, DEVICE = load_model(split='dev-clean')
    model.load_state_dict(torch.load("model_2026-02-24_09-54-38_0.pth", map_location=DEVICE))
    val_dataloader = load_libriSpeech('dev-clean', batch_size=1, n_mel_bins=model.n_mel_bins)
    validate(model, val_dataloader, tokenizer, DEVICE)
    #train(model, dataloader, optimizer, loss_fn, tokenizer, DEVICE)