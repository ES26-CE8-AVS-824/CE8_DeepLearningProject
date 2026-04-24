from fvcore.nn import FlopCountAnalysis
import torch
from mini_whisper.model import MiniWhisper
from fvcore.nn import FlopCountAnalysis

# pip install -U fvcore (install dependency)
# 1. inputs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 51865

# mel-spectrogram shape for Mini-Whisper
dummy_audio = torch.randn(1, 80, 3000, device=DEVICE)     #[batch, n_mels, time_frames]

# decoder prompt
dummy_decoder_input = torch.randint(0, VOCAB_SIZE, (1, 50), device=DEVICE)  # 50 tokens is realistic

# Build model
model = MiniWhisper(
    vocab_size=VOCAB_SIZE,
    n_mels=80,
    d_model=512,
    n_encoder_layers=4,
    n_decoder_layers=4,
    n_heads=8,
).to(DEVICE)

# FLOP counting
model.eval()

with torch.no_grad():
    # Encoder only
    encoder_flops = FlopCountAnalysis(model.encoder, dummy_audio)
    encoder_gflops = encoder_flops.total() / 1e9

    # Decoder only (encoder output)
    encoder_output, _ = model.encoder(dummy_audio)
    decoder_flops = FlopCountAnalysis(model.decoder, (dummy_decoder_input, encoder_output))
    decoder_gflops = decoder_flops.total() / 1e9

    total_gflops = encoder_gflops + decoder_gflops

flops = FlopCountAnalysis(model, (dummy_audio, dummy_decoder_input))
print(flops.total())
print(flops.by_operator())
print(flops.by_module())
print(flops.by_module_and_operator())

print("MINI-WHISPER FLOPs ESTIMATION")
print(f"Encoder          : {encoder_gflops:.2f} GFLOPs")
print(f"Decoder          : {decoder_gflops:.2f} GFLOPs")
print(f"Total (one pass) : {total_gflops:.2f} GFLOPs")
