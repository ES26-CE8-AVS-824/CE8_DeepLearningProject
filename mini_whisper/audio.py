import numpy as np
import torch
import torchaudio

from torch.nn.functional import pad
from torchaudio.transforms import MelSpectrogram


# Inspired by https://github.com/openai/whisper/blob/main/whisper/audio.py


SAMPLE_RATE = 16000                                    # "All audio is re-sampled to 16,000 Hz"
DEFAULT_AUDIO_LENGTH_SECONDS = 30                      # "We break audio files into 30-second segments"
N_SAMPLES = SAMPLE_RATE * DEFAULT_AUDIO_LENGTH_SECONDS # 30s at 16kHz = 480,000 samples

N_FFT = 400      # "on 25-millisecond windows"        -> 25ms window @ 16kHz = 400 samples
STRIDE = 160     # "with a stride of 10 milliseconds" -> 10ms stride @ 16kHz = 160 samples
N_MEL_BINS = 80  # "and an 80-channel log-magnitude Mel spectrogram representation is computed"

def load_and_resample(path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Load an audio file and resample it to the target sample rate if needed.
    Stereo audio is converted to mono by averaging the channels.

    :param path:      Path to the audio file
    :param target_sr: Target sample rate (default 16kHz as per the paper)
    :return:          Mono audio waveform at 16kHz, shape (1, T)
    """
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    # Stereo to mono by averaging channels if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform  # (1, T)



def pad_or_trim(array, length: int = N_SAMPLES, axis: int = -1):
    """
    Pad or trim an audio array/tensor to exactly `length` samples along the specified axis.

    If the array is longer than `length`, it is trimmed from the right.
    If shorter, it is zero-padded on the right. If already the correct
    length, it is returned unchanged.

    :param array: Audio data as either a NumPy array or a PyTorch tensor.
    :param length: Target number of samples along `axis`. Defaults to N_SAMPLES (30s at 16kHz).
    :param axis: Axis along which to pad or trim. Defaults to -1 (last axis).
    :return: Array or tensor of the same type as input, with `array.shape[axis] == length`.
    """

    # Trim
    if array.shape[axis] > length:
        array = array.take(range(length), axis=axis) if isinstance(array, np.ndarray) \
                else array.index_select(axis, torch.arange(length, device=array.device))

    # Pad
    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = np.pad(array, pad_widths) if isinstance(array, np.ndarray) \
                else pad(array, [p for sizes in pad_widths[::-1] for p in sizes])

    return array


def compute_log_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    n_mel_bins: int = N_MEL_BINS,
    n_fft: int = N_FFT,
    hop_length: int = STRIDE,
) -> torch.Tensor:
    """
    Convert the audio waveform to a log(10)-magnitude Mel spectrogram.

    :param waveform:    (1, T) mono audio waveform at 16kHz
    :param sample_rate: "All audio is re-sampled to 16,000 Hz"
    :param n_mel_bins:  "and an 80-channel log-magnitude Mel spectrogram representation is computed"
    :param n_fft:       "on 25-millisecond windows"        -> 25ms window @ 16kHz = 400 samples
    :param hop_length:  "with a stride of 10 milliseconds" -> 10ms stride @ 16kHz = 160 samples
    :return:             Log-magnitude Mel spectrogram of shape (1, n_mels, T') where T' depends on the length of the input waveform and the hop_length
    """
    mel_transform = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mel_bins,
        window_fn=torch.hann_window,
    )
    T = waveform.shape[1]           # T = 30 * 16000 = 480,000
    mel = mel_transform(waveform)   # (1, n_mels, T)
    T_ = mel.shape[2]
    # With center=True (the default), MelSpectrogram pads the waveform by P = n_fft // 2
    # on both sides before applying the STFT, giving:
    #   T' = floor((T + 2P - n_fft) / hop_length) + 1
    #      = floor(T / hop_length) + 1
    # For example, with T = 480,000 and hop_length = 160: T' = 3,000 + 1 = 3,001.
    # We drop the last frame to get a clean T' = 3,000, which is what the encoder expects.
    # This is safe because the final frame typically contains only silence or zero-padding.
    assert T_ == (T // hop_length) + 1
    mel = mel[:,:,:-1]
    log_mel = torch.log10(mel.clamp(min=1e-10))
    return log_mel  # (1, n_mels, T')


def normalize(log_mel: torch.Tensor) -> torch.Tensor:
    """
    Globally scale to [-1, 1] with approx zero mean.
    Constants mean(log_mel) ≈ -4.0 and scale(log_mel) ≈ 4.0 are reasonable defaults for any 16kHz speech-domain audio
    (most of the audio data is in the range [-4 - 4, -4 + 4] dB = [-8, 0] dB),
    and we do not want to be calculating them at inference time.
    Therefore, we shift by -(-4) = +4 and scale by /4.
    Clipping to max-8 dB helps with outliers.
    """
    # clip to max-8 dB to handle outliers and so that the range is always [-8, 0] and normalization can actually do its job
    log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
    # shift to [-1, 0], then scale to [-1, 1]
    log_mel = (log_mel + 4.0) / 4.0
    return log_mel
