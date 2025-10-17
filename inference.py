import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from lhotse import Recording
from lhotse.features import Fbank, FbankConfig
import math

@torch.inference_mode()
def wavs_to_fbank_tensors(
    wav_paths,
    target_sr: int = 16000,
    num_mel_bins: int = 80,
    device: str = "cpu",
):
    """
    Args:
        wav_paths: list[str] of WAV paths.
    Returns:
        batch: (B, T_max, F) float32 tensor (zero-padded in time)
        lengths: (B,) int32 tensor of valid frame lengths before padding
        ids: list[str] recording ids (filenames without extension)
    """
    PAD_VALUE = math.log(1e-10)
    extractor = Fbank(FbankConfig(sampling_rate=target_sr, num_mel_bins=num_mel_bins, device=device))
    resamplers = {}
    feats_list, lengths, ids = [], [], []

    for p in wav_paths:
        rec = Recording.from_file(p)                 # metadata + lazy loader
        wav = rec.load_audio()                       # np.ndarray, shape: (channels, samples)
        wav = wav[0] if wav.ndim == 2 else wav       # mono

        # Resample if needed
        if rec.sampling_rate != target_sr:
            if rec.sampling_rate not in resamplers:
                resamplers[rec.sampling_rate] = torchaudio.transforms.Resample(
                    rec.sampling_rate, target_sr
                )
            wav_t = torch.from_numpy(wav).unsqueeze(0)      # (1, samples)
            wav = resamplers[rec.sampling_rate](wav_t).squeeze(0).numpy()

        # FBANK in memory (np.ndarray -> torch)
        fbank_np = extractor.extract(samples=wav, sampling_rate=target_sr)  # (T, F) np.float32
        fbank_t = torch.from_numpy(fbank_np)                                # CPU tensor
        feats_list.append(fbank_t)
        lengths.append(fbank_t.shape[0])
        ids.append(rec.id)

    features = pad_sequence(feats_list, batch_first=True, padding_value=PAD_VALUE)      # (B, T_max, F), zero-padded
    lengths = torch.tensor(lengths, dtype=torch.int32)
    return {"inputs" : features, "supervisions" : {"num_frames" : lengths}}

# Example:
# batch, lengths, ids = wavs_to_fbank_tensors(list_of_wavs, device="cuda" if torch.cuda.is_available() else "cpu")
# logits = model(batch.to(model.device), lengths=lengths)  # depends on your model API
