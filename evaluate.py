# pip install -q jiwer==3.0.3

import jiwer
import torchaudio
from tqdm import tqdm
import torch

transformation = jiwer.Compose([
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ToLowerCase(),
    jiwer.RemoveKaldiNonWords(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

def compute_wer(refs, hyps, return_scalar=True):
    """
    Compute WER given references and hypotheses.

    Args:
        refs: A list of reference texts.
        hyps: A list of hypothesis texts.  
    Returns:
        wer: Word Error Rate (WER) score.
    """
    assert len(refs) == len(hyps), f"len(refs)={len(refs)} != len(hyps)={len(hyps)}"
    all_wer = []
    hyps = [str(x) for x in hyps]
    for pred, gt in tqdm(zip(hyps, refs)):
        t_pred = pred.replace("▁", " ").lower()
        wer_score = jiwer.wer(
            gt,
            t_pred,
            reference_transform=transformation,
            hypothesis_transform=transformation
        )

        all_wer.append(wer_score)
    wer = sum(all_wer) / len(all_wer)
    if return_scalar:
        return wer
    else:
        return all_wer

def get_duration(audio_path):
    info = torchaudio.info(audio_path)
    duration = info.num_frames / info.sample_rate
    return duration

def merge_audio(wav_paths, output_path):
# wav_paths = [
#     "/path/audio1.wav",
#     "/path/audio2.wav",
#     "/path/audio3.wav"
# ]
# output_path = "/path/output.wav"

    waveforms = []
    sample_rate = None

    for p in wav_paths:
        wav, sr = torchaudio.load(p)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(f"Sample rate mismatch: {p} has {sr}, expected {sample_rate}")
        waveforms.append(wav)

    # Concatenate along the time dimension
    concatenated = torch.cat(waveforms, dim=1)

    # Save result
    torchaudio.save(output_path, concatenated, sample_rate)
    print(f"✅ Concatenated WAV saved to {output_path}")