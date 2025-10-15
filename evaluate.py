# pip install -q jiwer==3.0.3

import jiwer
import torchaudio
from tqdm import tqdm

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