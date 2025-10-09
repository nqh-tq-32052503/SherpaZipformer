import os 
MAX_DURATION = 600
os.environ["MAX_DURATION"] = str(MAX_DURATION)
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*torchaudio\.sox_effects\.sox_effects\.apply_effects_tensor has been deprecated.*"
)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

from lhotse import load_manifest, CutSet
from lhotse.dataset import K2SpeechRecognitionDataset
from lhotse.dataset.sampling import SimpleCutSampler
from lhotse.dataset.cut_transforms import (
    ExtraPadding, PerturbTempo, PerturbVolume, ReverbWithImpulseResponse
)
from lhotse.features import Fbank, FbankConfig
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import random
from tester import Tester
import time 
import jiwer
from tqdm import tqdm
import pandas as pd


transformation = jiwer.Compose([
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ToLowerCase(),
    jiwer.RemoveKaldiNonWords(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

PAD_LOG = float(torch.log(torch.tensor(1e-10)))
TRAIN_CUTS = "/mnt/Disk800/thangnv102/gobangvts/processed_training_data/cuts_trim.jsonl.gz"
VALID_CUTS = "/mnt/Disk800/thangnv102/gobangvts/processed_validation_data/cuts_trim.jsonl.gz"
MATERIAL_DIR = "/mnt/Disk800/thangnv102/gobangvts/SherpaZipformer/pseudo_data"

CHECKPOINT_PATH = "/mnt/Disk800/thangnv102/gobangvts/checkpoints-v2/checkpoint-29.pt"

FREEZE_MODULES = ["encoder"]
IS_STREAMING = False
SAVE_DIR = "/mnt/Disk800/thangnv102/gobangvts/checkpoints-v2"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

class Pipeline(object):
    def __init__(self):
        self.cuts = load_manifest(TRAIN_CUTS)
        self.sampler = SimpleCutSampler(self.cuts, max_duration=MAX_DURATION, max_cuts=50, shuffle=True)
        self.feature_extractor = Fbank(FbankConfig(sampling_rate=16000, num_mel_bins=80, device="cpu"))
        self.input_strategy = OnTheFlyFeatures(self.feature_extractor)
        self.tester = Tester(folder_path=MATERIAL_DIR, 
                        checkpoint_path=CHECKPOINT_PATH, 
                        is_streaming=False,
                        decoding_method="greedy_search")
    def pad_to_max_time(self, x, x_lens, pad_value=PAD_LOG):
        B, T, F = x.shape
        T_needed = int(x_lens.max().item())
        if T == T_needed:
            return x, x_lens
        if T > T_needed:
            return x[:, :T_needed], x_lens
        out = x.new_full((B, T_needed, F), pad_value)
        out[:, :T] = x
        return out, x_lens

    def generate_augments(self):
        cut_transforms = [ExtraPadding(extra_seconds=5)]
        return cut_transforms

    def create_valid_loader(self):
        cuts = load_manifest(VALID_CUTS)  # <-- EAGER + đã TRIM
        print(len(cuts))  # OK

        dataset = K2SpeechRecognitionDataset(cuts)
        sampler = SimpleCutSampler(cuts, max_duration=600, max_cuts=50, shuffle=False)

        loader = DataLoader(
            dataset,
            sampler=sampler,        # dùng sampler= (đúng giao thức Lhotse)
            batch_size=None,
            num_workers=2,
            persistent_workers=False,
            pin_memory=True
        )

        print("DataLoader is ready!")
        return loader

    def test_one_epoch(self):
        cut_transforms = self.generate_augments()
        dataset = K2SpeechRecognitionDataset(
            input_strategy=self.input_strategy,
            cut_transforms=cut_transforms,
            return_cuts=False,     
        )
        test_loader = DataLoader(
                dataset,
                batch_size=None,            # IMPORTANT: sampler yields variable-size batches
                sampler=self.sampler,            # feeds batches of CutSet to dataset.collate
                num_workers=8,              # tune: 4–16 depending on CPU/IO
                prefetch_factor=2,          # how many batches per worker to prefetch
                persistent_workers=True,    # keep workers alive between epochs
                pin_memory=True,            # faster H2D copies later # use dataset’s collate for CutSet batches
            )
        print("[INFO] Start training epoch: Process data...")
        
        result = {"output" : [], "gt": []}
        batch_idx = 0
        for batch in tqdm(test_loader):
            x = batch["inputs"]
            x_lens = batch["supervisions"]["num_frames"]
            # Optionally move to GPU (model inputs only; features came from CPU workers)
            x, x_lens = self.pad_to_max_time(x, x_lens)
            batch["inputs"] = x
            batch["supervisions"]["num_frames"] = x_lens
            gt = batch["supervisions"]["text"]
            output = self.tester(batch)
            batch_idx += 1
            result["gt"].extend(gt)
            assert len(gt) == len(output), f"Fail at: {len(gt)} {len(output)} {gt} {output}"
            for o_ in output:
                result["output"].append(str(o_).replace("▁", " ").lower())
        all_wers = []
        outputs = result['output']
        gts = result['gt']

        for gt, hyp in tqdm(zip(gts, outputs), total=len(gts)):
            wer_score = jiwer.wer(
                str(gt),
                str(hyp),
                truth_transform=transformation,
                hypothesis_transform=transformation
            )
            all_wers.append(wer_score)
        mean_wer = sum(all_wers) / len(all_wers)
        results = pd.DataFrame.from_dict({"output" : outputs, "gt" : gts, "wer" : all_wers})
        results.to_csv("trial.csv", index=True, header=True, encoding="utf-8")
        print("[INFO] Mean WER: ", mean_wer)

if __name__ == "__main__":
    pipeline = Pipeline()
    # valid_dataloader = pipeline.create_valid_loader()
    pipeline.test_one_epoch()

