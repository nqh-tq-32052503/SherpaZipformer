import os 

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
    ExtraPadding, PerturbTempo, PerturbVolume, ReverbWithImpulseResponse, PerturbSpeed
)
from lhotse.features import Fbank, FbankConfig
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import random
from trainer import Trainer
import time 

PAD_LOG = float(torch.log(torch.tensor(1e-10)))
TRAIN_CUTS = os.environ.get("TRAIN_CUTS")
VALID_CUTS = os.environ.get("VALID_CUTS")
MATERIAL_DIR = os.environ.get("MATERIAL_DIR")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH")
FREEZE_MODULES = [os.environ.get("FREEZE_MODULES")]
IS_STREAMING = False
SAVE_DIR = os.environ.get("SAVE_DIR")
MAX_DURATION = int(os.environ.get("MAX_DURATION"))
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

class Pipeline(object):
    def __init__(self):
        self.cuts = load_manifest(TRAIN_CUTS)
        self.sampler = SimpleCutSampler(self.cuts, max_duration=MAX_DURATION, max_cuts=50, shuffle=True)
        self.feature_extractor = Fbank(FbankConfig(sampling_rate=16000, num_mel_bins=80, device="cpu"))
        self.input_strategy = OnTheFlyFeatures(self.feature_extractor)
        self.trainer = Trainer(folder_path=MATERIAL_DIR, 
                               checkpoint_path=CHECKPOINT_PATH, 
                               freeze_modules=FREEZE_MODULES,
                               is_streaming=IS_STREAMING,
                               max_duration=MAX_DURATION)
        
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
        cut_transforms = []
        if random.random() < 0.5:
            cut_transforms.append(ExtraPadding(extra_seconds=5))

            p1 = random.random()
            if p1 < 0.3:
                cut_transforms.append(PerturbTempo(factors=(1.4, 1.6), p=0.99999999))
            elif p1 < 0.6:
                cut_transforms.append(PerturbSpeed(factors=(1.1, 1.2), p=0.999))

            p2 = random.random()
            if p2 < 0.3:
                cut_transforms.append(PerturbVolume(p=0.999999, scale_low=0.1, scale_high=0.5))
            elif p2 < 0.6:
                cut_transforms.append(PerturbVolume(p=0.999999, scale_low=1.5, scale_high=2.0))

            if random.random() < 0.3:
                cut_transforms.append(ReverbWithImpulseResponse(p=0.99999))
                
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

    def create_train_loader(self):
        cut_transforms = self.generate_augments()
        dataset = K2SpeechRecognitionDataset(
            input_strategy=self.input_strategy,
            cut_transforms=cut_transforms,
            return_cuts=False, 
        )
        train_loader = DataLoader(
                dataset,
                batch_size=None,            # IMPORTANT: sampler yields variable-size batches
                sampler=self.sampler,            # feeds batches of CutSet to dataset.collate
                num_workers=8,              # tune: 4–16 depending on CPU/IO
                prefetch_factor=2,          # how many batches per worker to prefetch
                persistent_workers=True,    # keep workers alive between epochs
                pin_memory=True,            # faster H2D copies later # use dataset’s collate for CutSet batches
            )
        print("[INFO] Start training epoch: Process data...")
        batch = next(iter(train_loader))

    def train_one_epoch(self, epoch):
        cut_transforms = self.generate_augments()
        dataset = K2SpeechRecognitionDataset(
            input_strategy=self.input_strategy,
            cut_transforms=cut_transforms,
            return_cuts=False,     
        )
        train_loader = DataLoader(
                dataset,
                batch_size=None,            # IMPORTANT: sampler yields variable-size batches
                sampler=self.sampler,            # feeds batches of CutSet to dataset.collate
                num_workers=8,              # tune: 4–16 depending on CPU/IO
                prefetch_factor=2,          # how many batches per worker to prefetch
                persistent_workers=True,    # keep workers alive between epochs
                pin_memory=True,            # faster H2D copies later # use dataset’s collate for CutSet batches
            )
        print("[INFO] Start training epoch: Process data...")
        batch_idx = 0
        for batch in tqdm(train_loader):
            x = batch["inputs"]
            x_lens = batch["supervisions"]["num_frames"]
            # Optionally move to GPU (model inputs only; features came from CPU workers)
            x, x_lens = self.pad_to_max_time(x, x_lens)
            batch["inputs"] = x
            batch["supervisions"]["num_frames"] = x_lens
            self.trainer.train_one_batch(batch_idx, batch)
            batch_idx += 1
        print("[INFO] Finished processing data. Start saving checkpoint...")
        self.trainer.save(SAVE_DIR, epoch)

if __name__ == "__main__":
    pipeline = Pipeline()
    # valid_dataloader = pipeline.create_valid_loader()
    for epoch in range(1):
        print(f"[INFO] Starting epoch {epoch}")
        pipeline.train_one_epoch(epoch)
        print(f"[INFO] Finished epoch {epoch}")
        # all_wers = []
        # for batch in tqdm(valid_dataloader):
        #     wers = pipeline.trainer.test(batch)
        #     all_wers.extend(wers)
        # avg_wer = sum(all_wers) / len(all_wers)
        # print(f"[INFO] Average WER at epoch {epoch}: {avg_wer:.2f}")

