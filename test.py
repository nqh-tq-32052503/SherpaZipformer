from lhotse import load_manifest
from lhotse.dataset import K2SpeechRecognitionDataset
from lhotse.dataset.sampling import SimpleCutSampler

from torch.utils.data import DataLoader
from tester import Tester
import os
from tqdm import tqdm

cuts = load_manifest("/mnt/Disk800/thangnv102/gobangvts/processed_validation_data/cuts_with_feats_trim.jsonl.gz")  # <-- EAGER + đã TRIM
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
tester = Tester(folder_path="/mnt/Disk800/thangnv102/gobangvts/SherpaZipformer/pseudo_data", 
                checkpoint_path="/mnt/Disk800/thangnv102/gobangvts/checkpoints/checkpoint-29.pt", 
                is_streaming=False,
                decoding_method="fast_beam_search_one_best")
outliers = []
result = {"output" : [], "gt": []}
batch_index = 0
try:
    for batch in tqdm(loader):
        batch_index += 1
        gt = batch["supervisions"]["text"]
        contain_outlier = False
        for o in outliers:
            if o in gt:
                contain_outlier = True
        if not contain_outlier:
            output = tester(batch)
            result["gt"].extend(gt)
            for o_ in output:
                result["output"].append(o_.replace("▁", " ").lower())
except Exception as e:
    print(f"Error occurred at batch index {batch_index}: {e}")
    print(batch["supervisions"]["text"])
    batch_index += 1

import pandas as pd
df = pd.DataFrame(result)
df.to_csv("/mnt/Disk800/thangnv102/gobangvts/outputs/finetune_v1_beamsearch.csv", index=False, encoding="utf-8")
