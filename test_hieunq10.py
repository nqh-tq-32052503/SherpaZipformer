from lhotse import load_manifest
from lhotse.dataset import K2SpeechRecognitionDataset
from lhotse.dataset.sampling import SimpleCutSampler
from prepare_data import DataPreparation
from torch.utils.data import DataLoader
from tester import Tester
import os
from tqdm import tqdm
import pandas as pd

TEMP_DIR = "./temp"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

def test_checkpoint(valid_cuts, checkpoint_path, material_path, save_path, prefix_path=None, save_pandas=True):
    assert os.path.exists(valid_cuts) and os.path.exists(checkpoint_path) and os.path.exists(material_path), "Paths not found"

    cuts = load_manifest(valid_cuts)  # <-- EAGER + đã TRIM
    if prefix_path is not None:
        cuts = cuts.with_features_path_prefix(prefix_path)
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
    tester = Tester(folder_path=material_path, 
                    checkpoint_path=checkpoint_path, 
                    is_streaming=False,
                    decoding_method="greedy_search")
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

    if save_pandas:
        df = pd.DataFrame(result)
        df.to_csv(save_path, index=False, encoding="utf-8")
    else:
        return result["output"]

def prepare_inference_data(audio_file):
    assert os.path.exists(audio_file), "[ERROR] File not found: {0}".format(audio_file)
    list_audios = [audio_file]
    list_transcripts = ["Test"]
    DataPreparation(list_audios, list_transcripts, output_dir=TEMP_DIR)
    print("[INFO] Extracting FBank done")

def inference_one_file(audio_file):
    prepare_inference_data(audio_file)
    result = test_checkpoint(valid_cuts=TEMP_DIR + "/cuts_with_feats_trim.jsonl.gz",
                    checkpoint_path="./pretrained.pt",
                    material_path="./pseudo_data",
                    save_path=None,
                    save_pandas=False)
    return result