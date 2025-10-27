from inference import wavs_to_fbank_tensors
from tester import Tester
import os 

input_folder = ""
output_csv = ""


model = Tester(folder_path="./pseudo_data", checkpoint_path="./pretrained.pt", is_streaming=False, decoding_method="greedy_search", max_duration=300)

from tqdm import tqdm
device = "cuda:0"
files = os.listdir(input_folder)
# SAMPLE
answers = {}
for file in tqdm(files):
    input_paths = [input_folder + "/" + file]
    input_batch = wavs_to_fbank_tensors(input_paths, device=device)
    outputs = model(input_batch)
    answers[file] = outputs[0]['transcript']

import pandas as pd

d = pd.DataFrame.from_dict({"file" : list(answers.keys()), "transcript" : list(answers.values())})
d.to_csv(output_csv, index=True, encoding="utf-8")