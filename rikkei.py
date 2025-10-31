import os, shutil
import pandas as pd 
from tqdm import tqdm
train_path = ""

total_saving_dir = ""
audio_saving_dir = total_saving_dir + "/audios"
THRESHOLD = 5

os.makedirs(total_saving_dir, exist_ok=True)
os.makedirs(audio_saving_dir, exist_ok=True)
df = pd.read_csv(train_path, encoding="utf-8")
data = df['path\ttranscript\tduration'].tolist()

transcript_mapping = {}
for element in tqdm(data):
    path, transcript, duration = element.split("\t")
    duration = float(duration)
    if os.path.exists(path) and "RK" in path and duration >= THRESHOLD:
        transcript_mapping[path] = transcript

new_paths_mapping = {}
audio_index = 0
for audio_path in tqdm(transcript_mapping.keys()):
    new_audio_name = f"ID_{audio_index}.wav"
    new_audio_path = audio_saving_dir + "/" + new_audio_name
    shutil.copy(audio_path, new_audio_path)
    new_paths_mapping[new_audio_path] = audio_path

with open(total_saving_dir + "/transcript.txt", "w", encoding="utf-8") as f:
    for new_audio_path in new_paths_mapping.keys():
        content = transcript_mapping[new_paths_mapping[new_audio_path]]
        line = new_audio_path + "\t" + content + "\n"
        f.write(line)