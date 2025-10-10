import os
import shutil

txt_path = ""
data_dir = ""
new_dir = ""
os.mkdir(new_dir)

with open(txt_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines:
    path, transcript = line.split("\t")
    shutil.copy(data_dir + "/" + path, new_dir + "/" + path)

shutil.copy(txt_path, new_dir + "/train_5_30.txt")