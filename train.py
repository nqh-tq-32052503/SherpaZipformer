import os 

os.environ["MAX_DURATION"] = "600"

from trainer import Trainer
from lhotse import load_manifest
from lhotse.dataset import K2SpeechRecognitionDataset
from lhotse.dataset.sampling import SimpleCutSampler

from torch.utils.data import DataLoader

cuts = load_manifest("/mnt/Disk800/thangnv102/gobangvts/processed_training_data/cuts_with_feats_trim.jsonl.gz")  # <-- EAGER + đã TRIM
print(len(cuts))  # OK

dataset = K2SpeechRecognitionDataset(cuts)
sampler = SimpleCutSampler(cuts, max_duration=int(os.environ.get("MAX_DURATION")), shuffle=True)

train_dataloader = DataLoader(
    dataset,
    sampler=sampler,        # dùng sampler= (đúng giao thức Lhotse)
    batch_size=None,
    num_workers=16,
    persistent_workers=False,
    pin_memory=True
)
print("[INFO] DataLoader is ready!")
trainer_obj = Trainer(folder_path="/mnt/Disk800/thangnv102/gobangvts/SherpaZipformer/pseudo_data", 
                      checkpoint_path="/mnt/Disk800/thangnv102/gobangvts/pretrained.pt", 
                      freeze_modules=["encoder"], is_streaming=False)
print("[INFO] Trainer is ready!")

trainer_obj.train(train_dataloader, num_epochs=30, checkpoint_folder="/mnt/Disk800/thangnv102/gobangvts/checkpoints")

# nohup python train.py > training_log_v1_5.log 2>&1 &