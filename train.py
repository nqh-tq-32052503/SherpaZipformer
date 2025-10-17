from pipeline import Pipeline
import os
from test_hieunq10 import test_checkpoint
from evaluate import compute_wer


VALID_CUTS = os.environ.get("VALID_CUTS")
PREFIX_PATH = os.environ.get("PREFIX_PATH")
pipeline = Pipeline()
num_epochs = int(os.environ.get("NUM_EPOCHS"))
for epoch in range(num_epochs):
    print(f"[INFO] Starting epoch {epoch}")
    checkpoint_path = pipeline.train_one_epoch(epoch)
    print(f"[INFO] Finished epoch {epoch}")
    result = test_checkpoint(valid_cuts=VALID_CUTS, checkpoint_path=checkpoint_path, material_path="./pseudo_data", prefix_path=PREFIX_PATH, save_path=None, save_pandas=False)
    wer_score = compute_wer(result["gt"], result["output"], return_scalar=True, is_sherpa_format=True)
    print(f"[INFO] WER at checkpoint {epoch}: {wer_score}")