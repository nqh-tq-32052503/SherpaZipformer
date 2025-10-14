from pipeline import Pipeline
import os

pipeline = Pipeline()
num_epochs = int(os.environ.get("NUM_EPOCHS"))
for epoch in range(num_epochs):
    print(f"[INFO] Starting epoch {epoch}")
    pipeline.train_one_epoch(epoch)
    print(f"[INFO] Finished epoch {epoch}")