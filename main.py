import k2
import lhotse
import os
from tqdm import tqdm
from pathlib import Path
from lhotse import load_manifest_lazy
from lhotse.dataset.speech_recognition import K2SpeechRecognitionDataset
from lhotse.dataset import DynamicBucketingSampler, SimpleCutSampler
from torch.utils.data import DataLoader
import argparse

from trainer import Trainer
from tester import Tester
from evaluate import compute_wer

MAX_DURATION = 300
os.environ["MAX_DURATION"] = str(MAX_DURATION)

def get_parser():
    parser = argparse.ArgumentParser(description="Sherpa Training Script")
    parser.add_argument("--train_path", type=str, help="Path to training data")
    parser.add_argument("--valid_path", type=str, help="Path to validation data")
    parser.add_argument("--code_path", type=str, default="/pseudo_data", help="Path to pseudo data")
    parser.add_argument("--is_streaming", type=bool, help="Streaming mode")
    parser.add_argument("--checkpoint_path", type=str, help="Path to checkpoint")
    parser.add_argument("--freeze_modules", type=str, help="List of modules to freeze, split by comma")
    parser.add_argument("--save_path", type=str, help="Path to save checkpoints")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs")
    return parser

def create_dataloader(cuts_with_feats_path):
    cuts = load_manifest_lazy(cuts_with_feats_path)
    # cuts = cuts.with_features_path_prefix(
    #     "/kaggle/working/training_data"
    # )
    dataset = K2SpeechRecognitionDataset(cuts)
    sampler = SimpleCutSampler(
                    cuts,
                    max_duration=MAX_DURATION,
                    shuffle=True,
                )
    loader = DataLoader(dataset, sampler=sampler, num_workers=2, batch_size=None, persistent_workers=False)
    sample_batch = next(iter(loader))
    return DataLoader(dataset, sampler=sampler, num_workers=2, batch_size=None, persistent_workers=False)

def main():
    parser = get_parser()
    args = parser.parse_args()
    print(f"Training data path: {args.train_path}")
    print(f"Validation data path: {args.valid_path}")
    train_loader = create_dataloader(args.train_path)
    valid_loader = create_dataloader(args.valid_path)
    trainer = Trainer(folder_path=args.code_path, checkpoint_path=args.checkpoint_path, freeze_modules=args.freeze_modules.split(","), is_streaming=bool(args.is_streaming))
    tester = Tester(folder_path=args.code_path, checkpoint_path=args.checkpoint_path, is_streaming=bool(args.is_streaming))
    trainer.train_with_test(train_loader, valid_loader, num_epochs=int(args.num_epochs), tester=tester, checkpoint_folder=args.save_path)

if __name__ == "__main__":
    main()