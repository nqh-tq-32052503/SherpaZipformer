import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Sherpa Training Script")
    parser.add_argument("--train_path", type=str, default="train.csv", help="Path to training data")
    parser.add_argument("--valid_path", type=str, default="valid.csv", help="Path to validation data")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(f"Training data path: {args.train_path}")
    print(f"Validation data path: {args.valid_path}")
    