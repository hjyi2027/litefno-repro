from __future__ import annotations

import argparse
from pathlib import Path
from .download import download_from_config
from .preprocess import preprocess_from_config
from .train import run_evaluation, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LiteFNO reproduction scaffolding")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download a dataset split")
    download_parser.add_argument("--config", required=True, type=Path)

    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess dataset splits")
    preprocess_parser.add_argument("--config", required=True, type=Path)

    train_parser = subparsers.add_parser("train", help="Run training and evaluation")
    train_parser.add_argument("--config", required=True, type=Path)
    train_parser.add_argument("--set", action="append", default=[], help="Override config key=value")

    test_parser = subparsers.add_parser("test", help="Evaluate a checkpoint on a split")
    test_parser.add_argument("--config", required=True, type=Path)
    test_parser.add_argument("--checkpoint", required=True, type=Path)
    test_parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    test_parser.add_argument("--set", action="append", default=[], help="Override config key=value")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "download":
        download_from_config(args.config)
    elif args.command == "preprocess":
        preprocess_from_config(args.config)
    elif args.command == "train":
        run_training(args.config, overrides=args.set)
    elif args.command == "test":
        run_evaluation(args.config, args.checkpoint, split=args.split, overrides=args.set)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
