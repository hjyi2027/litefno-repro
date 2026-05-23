from pathlib import Path
import argparse

from litefno.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[], help="Override config key=value")
    args = parser.parse_args()
    run_training(args.config, overrides=args.set)


if __name__ == "__main__":
    main()
