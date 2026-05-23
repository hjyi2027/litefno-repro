from pathlib import Path
import argparse

from litefno.preprocess import preprocess_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    preprocess_from_config(args.config)


if __name__ == "__main__":
    main()
