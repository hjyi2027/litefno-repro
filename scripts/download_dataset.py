from pathlib import Path
import argparse

from litefno.download import download_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    download_from_config(args.config)


if __name__ == "__main__":
    main()
