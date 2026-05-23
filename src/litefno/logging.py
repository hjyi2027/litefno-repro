from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


class MetricsLogger:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.history = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        record = {"step": step, **metrics}
        self.history.append(record)
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
