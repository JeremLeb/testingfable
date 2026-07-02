"""Minimal CSV metric logger with running averages between flushes."""
from __future__ import annotations

import csv
import pathlib
from collections import defaultdict


class CSVLogger:
    def __init__(self, path: str):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._buf = defaultdict(list)
        self._rows: list[dict] = []
        self._keys: list[str] = ["step"]

    def add(self, **kv):
        for k, v in kv.items():
            self._buf[k].append(float(v))

    def flush(self, step: int) -> dict:
        row = {"step": step}
        for k, vals in self._buf.items():
            row[k] = sum(vals) / len(vals) if vals else 0.0
        self._buf.clear()
        self._rows.append(row)
        for k in row:
            if k not in self._keys:
                self._keys.append(k)
        self._rewrite()  # keys can grow over time; rewrite keeps CSV valid
        return row

    def _rewrite(self):
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._keys)
            writer.writeheader()
            for row in self._rows:
                writer.writerow({k: row.get(k, "") for k in self._keys})

    def close(self):
        self._rewrite()
