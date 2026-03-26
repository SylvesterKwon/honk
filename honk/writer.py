"""TSV workload output writer."""

import json
import uuid
from typing import IO, Any

import numpy as np
import pandas as pd


def _json_default(obj: Any) -> Any:
    """JSON serialization helper for pandas/numpy types."""
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if pd.isna(obj):
        return None
    raise TypeError(f"Not serializable: {type(obj)}")


class TSVWriter:
    """Writes workload operations to a TSV file."""

    def __init__(self, path: str):
        self._f: IO[str] = open(path, "w")
        self.writes = 0
        self.reads = 0
        self.updates = 0
        self.pauses = 0

    def write_row(self, pk: uuid.UUID, record: dict) -> None:
        record_json = json.dumps(record, default=_json_default, ensure_ascii=False)
        self._f.write(f"w\t{pk}\t{record_json}\n")
        self.writes += 1

    def write_update(self, pk: uuid.UUID, record: dict) -> None:
        record_json = json.dumps(record, default=_json_default, ensure_ascii=False)
        self._f.write(f"u\t{pk}\t{record_json}\n")
        self.updates += 1

    def write_read(self, filters: dict) -> None:
        filter_json = json.dumps({"filters": filters}, ensure_ascii=False)
        self._f.write(f"r\t{filter_json}\n")
        self.reads += 1

    def write_pause(self, seconds: float) -> None:
        self._f.write(f"p\t{seconds}\n")
        self.pauses += 1

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
