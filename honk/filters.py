"""Filter generation: uniform random, selectivity-targeted, and two-point strategies."""

import numpy as np
import pandas as pd

from .schema import Column, FilterType


def _to_python(val):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    if isinstance(val, np.datetime64):
        return pd.Timestamp(val).isoformat()
    return val


class UniformFilterGenerator:
    """Generate random filters by uniformly sampling column values."""

    def __init__(self, df: pd.DataFrame, columns: list[Column], rng: np.random.Generator):
        self.rng = rng
        self.columns = columns
        self.stats: dict[str, dict] = {}

        for col in columns:
            if col.name not in df.columns:
                continue
            series = df[col.name].dropna()
            if series.empty:
                continue

            if col.filter_type == FilterType.EQUALITY:
                self.stats[col.name] = {
                    "type": "equality",
                    "values": series.unique(),
                }
            else:
                if col.dtype == "datetime":
                    self.stats[col.name] = {
                        "type": "range",
                        "dtype": "datetime",
                        "min": series.min(),
                        "max": series.max(),
                    }
                else:
                    self.stats[col.name] = {
                        "type": "range",
                        "dtype": "numeric",
                        "min": float(series.min()),
                        "max": float(series.max()),
                    }

    def generate(self, query_attr_num: int) -> list[dict]:
        """Generate a filter with query_attr_num random attributes."""
        available = [c for c in self.columns if c.name in self.stats]
        k = min(query_attr_num, len(available))
        chosen = self.rng.choice(available, size=k, replace=False)

        filters = []
        for col in chosen:
            stat = self.stats[col.name]
            if stat["type"] == "equality":
                value = self.rng.choice(stat["values"])
                filters.append({"attr": col.name, "op": "eq", "value": _to_python(value)})
            elif stat["dtype"] == "datetime":
                min_ts = stat["min"].timestamp()
                max_ts = stat["max"].timestamp()
                a, b = sorted(self.rng.uniform(min_ts, max_ts, size=2))
                filters.append({
                    "attr": col.name, "op": "range",
                    "lo": pd.Timestamp(a, unit="s").isoformat(),
                    "hi": pd.Timestamp(b, unit="s").isoformat(),
                })
            else:
                a, b = sorted(self.rng.uniform(stat["min"], stat["max"], size=2))
                filters.append({
                    "attr": col.name, "op": "range",
                    "lo": round(a, 2), "hi": round(b, 2),
                })

        return filters


class TwoPointFilterGenerator:
    """Generate filters by sampling two actual data points and using their values as range bounds."""

    def __init__(self, df: pd.DataFrame, columns: list[Column], rng: np.random.Generator):
        self.rng = rng
        self.columns = columns
        self.df = df
        self.available_columns = [c for c in columns if c.name in df.columns]

    def generate(self, query_attr_num: int) -> list[dict]:
        """Pick k random attributes, sample 2 records, build filters from their values."""
        k = min(query_attr_num, len(self.available_columns))
        chosen_cols = list(self.rng.choice(self.available_columns, size=k, replace=False))

        indices = self.rng.choice(len(self.df), size=2, replace=False)
        row_a = self.df.iloc[indices[0]]
        row_b = self.df.iloc[indices[1]]

        filters = []
        for col in chosen_cols:
            val_a = row_a[col.name]
            val_b = row_b[col.name]

            if pd.isna(val_a) and pd.isna(val_b):
                continue
            if pd.isna(val_a):
                val_a = val_b
            if pd.isna(val_b):
                val_b = val_a

            if col.filter_type == FilterType.EQUALITY:
                chosen_val = val_a if self.rng.random() < 0.5 else val_b
                filters.append({
                    "attr": col.name,
                    "op": "eq",
                    "value": _to_python(chosen_val),
                })
            else:
                lo, hi = (val_a, val_b) if val_a <= val_b else (val_b, val_a)

                if col.dtype == "datetime":
                    lo_str = pd.Timestamp(lo).isoformat()
                    hi_str = pd.Timestamp(hi).isoformat()
                    if lo == hi:
                        hi_str = (pd.Timestamp(hi) + pd.Timedelta(seconds=1)).isoformat()
                    filters.append({
                        "attr": col.name,
                        "op": "range",
                        "lo": lo_str,
                        "hi": hi_str,
                    })
                else:
                    lo_f = round(float(lo), 2)
                    hi_f = round(float(hi), 2)
                    if lo_f == hi_f:
                        hi_f = lo_f + 0.01
                    filters.append({
                        "attr": col.name,
                        "op": "range",
                        "lo": lo_f,
                        "hi": hi_f,
                    })

        return filters
