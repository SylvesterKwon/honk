"""Filter generation: uniform random and selectivity-targeted strategies."""

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

    def generate(self, num_filters: int) -> list[dict]:
        """Generate a filter with num_filters random attributes."""
        available = [c for c in self.columns if c.name in self.stats]
        k = min(num_filters, len(available))
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


class SelectivityFilterGenerator:
    """Generate filters targeting a specific overall selectivity."""

    def __init__(self, df: pd.DataFrame, columns: list[Column], rng: np.random.Generator):
        self.rng = rng
        self.columns = columns
        self.col_meta: dict[str, dict] = {}

        for col in columns:
            if col.name not in df.columns:
                continue
            series = df[col.name].dropna()
            if series.empty:
                continue

            if col.filter_type == FilterType.EQUALITY:
                freqs = series.value_counts(normalize=True)
                # Store as sorted list of (value, frequency) for closest-match lookup
                self.col_meta[col.name] = {
                    "type": "equality",
                    "values": freqs.index.tolist(),
                    "freqs": freqs.values,  # aligned with values
                }
            else:
                if col.dtype == "datetime":
                    sorted_vals = np.sort(series.values.astype("datetime64[ns]"))
                    self.col_meta[col.name] = {
                        "type": "range",
                        "dtype": "datetime",
                        "sorted": sorted_vals,
                    }
                else:
                    sorted_vals = np.sort(series.values.astype(float))
                    self.col_meta[col.name] = {
                        "type": "range",
                        "dtype": "numeric",
                        "sorted": sorted_vals,
                    }

    def generate(
        self,
        sigma: float,
        k: int | None = None,
        attr_indices: list[int] | None = None,
    ) -> list[dict]:
        """Generate a filter with expected selectivity sigma.

        Args:
            sigma: Target overall selectivity (fraction of rows matching).
            k: Number of attributes to use (randomly chosen). Mutually exclusive with attr_indices.
            attr_indices: Specific column indices (0-based) to use.
        """
        available = [c for c in self.columns if c.name in self.col_meta]

        if attr_indices is not None:
            chosen = [available[i] for i in attr_indices if i < len(available)]
        elif k is not None:
            num = min(k, len(available))
            chosen = list(self.rng.choice(available, size=num, replace=False))
        else:
            raise ValueError("Either k or attr_indices must be provided")

        if not chosen:
            return []

        # Per-attribute selectivity under independence assumption
        per_attr_sel = sigma ** (1.0 / len(chosen))

        filters = []
        for col in chosen:
            meta = self.col_meta[col.name]

            if meta["type"] == "equality":
                filters.append({"attr": col.name, "op": "eq", "value": self._generate_equality(meta, per_attr_sel)})
            elif meta["dtype"] == "datetime":
                lo, hi = self._generate_range_datetime(meta, per_attr_sel)
                filters.append({"attr": col.name, "op": "range", "lo": lo, "hi": hi})
            else:
                lo, hi = self._generate_range_numeric(meta, per_attr_sel)
                filters.append({"attr": col.name, "op": "range", "lo": lo, "hi": hi})

        return filters

    def _generate_equality(self, meta: dict, target_sel: float):
        """Pick the value whose frequency is closest to target selectivity."""
        freqs = meta["freqs"]
        values = meta["values"]
        # Find value with frequency closest to target_sel
        idx = int(np.argmin(np.abs(freqs - target_sel)))
        return _to_python(values[idx])

    def _generate_range_numeric(self, meta: dict, target_sel: float) -> tuple:
        """Generate a numeric range [lo, hi) covering ~target_sel fraction of data."""
        sorted_vals = meta["sorted"]
        n = len(sorted_vals)

        # Width in quantile space
        width = max(min(target_sel, 1.0), 1.0 / n)
        # Random start position
        max_start = max(1.0 - width, 0.0)
        start = self.rng.uniform(0, max_start) if max_start > 0 else 0.0

        lo_idx = int(start * (n - 1))
        hi_idx = int((start + width) * (n - 1))
        hi_idx = min(hi_idx, n - 1)

        lo = round(float(sorted_vals[lo_idx]), 2)
        hi = round(float(sorted_vals[hi_idx]), 2)
        return (lo, hi)

    def _generate_range_datetime(self, meta: dict, target_sel: float) -> tuple:
        """Generate a datetime range [lo, hi) covering ~target_sel fraction of data."""
        sorted_vals = meta["sorted"]
        n = len(sorted_vals)

        width = max(min(target_sel, 1.0), 1.0 / n)
        max_start = max(1.0 - width, 0.0)
        start = self.rng.uniform(0, max_start) if max_start > 0 else 0.0

        lo_idx = int(start * (n - 1))
        hi_idx = int((start + width) * (n - 1))
        hi_idx = min(hi_idx, n - 1)

        lo = pd.Timestamp(sorted_vals[lo_idx]).isoformat()
        hi = pd.Timestamp(sorted_vals[hi_idx]).isoformat()
        return (lo, hi)
