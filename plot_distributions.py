#!/usr/bin/env python3
"""Plot distribution of every column in Yellow Taxi parquet data."""

import glob
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = "data"
SCHEMA_PATH = "schema.json"
SAMPLE_FRAC = 0.05  # 5% sample for speed


def load_sample() -> pd.DataFrame:
    files = sorted(glob.glob(f"{DATA_DIR}/yellow_tripdata_2025-*.parquet"))
    if not files:
        print("No 2025 parquet files found in data/")
        sys.exit(1)
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        frames.append(df.sample(frac=SAMPLE_FRAC, random_state=42))
    return pd.concat(frames, ignore_index=True)


def load_dict() -> dict:
    with open(SCHEMA_PATH) as f:
        dd = json.load(f)
    return {field["name"]: field for field in dd["fields"]}


def plot_all(df: pd.DataFrame, data_dict: dict):
    cols = [c for c in df.columns if c in data_dict]
    n = len(cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 4.5 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        info = data_dict[col]
        desc = info.get("description", "")
        codes = info.get("codes")
        dtype = info.get("type")

        if codes or dtype == "str":
            # Categorical: bar chart
            vc = df[col].dropna().value_counts().sort_index()
            labels = [codes.get(str(int(k) if isinstance(k, float) else k), str(k))
                      if codes else str(k) for k in vc.index]
            # Truncate long labels
            labels = [l[:20] for l in labels]
            bars = ax.bar(range(len(vc)), vc.values, color="steelblue", edgecolor="white")
            ax.set_xticks(range(len(vc)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Count")
        elif dtype == "datetime":
            # Datetime: histogram by day
            series = pd.to_datetime(df[col], errors="coerce").dropna()
            ax.hist(series.astype(np.int64) // 10**9, bins=60, color="steelblue", edgecolor="white")
            # Relabel x-axis with dates
            ticks = ax.get_xticks()
            ax.set_xticklabels(
                [pd.Timestamp(t, unit="s").strftime("%m-%d") for t in ticks],
                rotation=45, ha="right", fontsize=7,
            )
            ax.set_ylabel("Count")
        else:
            # Numeric: histogram with outlier clipping (1st-99th percentile)
            series = df[col].dropna()
            lo, hi = series.quantile(0.01), series.quantile(0.99)
            clipped = series[(series >= lo) & (series <= hi)]
            ax.hist(clipped, bins=60, color="steelblue", edgecolor="white")
            ax.set_ylabel("Count")

        title = f"{col}\n({desc[:60]})" if desc else col
        ax.set_title(title, fontsize=9, pad=6)
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 3))

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("NYC Yellow Taxi 2025 — Column Distributions (5% sample)", fontsize=15, y=1.01)
    fig.tight_layout()
    out = "distributions_2025.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    print("Loading 5% sample from all 2025 files...")
    df = load_sample()
    print(f"Sample size: {len(df):,} rows")
    data_dict = load_dict()
    plot_all(df, data_dict)
