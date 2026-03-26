# Honk

Multi-attribute filtering query benchmark workload generator using NYC Yellow Taxi data.

Generates realistic OLTP workloads (write, update, read) targeting KV storage systems.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Download Data

```bash
# Download January to March 2024 data
python download.py --year 2024 --month 1 2 3

# Download all data from 2022 to 2024 (range)
python download.py --year 2022-2024

# Skip confirmation prompt
python download.py --year 2024 --month 1 -y
```

### 2. Generate Workload

Write a JSON config file, then run `honk run` to generate the workload.

```bash
python -m honk.cli run workloads/sample.json --output_dir ./output --data_dir ./data
```

| Argument | Description | Default |
|----------|-------------|---------|
| `config` (positional) | Path to workload JSON config file | (required) |
| `--output_dir` | Output directory | (required) |
| `--data_dir` | Base directory for dataset parquet files | `./data` |

Output files:
- `workload.tsv` — Generated workload
- `workload.json` — Copy of config for reproducibility
- `honk.log` — Execution log

## Config Format

```json
{
  "dataset": ["yellow_tripdata_2025-01.parquet", ...],
  "seed": 42,
  "phases": [...]
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset` | `string[]` | (required) | List of parquet filenames |
| `seed` | `int` | `42` | Random seed for reproducibility |
| `phases` | `Phase[]` | `[]` | List of phase definitions |

### Phases

#### `write_only` — Pure writes + updates

```json
{
  "label": "preload",
  "type": "write_only",
  "rows": 10000,
  "update_ratio": 0.1
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rows` | `int` | (required) | Number of rows to write |
| `update_ratio` | `float` | `0.0` | Average number of updates per write |

#### `pause` — Wait

```json
{
  "label": "stabilize",
  "type": "pause",
  "duration_seconds": 60
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `duration_seconds` | `float` | (required) | Pause duration in seconds |

#### `mixed` — Interleaved writes, updates, and reads

```json
{
  "label": "mixed",
  "type": "mixed",
  "rows": 100000,
  "read_ratio": 10,
  "update_ratio": 0.1,
  "queries": [...]
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rows` | `int` | (required) | Number of rows to write |
| `read_ratio` | `float` | (required) | Average number of reads per write |
| `update_ratio` | `float` | `0.0` | Average number of updates per write |
| `queries` | `QueryBlock[]` | (required) | Query block definitions |

### Query Blocks

Defines read queries for `mixed` phases. Blocks are sampled probabilistically by `weight`.

All query blocks share these fields:

| Field | Type | Description |
|-------|------|-------------|
| `query_attr_num` | `int \| int[]` | Number of random attributes to filter on. Arrays are Cartesian-expanded |
| `weight` | `float` | Sampling weight (default: 1) |

#### `uniform` strategy — Uniform random filtering

```json
{
  "label": "baseline",
  "strategy": "uniform",
  "query_attr_num": 3,
  "weight": 1
}
```

Generates filters by uniformly sampling random column values.

#### `two_point` strategy — Data-driven range filtering

```json
{
  "label": "two_point_sweep",
  "strategy": "two_point",
  "query_attr_num": [1, 2, 3, 4],
  "weight": 1
}
```

Generates queries by sampling two actual records from the dataset.
- Categorical attributes: equality filter on one of the two sampled values
- Continuous/datetime attributes: range filter [min(val_a, val_b), max(val_a, val_b))

## Output Format

Tab-delimited TSV:

```
w	550e8400-...	{"VendorID":2,"fare_amount":12.50,...}
u	550e8400-...	{"VendorID":1,"fare_amount":8.75,...}
r	{"filters":[{"attr":"payment_type","op":"eq","value":1},{"attr":"fare_amount","op":"range","lo":5.0,"hi":25.0}]}
p	60
```

| Op | Format | Description |
|----|--------|-------------|
| `w` | `w\t<UUID>\t<JSON>` | Write (INSERT) — insert a new record |
| `u` | `u\t<UUID>\t<JSON>` | Update (FULL REPLACE) — replace an existing record at the given PK |
| `r` | `r\t<JSON>` | Read (SELECT) — query with filter conditions |
| `p` | `p\t<seconds>` | Pause — wait for the specified duration |

Each filter object has an explicit `op` field:

| Op | Fields | Description |
|----|--------|-------------|
| `eq` | `attr`, `value` | Equality match on a single value |
| `in` | `attr`, `values` | Equality match on any of the given values |
| `range` | `attr`, `lo`, `hi` | Range match `[lo, hi)` |

## Update Target Sampling

When `update_ratio > 0`, the generator must select previously written keys as update targets.
Rather than storing every key ever written (which would require O(N) memory), honk uses reservoir sampling with a fixed-capacity buffer (default 100,000 keys).

## Row Budget

Dataset rows consumed per phase: `rows + ceil(rows * update_ratio)`.

An error is raised if the total across all phases exceeds the available dataset rows.
