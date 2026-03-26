# Honk

Multi-attribute filtering query benchmark workload generator using NYC Yellow Taxi data.

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

Before downloading, the file list and total size are displayed with a `[y/N]` confirmation.

### 2. Generate Workload

```bash
# Generate with default settings
python generate.py --input-dir ./data --output workload.tsv

# Custom settings
python generate.py \
  --input-dir ./data \
  --output workload.tsv \
  --rw-ratio 5 \
  --num-filters 2 \
  --filter-columns fare_amount trip_distance payment_type \
  --limit 1000 \
  --seed 123
```

### Output Format

Tab-delimited TSV file:

```
TIMESTAMP	COMMAND	VALUE
0	w	{"tpep_pickup_datetime":"2024-01-01T00:57:55","passenger_count":1,...}
0	r	{"filters":{"payment_type":1,"fare_amount":[5.0,25.0],"trip_distance":[0.5,3.0]}}
```

| Field | Description |
|-------|-------------|
| `TIMESTAMP` | Millisecond offset from the first record |
| `w` | Write — original record JSON (for INSERT) |
| `r` | Read — multi-attribute filter condition JSON (for SELECT/FILTER) |

## Options

### download.py

| Option | Description | Default |
|--------|-------------|---------|
| `--year` | Year(s) (range: `2022-2024`, individual: `2022 2023`) | (required) |
| `--month` | Month(s) | 1-12 |
| `--output-dir` | Save directory | `./data` |
| `-y` | Skip confirmation | `false` |

### generate.py

| Option | Description | Default |
|--------|-------------|---------|
| `--input-dir` | Parquet directory | `./data` |
| `--output` | Output file | `./workload.tsv` |
| `--rw-ratio` | Read:write ratio | `10` |
| `--num-filters` | Number of filter attributes per read query | `3` |
| `--filter-columns` | Filter columns to use | all |
| `--limit` | Max number of instructions | unlimited |
| `--seed` | Random seed | `42` |

## Filterable Columns
TODO: Add link to NYC Taxi dataset documentation
