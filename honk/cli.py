"""CLI entry point for Honk workload generator."""

import argparse
import logging
import os
import shutil
import sys
import time

import numpy as np

from .config import HonkConfigError, load_config, resolve_rows, validate_row_budget
from .dataset import DatasetCursor
from .filters import TwoPointFilterGenerator, UniformFilterGenerator
from .phases import execute_phases
from .schema import ALL_COLUMNS
from .writer import TSVWriter

logger = logging.getLogger("honk")


def _setup_logging(log_path: str) -> None:
    """Configure logging to stderr and a file."""
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)

    root = logging.getLogger("honk")
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stderr_handler)


def cmd_run(args: argparse.Namespace) -> None:
    """Execute the 'run' subcommand."""
    config_path = args.config
    output_dir = args.output_dir
    data_dir = args.data_dir

    # Load config
    try:
        config = load_config(config_path)
    except HonkConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_path = os.path.join(output_dir, "honk.log")
    _setup_logging(log_path)

    # Copy config for reproducibility
    config_copy_path = os.path.join(output_dir, "workload.json")
    shutil.copy2(config_path, config_copy_path)
    logger.info("Config copied to %s", config_copy_path)

    # Resolve dataset file paths
    file_paths = []
    for fname in config.dataset:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            logger.error("Dataset file not found: %s", path)
            sys.exit(1)
        file_paths.append(path)

    # Load dataset
    logger.info("Loading %d dataset file(s)...", len(file_paths))
    t0 = time.time()
    cursor = DatasetCursor(file_paths)
    logger.info("Loaded %d rows in %.1fs", cursor.total_rows, time.time() - t0)

    # Resolve unspecified rows to full dataset size
    resolve_rows(config, cursor.total_rows)

    # Validate row budget
    try:
        validate_row_budget(config, cursor.total_rows)
    except HonkConfigError as e:
        logger.error(str(e))
        sys.exit(1)

    # Build filter generators
    rng = np.random.default_rng(config.seed)
    columns = ALL_COLUMNS

    logger.info("Learning data distributions...")
    df = cursor.dataframe
    uniform_gen = UniformFilterGenerator(df, columns, rng)
    two_point_gen = TwoPointFilterGenerator(df, columns, rng)

    # Execute phases
    tsv_path = os.path.join(output_dir, "workload.tsv")
    logger.info("Generating workload → %s", tsv_path)
    t0 = time.time()

    with TSVWriter(tsv_path) as writer:
        execute_phases(config, cursor, uniform_gen, two_point_gen, writer, rng, df)

    elapsed = time.time() - t0
    logger.info("Done in %.1fs", elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="honk",
        description="Honk — KV storage benchmark workload generator",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Generate a workload from a JSON config")
    run_parser.add_argument("config", help="Path to workload JSON config file")
    run_parser.add_argument("--output_dir", required=True, help="Output directory")
    run_parser.add_argument("--data_dir", default="./data", help="Base directory for dataset files (default: ./data)")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        cmd_run(args)
