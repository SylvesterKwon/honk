"""Phase execution orchestration."""

import logging
import math
import uuid

import numpy as np
import pandas as pd

from .config import (
    ExpandedQueryBlock,
    MixedPhase,
    PausePhase,
    WorkloadConfig,
    WriteOnlyPhase,
)
from .dataset import DatasetCursor
from .filters import SelectivityFilterGenerator, UniformFilterGenerator
from .writer import TSVWriter

logger = logging.getLogger(__name__)


def _row_to_dict(row: pd.Series) -> dict:
    """Convert a DataFrame row to a plain dict."""
    return row.to_dict()


def _make_uuid(rng: np.random.Generator) -> uuid.UUID:
    """Generate a random UUID v4 using the seeded RNG."""
    return uuid.UUID(bytes=rng.bytes(16), version=4)


def _build_query_sampler(
    queries: list[ExpandedQueryBlock],
) -> tuple[np.ndarray, list[ExpandedQueryBlock]]:
    """Build a weight-probability array for query block sampling."""
    weights = np.array([q.weight for q in queries], dtype=float)
    probs = weights / weights.sum()
    return probs, queries


def _generate_read(
    block: ExpandedQueryBlock,
    uniform_gen: UniformFilterGenerator,
    selectivity_gen: SelectivityFilterGenerator,
    rng: np.random.Generator,
) -> dict:
    """Generate a single read filter dict from an expanded query block."""
    if block.strategy == "uniform":
        return uniform_gen.generate(block.num_filters)
    elif block.strategy == "selectivity":
        return selectivity_gen.generate(
            sigma=block.expected_selectivity,
            k=block.query_attr_num,
            attr_indices=block.query_attr_indices,
        )
    else:
        raise ValueError(f"Unknown strategy: {block.strategy}")


def execute_phases(
    config: WorkloadConfig,
    cursor: DatasetCursor,
    uniform_gen: UniformFilterGenerator,
    selectivity_gen: SelectivityFilterGenerator,
    writer: TSVWriter,
    rng: np.random.Generator,
) -> None:
    """Execute all phases in sequence."""
    for phase in config.phases:
        if isinstance(phase, WriteOnlyPhase):
            _run_write_only(phase, cursor, writer, rng)
        elif isinstance(phase, PausePhase):
            _run_pause(phase, writer)
        elif isinstance(phase, MixedPhase):
            _run_mixed(phase, cursor, uniform_gen, selectivity_gen, writer, rng)

    logger.info(
        "Generation complete: %d writes, %d reads, %d pauses",
        writer.writes, writer.reads, writer.pauses,
    )


def _run_write_only(
    phase: WriteOnlyPhase,
    cursor: DatasetCursor,
    writer: TSVWriter,
    rng: np.random.Generator,
) -> None:
    logger.info("Phase '%s': write_only, %d rows", phase.label, phase.rows)
    chunk = cursor.consume(phase.rows)
    for _, row in chunk.iterrows():
        pk = _make_uuid(rng)
        writer.write_row(pk, _row_to_dict(row))


def _run_pause(phase: PausePhase, writer: TSVWriter) -> None:
    logger.info("Phase '%s': pause, %s seconds", phase.label, phase.duration_seconds)
    writer.write_pause(phase.duration_seconds)


def _run_mixed(
    phase: MixedPhase,
    cursor: DatasetCursor,
    uniform_gen: UniformFilterGenerator,
    selectivity_gen: SelectivityFilterGenerator,
    writer: TSVWriter,
    rng: np.random.Generator,
) -> None:
    logger.info(
        "Phase '%s': mixed, %d rows, read_ratio=%.2f, %d query blocks",
        phase.label, phase.rows, phase.read_ratio, len(phase.queries),
    )

    probs, blocks = _build_query_sampler(phase.queries)
    block_indices = np.arange(len(blocks))

    int_part = int(phase.read_ratio)
    frac_part = phase.read_ratio - int_part

    chunk = cursor.consume(phase.rows)
    for _, row in chunk.iterrows():
        # Write
        pk = _make_uuid(rng)
        writer.write_row(pk, _row_to_dict(row))

        # Determine read count
        num_reads = int_part
        if frac_part > 0 and rng.random() < frac_part:
            num_reads += 1

        # Generate reads
        for _ in range(num_reads):
            idx = rng.choice(block_indices, p=probs)
            block = blocks[idx]
            filters = _generate_read(block, uniform_gen, selectivity_gen, rng)
            writer.write_read(filters)
