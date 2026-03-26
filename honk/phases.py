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


class PKReservoir:
    """Fixed-size reservoir maintaining uniform random sampling over all appended keys.

    Uses Algorithm R (Vitter, 1985) to keep at most *capacity* keys in memory
    while guaranteeing that every key ever appended has equal probability of
    being selected by :meth:`random_choice`.  This bounds memory usage to
    O(capacity) regardless of total write count.
    """

    _DEFAULT_CAPACITY = 100_000

    def __init__(self, capacity: int = _DEFAULT_CAPACITY) -> None:
        self._buf: list[uuid.UUID] = []
        self._capacity = capacity
        self._total = 0

    def append(self, pk: uuid.UUID, rng: np.random.Generator) -> None:
        self._total += 1
        if len(self._buf) < self._capacity:
            self._buf.append(pk)
        else:
            j = int(rng.integers(self._total))
            if j < self._capacity:
                self._buf[j] = pk

    def random_choice(self, rng: np.random.Generator) -> uuid.UUID:
        return self._buf[int(rng.integers(len(self._buf)))]

    def __len__(self) -> int:
        return self._total

    def __bool__(self) -> bool:
        return self._total > 0


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
            attr_names=block.query_attrs,
        )
    else:
        raise ValueError(f"Unknown strategy: {block.strategy}")


def _emit_updates(
    update_ratio: float,
    pk_buffer: PKReservoir,
    cursor: DatasetCursor,
    writer: TSVWriter,
    rng: np.random.Generator,
) -> None:
    """Generate update operations using probabilistic rounding."""
    if update_ratio <= 0 or not pk_buffer:
        return
    int_part = int(update_ratio)
    frac_part = update_ratio - int_part
    num_updates = int_part
    if frac_part > 0 and rng.random() < frac_part:
        num_updates += 1
    for _ in range(num_updates):
        target_pk = pk_buffer.random_choice(rng)
        new_row = cursor.consume(1).iloc[0]
        writer.write_update(target_pk, _row_to_dict(new_row))


def execute_phases(
    config: WorkloadConfig,
    cursor: DatasetCursor,
    uniform_gen: UniformFilterGenerator,
    selectivity_gen: SelectivityFilterGenerator,
    writer: TSVWriter,
    rng: np.random.Generator,
) -> None:
    """Execute all phases in sequence."""
    pk_buffer = PKReservoir()
    for phase in config.phases:
        if isinstance(phase, WriteOnlyPhase):
            _run_write_only(phase, cursor, writer, rng, pk_buffer)
        elif isinstance(phase, PausePhase):
            _run_pause(phase, writer)
        elif isinstance(phase, MixedPhase):
            _run_mixed(phase, cursor, uniform_gen, selectivity_gen, writer, rng, pk_buffer)

    logger.info(
        "Generation complete: %d writes, %d updates, %d reads, %d pauses",
        writer.writes, writer.updates, writer.reads, writer.pauses,
    )


def _run_write_only(
    phase: WriteOnlyPhase,
    cursor: DatasetCursor,
    writer: TSVWriter,
    rng: np.random.Generator,
    pk_buffer: PKReservoir,
) -> None:
    logger.info("Phase '%s': write_only, %d rows, update_ratio=%.2f", phase.label, phase.rows, phase.update_ratio)
    chunk = cursor.consume(phase.rows)
    for _, row in chunk.iterrows():
        pk = _make_uuid(rng)
        writer.write_row(pk, _row_to_dict(row))
        pk_buffer.append(pk, rng)
        _emit_updates(phase.update_ratio, pk_buffer, cursor, writer, rng)


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
    pk_buffer: PKReservoir,
) -> None:
    logger.info(
        "Phase '%s': mixed, %d rows, read_ratio=%.2f, update_ratio=%.2f, %d query blocks",
        phase.label, phase.rows, phase.read_ratio, phase.update_ratio, len(phase.queries),
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
        pk_buffer.append(pk, rng)

        # Updates
        _emit_updates(phase.update_ratio, pk_buffer, cursor, writer, rng)

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
