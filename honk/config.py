"""Workload configuration loading, validation, and Cartesian expansion."""

import itertools
import json
from dataclasses import dataclass, field


class HonkConfigError(Exception):
    """Raised for invalid workload configuration."""
    pass


# --- Dataclasses ---

@dataclass
class ExpandedQueryBlock:
    label: str
    strategy: str  # "selectivity" | "uniform"
    weight: float
    expected_selectivity: float | None = None
    query_attr_num: int | None = None
    query_attr_indices: list[int] | None = None
    num_filters: int | None = None


@dataclass
class WriteOnlyPhase:
    label: str
    rows: int


@dataclass
class PausePhase:
    label: str
    duration_seconds: float


@dataclass
class MixedPhase:
    label: str
    rows: int
    read_ratio: float
    queries: list[ExpandedQueryBlock] = field(default_factory=list)


Phase = WriteOnlyPhase | PausePhase | MixedPhase


@dataclass
class WorkloadConfig:
    dataset: list[str]
    seed: int
    phases: list[Phase]

    @property
    def total_write_rows(self) -> int:
        """Total rows consumed across all phases."""
        total = 0
        for p in self.phases:
            if isinstance(p, (WriteOnlyPhase, MixedPhase)):
                total += p.rows
        return total


# --- Cartesian expansion ---

EXPAND_FIELDS = ("expected_selectivity", "query_attr_num")


def _expand_queries(raw_queries: list[dict]) -> list[ExpandedQueryBlock]:
    """Expand query blocks via Cartesian product of list-valued params."""
    result = []
    for q in raw_queries:
        _validate_query_block(q)
        label = q["label"]
        strategy = q["strategy"]
        weight = q.get("weight", 1)

        # Identify list-valued expansion params
        expand_params: dict[str, list] = {}
        for key in EXPAND_FIELDS:
            val = q.get(key)
            if val is None:
                continue
            if isinstance(val, list):
                expand_params[key] = val
            else:
                expand_params[key] = [val]

        # Static params (not expanded)
        query_attr_indices = q.get("query_attr_indices")
        num_filters = q.get("num_filters")

        if not expand_params:
            result.append(ExpandedQueryBlock(
                label=label,
                strategy=strategy,
                weight=weight,
                query_attr_indices=query_attr_indices,
                num_filters=num_filters,
            ))
            continue

        keys = list(expand_params.keys())
        for combo in itertools.product(*[expand_params[k] for k in keys]):
            overrides = dict(zip(keys, combo))
            result.append(ExpandedQueryBlock(
                label=label,
                strategy=strategy,
                weight=weight,
                expected_selectivity=overrides.get("expected_selectivity"),
                query_attr_num=overrides.get("query_attr_num"),
                query_attr_indices=query_attr_indices,
                num_filters=num_filters,
            ))

    return result


# --- Validation ---

def _validate_query_block(q: dict) -> None:
    """Validate a single query block definition."""
    label = q.get("label", "<unnamed>")
    strategy = q.get("strategy")

    if strategy not in ("selectivity", "uniform"):
        raise HonkConfigError(
            f"Query '{label}': unknown strategy '{strategy}', expected 'selectivity' or 'uniform'"
        )

    if strategy == "selectivity":
        if "expected_selectivity" not in q:
            raise HonkConfigError(
                f"Query '{label}': selectivity strategy requires 'expected_selectivity'"
            )
        has_num = "query_attr_num" in q
        has_indices = "query_attr_indices" in q
        if has_num and has_indices:
            raise HonkConfigError(
                f"Query '{label}': query_attr_num and query_attr_indices are mutually exclusive"
            )

    if strategy == "uniform":
        if "num_filters" not in q:
            raise HonkConfigError(
                f"Query '{label}': uniform strategy requires 'num_filters'"
            )


def _parse_phase(raw: dict) -> Phase:
    """Parse a single phase from raw JSON dict."""
    label = raw.get("label", "<unnamed>")
    phase_type = raw.get("type")

    if phase_type == "write_only":
        return WriteOnlyPhase(label=label, rows=raw["rows"])

    elif phase_type == "pause":
        return PausePhase(label=label, duration_seconds=raw["duration_seconds"])

    elif phase_type == "mixed":
        queries = _expand_queries(raw.get("queries", []))
        return MixedPhase(
            label=label,
            rows=raw["rows"],
            read_ratio=raw["read_ratio"],
            queries=queries,
        )

    else:
        raise HonkConfigError(f"Phase '{label}': unknown type '{phase_type}'")


def load_config(path: str) -> WorkloadConfig:
    """Load and validate a workload configuration from JSON."""
    with open(path) as f:
        raw = json.load(f)

    dataset = raw.get("dataset", [])
    if not dataset:
        raise HonkConfigError("'dataset' must be a non-empty list of file paths")

    seed = raw.get("seed", 42)
    phases = [_parse_phase(p) for p in raw.get("phases", [])]

    return WorkloadConfig(dataset=dataset, seed=seed, phases=phases)


def validate_row_budget(config: WorkloadConfig, available_rows: int) -> None:
    """Check that phases don't consume more rows than available."""
    needed = config.total_write_rows
    if needed > available_rows:
        raise HonkConfigError(
            f"Phases require {needed:,} rows but dataset has only {available_rows:,}"
        )
