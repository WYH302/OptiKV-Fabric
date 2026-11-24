#!/usr/bin/env python3
"""Characterize downloaded OptiKV-Fabric traces and draw Figure 2.

Reads whatever parquet files are actually present. Does not invent rows,
token lengths, or block ids. Writes:

- results/trace_stats/summary.json
- results/trace_stats/summary.csv
- results/trace_stats/freeinference_split.json  (only if that week is present)
- figures/fig2_trace_characterization.png
- figures/fig2_trace_characterization.pdf
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import sys
import textwrap
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
RESULTS = ROOT / "results" / "trace_stats"
FIGURES = ROOT / "figures"
SUMMARY_JSON = RESULTS / "summary.json"
SUMMARY_CSV = RESULTS / "summary.csv"
SPLIT_JSON = RESULTS / "freeinference_split.json"
FIG_PNG = FIGURES / "fig2_trace_characterization.png"
FIG_PDF = FIGURES / "fig2_trace_characterization.pdf"
SCRATCH = RESULTS / "_scratch"

INPUT_CANDIDATES = (
    "input_length",
    "prompt_tokens",
    "input_tokens",
    "prompt_length",
)
OUTPUT_CANDIDATES = (
    "output_length",
    "completion_tokens",
    "output_tokens",
    "completion_length",
)
BLOCK_CANDIDATES = ("hash_ids", "block_ids")
TIME_CANDIDATES = ("timestamp", "start_ms")
SESSION_CANDIDATES = ("session_id",)
CACHE_CANDIDATES = ("cache_read_tokens",)
ORDER = (
    "mooncake",
    "conversation",
    "synthetic",
    "toolagent",
    "freeinference",
    "lmcache",
)
DISPLAY = {
    "mooncake": "Mooncake",
    "conversation": "Conversation",
    "synthetic": "Synthetic",
    "toolagent": "ToolAgent",
    "freeinference": "FreeInference",
    "lmcache": "LMCache",
}
# Okabe-Ito, avoiding yellow on white.
COLORS = {
    "mooncake": "#0072B2",
    "conversation": "#E69F00",
    "synthetic": "#009E73",
    "toolagent": "#D55E00",
    "freeinference": "#CC79A7",
    "lmcache": "#000000",
}
LINESTYLES = {
    "mooncake": "-",
    "conversation": (0, (4, 1.4)),
    "synthetic": (0, (1, 1)),
    "toolagent": (0, (5, 1.0, 1, 1.0)),
    "freeinference": (0, (3, 1.0, 1, 1.0, 1, 1.0)),
    "lmcache": "-",
}
DOCUMENTED_BLOCK_TOKENS = {
    "mooncake": 512,
    "conversation": 512,
    "synthetic": 512,
    "toolagent": 512,
    "freeinference": 16,
}
DOCUMENTED_BLOCK_SOURCE = {
    "mooncake": "valeriol29/mooncake-traces dataset card states 512-token hash blocks; not re-tokenized here",
    "conversation": "valeriol29/mooncake-traces dataset card states 512-token hash blocks; not re-tokenized here",
    "synthetic": "valeriol29/mooncake-traces dataset card states 512-token hash blocks; not re-tokenized here",
    "toolagent": "valeriol29/mooncake-traces dataset card states 512-token hash blocks; not re-tokenized here",
    "freeinference": "FreeInference dataset card states chained 16-token blocks local to the week; not re-tokenized here",
}
PREFIX_DEFINITION = (
    "Prefix reuse is the fraction of block-id occurrences whose id appears "
    "in any strictly earlier request. Requests are ordered by "
    "(timestamp is null, timestamp, original row index), all ascending. "
    "Ids are marked seen only after the whole request is processed, so a "
    "repeated id inside one request is not reuse unless that id occurred "
    "in an earlier request. This is not the provider cache_read_tokens ratio."
)

CSV_FIELDS = [
    "trace_id",
    "display_name",
    "n_requests",
    "n_sessions",
    "timestamp_column",
    "input_length_column",
    "output_length_column",
    "block_id_column",
    "input_p50",
    "input_p90",
    "input_p95",
    "input_p99",
    "input_mean",
    "input_min",
    "input_max",
    "output_p50",
    "output_p90",
    "output_p95",
    "output_p99",
    "output_mean",
    "output_min",
    "output_max",
    "prefix_reuse_fraction",
    "n_block_occurrences",
    "n_reused_block_occurrences",
    "n_unique_block_ids",
    "pre_gap_p50",
    "pre_gap_p90",
    "pre_gap_p99",
    "pre_gap_mean",
    "pre_gap_frac_zero",
    "provider_cache_read_token_fraction",
    "requests_per_session_p50",
    "requests_per_session_mean",
    "requests_per_session_max",
    "source_files",
    "notes",
]


def _stdout_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def available_ram_bytes() -> int | None:
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullAvailPhys)
    except Exception:
        return None


def index_span_limit() -> int:
    ram = available_ram_bytes()
    if ram is None:
        return 400_000_000
    return int(min(800_000_000, max(1_000_000, ram // 8)))


def is_data_parquet(path: Path) -> bool:
    if not path.is_file():
        return False
    if any(part in {".cache", ".hf_cache", "_INCOMPLETE"} or part.endswith("_INCOMPLETE") for part in path.parts):
        return False
    try:
        if path.stat().st_size < 12:
            return False
        with path.open("rb") as handle:
            head = handle.read(4)
            handle.seek(-4, os.SEEK_END)
            tail = handle.read(4)
        return head == b"PAR1" and tail == b"PAR1"
    except OSError:
        return False


def discover() -> list[dict]:
    jobs = []
    mooncake = DATASETS / "mooncake-traces"
    if mooncake.is_dir():
        files = [path for path in sorted(mooncake.rglob("*.parquet")) if is_data_parquet(path)]
        for path in files:
            trace_id = path.parent.name if path.parent != mooncake else path.stem
            jobs.append(
                {
                    "trace_id": trace_id,
                    "display": DISPLAY.get(trace_id, trace_id),
                    "kind": "mooncake",
                    "source_repo": "valeriol29/mooncake-traces",
                    "files": [path],
                }
            )
    lmcache = DATASETS / "lmcache-agentic-traces"
    if lmcache.is_dir():
        files = [path for path in sorted(lmcache.rglob("*.parquet")) if is_data_parquet(path)]
        if files:
            jobs.append(
                {
                    "trace_id": "lmcache",
                    "display": "LMCache",
                    "kind": "lmcache",
                    "source_repo": "sammshen/lmcache-agentic-traces or zeelHz/lmcache-agentic-traces",
                    "files": files,
                }
            )
    free_root = DATASETS / "freeinference-agentic-trace"
    if free_root.is_dir():
        files = [
            path
            for path in sorted(free_root.rglob("requests.parquet"))
            if is_data_parquet(path) and ("2026-08-30" in path.as_posix() or "week_2026_08_30" in path.as_posix())
        ]
        if files:
            jobs.append(
                {
                    "trace_id": "freeinference",
                    "display": "FreeInference",
                    "kind": "freeinference",
                    "source_repo": "harvardMadsys/freeinference_agentic_trace",
                    "files": files,
                }
            )
    return jobs


def schema_of(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    schema = pq.ParquetFile(path).schema_arrow
    return [{"name": field.name, "type": str(field.type), "nullable": bool(field.nullable)} for field in schema]


def common_columns(files: list[Path]) -> tuple[list[str], list[dict]]:
    import pyarrow.parquet as pq

    name_sets = []
    schemas = []
    for path in files:
        file_schema = pq.ParquetFile(path).schema_arrow
        names = [field.name for field in file_schema]
        name_sets.append(set(names))
        schemas.append({"path": rel(path), "columns": schema_of(path)})
    if not name_sets:
        return [], schemas
    shared = set.intersection(*name_sets)
    # Preserve the first file's column order.
    first_names = [field.name for field in pq.ParquetFile(files[0]).schema_arrow]
    ordered = [name for name in first_names if name in shared]
    return ordered, schemas


def pick(names: set[str], candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in names:
            return name
    return None


def describe_numeric(values: np.ndarray, column: str | None) -> dict | None:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    result = {
        "column": column,
        "n_nonnull": int(finite.size),
        "n_null_or_nonfinite": int(array.size - finite.size),
        "percentile_method": "numpy.percentile linear",
    }
    if finite.size == 0:
        return result
    quantiles = [0, 50, 90, 95, 99, 100]
    percentiles = np.percentile(finite, quantiles)
    result["mean"] = float(finite.mean())
    for quantile, value in zip(quantiles, percentiles):
        key = "min" if quantile == 0 else "max" if quantile == 100 else f"p{quantile}"
        result[key] = float(value)
    return result


def read_numeric(table, name: str) -> np.ndarray:
    import pyarrow as pa
    import pyarrow.compute as pc

    column = table.column(name).combine_chunks()
    if pa.types.is_timestamp(column.type):
        column = pc.cast(column, pa.int64())
    casted = pc.cast(column, pa.float64(), safe=False)
    filled = pc.if_else(
        pc.is_null(casted),
        pa.scalar(np.nan, type=pa.float64()),
        casted,
    )
    return np.asarray(filled.to_numpy(zero_copy_only=False), dtype=np.float64)


def read_int_and_null(table, name: str) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow as pa
    import pyarrow.compute as pc

    column = table.column(name).combine_chunks()
    nulls = np.asarray(pc.is_null(column).to_numpy(zero_copy_only=False), dtype=bool)
    if pa.types.is_timestamp(column.type):
        column = pc.cast(column, pa.int64())
    filled = pc.if_else(pc.is_null(column), pa.scalar(0, type=column.type), column)
    values = np.asarray(filled.to_numpy(zero_copy_only=False))
    if not np.issubdtype(values.dtype, np.integer):
        values = values.astype(np.int64, copy=False)
    else:
        values = values.astype(np.int64, copy=False)
    return values, nulls


def causal_order(times: np.ndarray, nulls: np.ndarray) -> np.ndarray:
    rows = np.arange(times.size, dtype=np.int64)
    return np.lexsort((rows, times, nulls.astype(np.int8)))


def reuse_from_pieces(pieces, min_id: int, max_id: int, span_limit: int) -> dict:
    """pieces is an iterable of 1-d int64 arrays in causal request order."""
    span = int(max_id) - int(min_id)
    total = 0
    reused = 0
    method = None
    n_unique = None
    if span >= 0 and span <= span_limit:
        try:
            seen = np.zeros(span + 1, dtype=np.uint8)
        except MemoryError:
            seen = None
        if seen is not None:
            method = "shifted_uint8_index"
            base = np.int64(min_id)
            for piece in pieces:
                piece = np.asarray(piece, dtype=np.int64)
                if piece.size == 0:
                    continue
                shifted = piece - base
                total += int(piece.size)
                reused += int(np.count_nonzero(seen[shifted]))
                seen[shifted] = 1
            n_unique = int(np.count_nonzero(seen))
            del seen
    if method is None:
        method = "python_set"
        seen_set: set[int] = set()
        for piece in pieces:
            piece = np.asarray(piece, dtype=np.int64)
            if piece.size == 0:
                continue
            ids = piece.tolist()
            total += len(ids)
            for value in ids:
                if value in seen_set:
                    reused += 1
            seen_set.update(ids)
        n_unique = len(seen_set)
    return {
        "n_block_occurrences": int(total),
        "n_reused_occurrences": int(reused),
        "n_unique_block_ids": int(n_unique),
        "reuse_fraction": (float(reused) / float(total)) if total else None,
        "method": method,
        "block_id_min": int(min_id),
        "block_id_max": int(max_id),
    }


def close_memmap(memmap) -> None:
    if memmap is None:
        return
    try:
        memmap.flush()
    except Exception:
        pass
    mapped = getattr(memmap, "_mmap", None)
    if mapped is not None:
        try:
            mapped.close()
        except Exception:
            pass


def unlink_retry(path: Path) -> None:
    for _ in range(6):
        try:
            if path.exists():
                path.unlink()
            return
        except PermissionError:
            gc.collect()
            time.sleep(0.25)
    if path.exists():
        path.unlink()


def list_values_and_offsets(column):
    import pyarrow as pa

    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    offsets = np.asarray(column.offsets.to_numpy(), dtype=np.int64)
    if offsets.size == 0:
        return np.empty(0, dtype=np.int64), np.zeros(1, dtype=np.int64)
    start = int(offsets[0])
    stop = int(offsets[-1])
    values = column.values
    if values.null_count:
        raise RuntimeError("block-id list contains null items; reuse was not computed")
    array = np.asarray(values.to_numpy(zero_copy_only=False))
    if start != 0 or stop != array.size:
        if start < 0 or stop > array.size:
            raise RuntimeError(
                f"list offsets [{start}, {stop}) do not match values length {array.size}"
            )
        array = array[start:stop]
        offsets = offsets - start
    if array.dtype != np.int64:
        array = array.astype(np.int64, copy=False)
    return array, offsets


def block_reuse_for_files(
    files: list[Path],
    block_col: str,
    order: np.ndarray,
    scratch_path: Path,
) -> tuple[dict, np.ndarray]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    for path in files:
        field = pq.ParquetFile(path).schema_arrow.field(block_col)
        if not (pa.types.is_list(field.type) or pa.types.is_large_list(field.type)):
            raise RuntimeError(f"{block_col} has type {field.type}, not a list of ids")
    n_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    if order.size != n_rows:
        raise RuntimeError(f"causal order has {order.size} rows but parquet has {n_rows}")
    offsets = np.zeros(n_rows + 1, dtype=np.int64)
    scratch_path.parent.mkdir(parents=True, exist_ok=True)
    unlink_retry(scratch_path)
    row = 0
    min_id = None
    max_id = None
    memmap = None
    try:
        with scratch_path.open("wb") as handle:
            for path in files:
                parquet = pq.ParquetFile(path, memory_map=True)
                for batch in parquet.iter_batches(columns=[block_col], batch_size=2048):
                    column = batch.column(0)
                    values, local_offsets = list_values_and_offsets(column)
                    if values.size:
                        batch_min = int(values.min())
                        batch_max = int(values.max())
                        min_id = batch_min if min_id is None else min(min_id, batch_min)
                        max_id = batch_max if max_id is None else max(max_id, batch_max)
                        values.tofile(handle)
                    count = int(local_offsets.size - 1)
                    if count != batch.num_rows:
                        raise RuntimeError("block-id offsets do not match the batch row count")
                    base = int(offsets[row])
                    offsets[row : row + count + 1] = base + local_offsets
                    row += count
                    if row % 50000 < count:
                        print(f"  scanned {row:,} / {n_rows:,} rows for {block_col}", flush=True)
        if row != n_rows:
            raise RuntimeError(f"scanned {row} rows, expected {n_rows}")
        block_counts = np.diff(offsets).astype(np.int64, copy=False)
        total_ids = int(offsets[-1])
        if total_ids == 0 or min_id is None or max_id is None:
            return (
                {
                    "n_block_occurrences": 0,
                    "n_reused_occurrences": 0,
                    "n_unique_block_ids": 0,
                    "reuse_fraction": None,
                    "method": "empty",
                    "block_id_min": None,
                    "block_id_max": None,
                },
                block_counts,
            )
        memmap = np.memmap(scratch_path, dtype=np.int64, mode="r", shape=(total_ids,))

        def pieces():
            for index in order:
                start = int(offsets[int(index)])
                stop = int(offsets[int(index) + 1])
                if stop > start:
                    yield memmap[start:stop]

        stats = reuse_from_pieces(pieces(), int(min_id), int(max_id), index_span_limit())
        stats["mean_block_ids_per_request"] = float(block_counts.mean()) if block_counts.size else None
        return stats, block_counts
    finally:
        close_memmap(memmap)
        memmap = None
        gc.collect()
        if scratch_path.exists():
            unlink_retry(scratch_path)


def ratio_summary(lengths: np.ndarray, counts: np.ndarray) -> dict | None:
    mask = np.isfinite(lengths) & (counts > 0) & (lengths >= 0)
    if not np.any(mask):
        return None
    ratios = lengths[mask] / counts[mask].astype(np.float64)
    summary = describe_numeric(ratios, "recorded_input_length_per_stored_block_id")
    if summary is not None:
        summary["n_rows"] = int(mask.sum())
        summary["meaning"] = (
            "Recorded input length divided by the number of stored block ids. "
            "This is a comparison with the documented block size, not a new tokenization."
        )
    return summary


def requests_per_session(session_ids: list) -> dict | None:
    counts = Counter(value for value in session_ids if value is not None)
    if not counts:
        return None
    array = np.fromiter(counts.values(), dtype=np.int64)
    summary = describe_numeric(array.astype(np.float64), "requests_per_session")
    if summary is not None:
        summary["n_sessions"] = int(len(counts))
    return summary


def maybe_unix_ms_iso(value: int) -> str | None:
    if value < 10**12 or value > 2 * 10**12:
        return None
    try:
        return datetime.fromtimestamp(value / 1000.0, timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def build_split(times: np.ndarray, nulls: np.ndarray, order: np.ndarray, session_ids: list | None, source_files: list[str]) -> dict:
    n_rows = int(times.size)
    n_train = (n_rows * 7) // 10
    n_test = n_rows - n_train
    document = {
        "dataset": "harvardMadsys/freeinference_agentic_trace",
        "config": "requests",
        "split": "week_2026_08_30",
        "files": source_files,
        "timestamp_column": "start_ms",
        "n_rows": n_rows,
        "train_fraction_target": 0.7,
        "n_train": n_train,
        "n_test": n_test,
        "actual_train_fraction": (n_train / n_rows) if n_rows else None,
        "n_null_timestamp": int(nulls.sum()),
        "rule": (
            "Let K = (timestamp_is_null, timestamp_value_or_0_when_null, original_row_index) "
            "with original_row_index the 0-based parquet scan order. Sort K ascending. "
            "n_train = (n_rows * 7) // 10. The first n_train rows are train and the rest are test. "
            "Because K includes the row index, keys are unique, so train rows are exactly those with K <= K_cut."
        ),
        "scheduler_tuning": False,
    }
    if n_rows == 0 or n_train == 0:
        document["note"] = "Not enough rows to form both sides of a 70/30 row split."
        return document
    cut = int(order[n_train - 1])
    document["cut"] = {
        "last_train_original_row": cut,
        "last_train_timestamp_is_null": bool(nulls[cut]),
        "last_train_timestamp": None if bool(nulls[cut]) else int(times[cut]),
    }
    if n_test:
        first_test = int(order[n_train])
        document["cut"]["first_test_original_row"] = first_test
        document["cut"]["first_test_timestamp_is_null"] = bool(nulls[first_test])
        document["cut"]["first_test_timestamp"] = None if bool(nulls[first_test]) else int(times[first_test])
        same = (
            (not bool(nulls[cut]))
            and (not bool(nulls[first_test]))
            and int(times[cut]) == int(times[first_test])
        )
        document["boundary_timestamp_shared_by_train_and_test"] = bool(same)
        if same:
            stamp = int(times[cut])
            on_stamp = ~nulls & (times == stamp)
            document["boundary_timestamp"] = stamp
            document["n_rows_with_boundary_timestamp"] = int(on_stamp.sum())
            document["n_boundary_timestamp_rows_in_train"] = int(np.isin(order[:n_train], np.flatnonzero(on_stamp)).sum())
            document["n_boundary_timestamp_rows_in_test"] = int(np.isin(order[n_train:], np.flatnonzero(on_stamp)).sum())
            document["threshold_warning"] = (
                "Do not reproduce this split with timestamp < T alone. "
                "Rows with the boundary timestamp fall on both sides because the cut is a row count."
            )
    train_times = times[order[:n_train]]
    train_nulls = nulls[order[:n_train]]
    test_times = times[order[n_train:]]
    test_nulls = nulls[order[n_train:]]
    if np.any(~train_nulls):
        kept = train_times[~train_nulls]
        document["train_timestamp_min"] = int(kept.min())
        document["train_timestamp_max"] = int(kept.max())
        document["train_timestamp_min_utc"] = maybe_unix_ms_iso(int(kept.min()))
        document["train_timestamp_max_utc"] = maybe_unix_ms_iso(int(kept.max()))
    if test_times.size and np.any(~test_nulls):
        kept = test_times[~test_nulls]
        document["test_timestamp_min"] = int(kept.min())
        document["test_timestamp_max"] = int(kept.max())
        document["test_timestamp_min_utc"] = maybe_unix_ms_iso(int(kept.min()))
        document["test_timestamp_max_utc"] = maybe_unix_ms_iso(int(kept.max()))
    if document.get("train_timestamp_min_utc") is None and document.get("train_timestamp_min") is not None:
        document["timestamp_interpretation"] = (
            "Values were left as raw integers. They were not converted to UTC because they are outside the Unix-millisecond range used by this script."
        )
    elif document.get("train_timestamp_min_utc") is not None:
        document["timestamp_interpretation"] = (
            "start_ms values in [1e12, 2e12] were also written as UTC timestamps, interpreting the column as Unix epoch milliseconds."
        )
    if session_ids is not None:
        train_sessions = {session_ids[int(index)] for index in order[:n_train] if session_ids[int(index)] is not None}
        test_sessions = {session_ids[int(index)] for index in order[n_train:] if session_ids[int(index)] is not None}
        document["n_sessions_train"] = len(train_sessions)
        document["n_sessions_test"] = len(test_sessions)
        document["n_sessions_in_both"] = len(train_sessions & test_sessions)
        document["session_split_note"] = (
            "The split is by request row, not by session. Sessions listed in both sides cross the time cut."
        )
    return document


def pre_gap_unit(text: str) -> tuple[str | None, str | None]:
    lowered = text.lower()
    start = 0
    while True:
        index = lowered.find("pre_gap", start)
        if index < 0:
            return None, None
        window = text[index : index + 500]
        window_lower = window.lower()
        if "millisecond" in window_lower:
            return "milliseconds", " ".join(window.split())
        if "second" in window_lower:
            return "seconds", " ".join(window.split())
        start = index + len("pre_gap")


def find_lmcache_readme() -> Path | None:
    root = DATASETS / "lmcache-agentic-traces"
    if not root.is_dir():
        return None
    for path in sorted(root.rglob("README.md")):
        if ".hf_cache" in path.parts or ".cache" in path.parts:
            continue
        return path
    return None


def characterize_job(job: dict) -> tuple[dict, dict, dict | None]:
    import pyarrow.parquet as pq

    files: list[Path] = job["files"]
    columns, schemas = common_columns(files)
    names = set(columns)
    input_col = pick(names, INPUT_CANDIDATES)
    output_col = pick(names, OUTPUT_CANDIDATES)
    block_col = pick(names, BLOCK_CANDIDATES)
    time_col = pick(names, TIME_CANDIDATES)
    session_col = pick(names, SESSION_CANDIDATES)
    cache_col = pick(names, CACHE_CANDIDATES)
    load_cols = [
        name
        for name in (input_col, output_col, time_col, session_col, cache_col, "pre_gap", "model")
        if name and name in names
    ]
    # pre_gap and model are optional exact names; include them only when present.
    if "pre_gap" in names and "pre_gap" not in load_cols:
        load_cols.append("pre_gap")
    if "model" in names and "model" not in load_cols:
        load_cols.append("model")
    table = pa_concat(files, load_cols) if load_cols else None
    n_requests = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    notes = []
    all_column_names = schemas[0]["columns"] if schemas else []
    column_name_set = {item["name"] for item in all_column_names}
    if "input" in column_name_set and input_col is None:
        notes.append(
            "The input column is present but is not a numeric length. "
            "Message text was not tokenized, so input-length percentiles were not computed."
        )
    if block_col is None:
        notes.append("No hash_ids or block_ids column is present, so prefix reuse was not computed.")
    if session_col is None:
        notes.append("No session_id column is present, so the session count is null.")

    input_values = read_numeric(table, input_col) if table is not None and input_col else None
    output_values = read_numeric(table, output_col) if table is not None and output_col else None
    pre_gap_values = read_numeric(table, "pre_gap") if table is not None and "pre_gap" in names else None
    cache_values = read_numeric(table, cache_col) if table is not None and cache_col else None
    session_ids = table.column(session_col).combine_chunks().to_pylist() if table is not None and session_col else None
    model_counts = None
    if table is not None and "model" in names:
        counts = Counter(value for value in table.column("model").combine_chunks().to_pylist() if value is not None)
        if len(counts) <= 40:
            model_counts = dict(counts)
        else:
            most_common = counts.most_common(20)
            model_counts = {key: value for key, value in most_common}
            model_counts["__other__"] = int(sum(counts.values()) - sum(value for _, value in most_common))
            notes.append("model_counts lists the 20 most common models; __other__ sums the rest.")

    times = None
    nulls = None
    order = None
    if table is not None and time_col:
        times, nulls = read_int_and_null(table, time_col)
        order = causal_order(times, nulls)
    elif block_col:
        notes.append("Block ids exist, but there is no timestamp or start_ms column, so causal prefix reuse was not computed.")

    input_stats = describe_numeric(input_values, input_col) if input_values is not None else None
    output_stats = describe_numeric(output_values, output_col) if output_values is not None else None
    pre_gap_stats = None
    pre_unit = None
    pre_evidence = None
    if pre_gap_values is not None:
        pre_gap_stats = describe_numeric(pre_gap_values, "pre_gap")
        finite = pre_gap_values[np.isfinite(pre_gap_values)]
        if pre_gap_stats is not None:
            pre_gap_stats["frac_zero"] = float(np.mean(finite == 0)) if finite.size else None
            pre_gap_stats["frac_negative"] = float(np.mean(finite < 0)) if finite.size else None
            pre_gap_stats["n_zero"] = int(np.sum(finite == 0)) if finite.size else 0
            pre_gap_stats["n_negative"] = int(np.sum(finite < 0)) if finite.size else 0
            pre_gap_stats["n_positive"] = int(np.sum(finite > 0)) if finite.size else 0
        readme = find_lmcache_readme()
        if readme is not None:
            pre_unit, pre_evidence = pre_gap_unit(readme.read_text(encoding="utf-8", errors="replace"))
            if pre_gap_stats is not None:
                pre_gap_stats["unit"] = pre_unit
                pre_gap_stats["unit_evidence"] = pre_evidence
                pre_gap_stats["unit_source"] = rel(readme)
            if pre_unit is None:
                notes.append("pre_gap is present, but the README next to the dataset did not state its unit.")

    session_summary = requests_per_session(session_ids) if session_ids is not None else None
    n_sessions = None if session_summary is None else session_summary.get("n_sessions")
    n_null_sessions = None
    if session_ids is not None:
        n_null_sessions = int(sum(value is None for value in session_ids))

    reuse = None
    ratios = None
    if block_col and order is not None:
        scratch = SCRATCH / f"{job['trace_id']}_block_ids.bin"
        try:
            reuse, block_counts = block_reuse_for_files(files, block_col, order, scratch)
            reuse["definition"] = PREFIX_DEFINITION
            reuse["timestamp_column"] = time_col
            reuse["block_id_column"] = block_col
            documented = DOCUMENTED_BLOCK_TOKENS.get(job["trace_id"])
            reuse["documented_block_tokens"] = documented
            reuse["documented_block_tokens_source"] = DOCUMENTED_BLOCK_SOURCE.get(job["trace_id"])
            if input_values is not None:
                ratios = ratio_summary(input_values, block_counts)
        except Exception as exc:
            notes.append(f"Prefix reuse failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            reuse = None

    provider_cache = None
    if cache_values is not None and input_values is not None:
        mask = np.isfinite(cache_values) & np.isfinite(input_values) & (input_values > 0)
        if np.any(mask):
            weighted = float(cache_values[mask].sum() / input_values[mask].sum())
            per_request = cache_values[mask] / input_values[mask]
            provider_cache = {
                "cache_column": cache_col,
                "input_column": input_col,
                "token_weighted_cache_read_over_prompt": weighted,
                "median_per_request_cache_read_over_prompt": float(np.median(per_request)),
                "n_rows": int(mask.sum()),
                "note": "Provider-reported cache_read_tokens. This is not the causal block-id reuse fraction.",
            }

    split_doc = None
    if job["kind"] == "freeinference" and times is not None and nulls is not None and order is not None:
        if time_col != "start_ms":
            notes.append(
                f"FreeInference split used timestamp column {time_col} because start_ms was not present."
            )
        split_doc = build_split(
            times,
            nulls,
            order,
            session_ids,
            [rel(path) for path in files],
        )
        split_doc["timestamp_column"] = time_col

    record = {
        "trace_id": job["trace_id"],
        "display_name": job["display"],
        "kind": job["kind"],
        "source_repo": job["source_repo"],
        "files": [{"path": rel(path), "bytes": path.stat().st_size} for path in files],
        "n_requests": int(n_requests),
        "n_sessions": n_sessions,
        "n_null_session_id": n_null_sessions,
        "columns": schemas[0]["columns"] if len(schemas) == 1 else schemas,
        "schemas_identical": len({json.dumps(item["columns"], sort_keys=True) for item in schemas}) == 1,
        "timestamp_column": time_col,
        "input_length": input_stats,
        "output_length": output_stats,
        "prefix_reuse": reuse,
        "input_tokens_per_block": ratios,
        "pre_gap": pre_gap_stats,
        "requests_per_session": session_summary,
        "provider_cache": provider_cache,
        "model_counts": model_counts,
        "notes": notes,
    }
    payload = {
        "trace_id": job["trace_id"],
        "display": job["display"],
        "n_requests": int(n_requests),
        "input": None if input_values is None else np.array(input_values, dtype=np.float64, copy=True),
        "input_column": input_col,
        "reuse_fraction": None if not reuse else reuse.get("reuse_fraction"),
        "pre_gap": None if pre_gap_values is None else np.array(pre_gap_values, dtype=np.float64, copy=True),
        "pre_gap_unit": pre_unit,
    }
    return record, payload, split_doc


def pa_concat(files: list[Path], columns: list[str]):
    import pyarrow as pa
    import pyarrow.parquet as pq

    tables = [pq.read_table(path, columns=columns) for path in files]
    if len(tables) == 1:
        return tables[0]
    return pa.concat_tables(tables, promote_options="default")


def stat_get(stats: dict | None, key: str):
    if not stats:
        return None
    return stats.get(key)


def csv_row(record: dict) -> dict:
    reuse = record.get("prefix_reuse") or {}
    sessions = record.get("requests_per_session") or {}
    provider = record.get("provider_cache") or {}
    notes = " ".join(record.get("notes") or [])
    row = {
        "trace_id": record["trace_id"],
        "display_name": record["display_name"],
        "n_requests": record["n_requests"],
        "n_sessions": record["n_sessions"],
        "timestamp_column": record.get("timestamp_column"),
        "input_length_column": stat_get(record.get("input_length"), "column"),
        "output_length_column": stat_get(record.get("output_length"), "column"),
        "block_id_column": reuse.get("block_id_column"),
        "input_p50": stat_get(record.get("input_length"), "p50"),
        "input_p90": stat_get(record.get("input_length"), "p90"),
        "input_p95": stat_get(record.get("input_length"), "p95"),
        "input_p99": stat_get(record.get("input_length"), "p99"),
        "input_mean": stat_get(record.get("input_length"), "mean"),
        "input_min": stat_get(record.get("input_length"), "min"),
        "input_max": stat_get(record.get("input_length"), "max"),
        "output_p50": stat_get(record.get("output_length"), "p50"),
        "output_p90": stat_get(record.get("output_length"), "p90"),
        "output_p95": stat_get(record.get("output_length"), "p95"),
        "output_p99": stat_get(record.get("output_length"), "p99"),
        "output_mean": stat_get(record.get("output_length"), "mean"),
        "output_min": stat_get(record.get("output_length"), "min"),
        "output_max": stat_get(record.get("output_length"), "max"),
        "prefix_reuse_fraction": reuse.get("reuse_fraction"),
        "n_block_occurrences": reuse.get("n_block_occurrences"),
        "n_reused_block_occurrences": reuse.get("n_reused_occurrences"),
        "n_unique_block_ids": reuse.get("n_unique_block_ids"),
        "pre_gap_p50": stat_get(record.get("pre_gap"), "p50"),
        "pre_gap_p90": stat_get(record.get("pre_gap"), "p90"),
        "pre_gap_p99": stat_get(record.get("pre_gap"), "p99"),
        "pre_gap_mean": stat_get(record.get("pre_gap"), "mean"),
        "pre_gap_frac_zero": stat_get(record.get("pre_gap"), "frac_zero"),
        "provider_cache_read_token_fraction": provider.get("token_weighted_cache_read_over_prompt"),
        "requests_per_session_p50": sessions.get("p50"),
        "requests_per_session_mean": sessions.get("mean"),
        "requests_per_session_max": sessions.get("max"),
        "source_files": "; ".join(item["path"] for item in record.get("files") or []),
        "notes": notes,
    }
    return {key: row.get(key) for key in CSV_FIELDS}


def format_csv(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format(value, ".8g")
    return str(value)


def sanitize(value):
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None:
        return None
    return value


def write_csv(records: list[dict]) -> None:
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({key: format_csv(value) for key, value in csv_row(record).items()})


def ecdf_positive(values: np.ndarray, max_points: int = 900) -> tuple[np.ndarray, np.ndarray, int, int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    n_all = int(finite.size)
    n_nonpositive = int(np.sum(finite <= 0))
    positive = np.sort(finite[finite > 0])
    if positive.size == 0:
        return positive, positive, n_all, n_nonpositive
    if positive.size > max_points:
        indices = np.unique(np.linspace(0, positive.size - 1, max_points).astype(int))
    else:
        indices = np.arange(positive.size)
    xs = positive[indices]
    ys = (n_nonpositive + indices + 1) / n_all
    return xs, ys, n_all, n_nonpositive


def ecdf_conditional_positive(values: np.ndarray, max_points: int = 900):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    n_all = int(finite.size)
    n_zero = int(np.sum(finite == 0))
    n_negative = int(np.sum(finite < 0))
    positive = np.sort(finite[finite > 0])
    if positive.size == 0:
        return positive, positive, n_all, n_zero, n_negative
    if positive.size > max_points:
        indices = np.unique(np.linspace(0, positive.size - 1, max_points).astype(int))
    else:
        indices = np.arange(positive.size)
    xs = positive[indices]
    ys = (indices + 1) / positive.size
    return xs, ys, n_all, n_zero, n_negative


def ordered_payloads(payloads: list[dict]) -> list[dict]:
    rank = {trace_id: index for index, trace_id in enumerate(ORDER)}
    return sorted(payloads, key=lambda item: rank.get(item["trace_id"], 100))


def make_figure(payloads: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter

    payloads = ordered_payloads(payloads)
    input_items = [item for item in payloads if item["input"] is not None and np.isfinite(item["input"]).any()]
    reuse_items = [item for item in payloads if item["reuse_fraction"] is not None]
    pre_items = [item for item in payloads if item["pre_gap"] is not None and np.isfinite(item["pre_gap"]).any()]
    if not input_items and not reuse_items and not pre_items:
        raise RuntimeError("Figure 2 has no plottable columns.")

    available_fonts = {font.name for font in matplotlib.font_manager.fontManager.ttflist}
    if "Times New Roman" in available_fonts:
        serif = ["Times New Roman", "Times", "DejaVu Serif"]
    else:
        serif = ["Times", "DejaVu Serif"]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": serif,
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": "#222222",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
        }
    )

    n_top = int(bool(input_items)) + int(bool(reuse_items))
    has_bottom = bool(pre_items)
    if n_top and has_bottom:
        figsize = (7.4, 7.15)
    elif n_top == 2:
        figsize = (7.4, 4.9)
    else:
        figsize = (4.9, 3.45)
    fig = plt.figure(figsize=figsize, facecolor="white")
    title_bits = []
    if input_items:
        title_bits.append("input length")
    if reuse_items:
        title_bits.append("causal block reuse")
    if pre_items:
        title_bits.append("LMCache pre_gap")
    fig.suptitle("Trace characterization: " + ", ".join(title_bits), fontsize=11, y=0.985)

    footnote_parts = []
    if input_items:
        columns = sorted({item["input_column"] for item in input_items if item["input_column"]})
        footnote_parts.append(
            "Input length is the recorded column " + " or ".join(columns) + "; curves are empirical CDFs."
        )
    if reuse_items:
        footnote_parts.append(
            "Reuse is the share of block ids already seen in an earlier request, ordered by timestamp."
        )
        documented = []
        if any(item["trace_id"] in {"mooncake", "conversation", "synthetic", "toolagent"} for item in reuse_items):
            documented.append("Mooncake-family hash ids are documented as 512-token blocks")
        if any(item["trace_id"] == "freeinference" for item in reuse_items):
            documented.append("FreeInference block ids are documented as 16-token blocks and cover week_2026_08_30 only")
        if documented:
            footnote_parts.append("; ".join(documented) + ".")
    if pre_items:
        unit = pre_items[0]["pre_gap_unit"]
        unit_text = f" in {unit}" if unit else ""
        footnote_parts.append(
            f"LMCache is omitted from the input and reuse panels because the download has pre_gap{unit_text} but no input-length or block-id column."
        )

    top_axes = []
    if n_top and has_bottom:
        grid = fig.add_gridspec(
            2,
            max(n_top, 1),
            height_ratios=[1.22, 1.0],
            left=0.08,
            right=0.985,
            top=0.92,
            bottom=0.23,
            hspace=0.48,
            wspace=0.48,
        )
        for index in range(n_top):
            top_axes.append(fig.add_subplot(grid[0, index]))
        axis_pre = fig.add_subplot(grid[1, :])
    elif n_top == 2:
        grid = fig.add_gridspec(1, 2, left=0.08, right=0.985, top=0.88, bottom=0.40, wspace=0.50)
        top_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
        axis_pre = None
    elif n_top == 1:
        grid = fig.add_gridspec(1, 1, left=0.14, right=0.97, top=0.84, bottom=0.24)
        top_axes = [fig.add_subplot(grid[0, 0])]
        axis_pre = None
    else:
        grid = fig.add_gridspec(1, 1, left=0.12, right=0.97, top=0.84, bottom=0.22)
        top_axes = []
        axis_pre = fig.add_subplot(grid[0, 0])

    letters = []
    axis_input = None
    axis_reuse = None
    cursor = 0
    if input_items:
        axis_input = top_axes[cursor]
        cursor += 1
        letters.append(("a", axis_input, "Input length"))
    if reuse_items:
        axis_reuse = top_axes[cursor]
        cursor += 1
        letters.append(("b" if input_items else "a", axis_reuse, "Causal block reuse"))
    if axis_pre is not None:
        next_letter = chr(ord("a") + len(letters))
        letters.append((next_letter, axis_pre, "LMCache pre_gap"))

    if axis_input is not None:
        nonpositive_notes = []
        for item in input_items:
            xs, ys, n_all, n_nonpositive = ecdf_positive(item["input"])
            if xs.size == 0:
                continue
            axis_input.plot(
                xs,
                ys,
                color=COLORS.get(item["trace_id"], "#0072B2"),
                linestyle=LINESTYLES.get(item["trace_id"], "-"),
                linewidth=1.7,
                label=f"{item['display']} (n={item['n_requests']:,})",
                drawstyle="steps-post",
                solid_capstyle="round",
            )
            if n_nonpositive:
                nonpositive_notes.append(f"{item['display']} excludes {n_nonpositive:,} non-positive values from the x-axis")
        axis_input.set_xscale("log")
        axis_input.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
        axis_input.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=12))
        axis_input.xaxis.set_minor_formatter(NullFormatter())
        axis_input.set_ylim(-0.02, 1.04)
        axis_input.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axis_input.set_xlabel("Input length (tokens, log scale)")
        axis_input.set_ylabel("Cumulative fraction of requests")
        axis_input.grid(True, which="major", axis="both", color="#D0D0D0", linewidth=0.5, zorder=0)
        axis_input.set_axisbelow(True)
        handles, labels = axis_input.get_legend_handles_labels()
        legend_y = 0.27 if axis_pre is None else 0.145
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=2 if len(labels) <= 4 else 3,
            frameon=False,
            handlelength=2.6,
            columnspacing=1.3,
            labelspacing=0.35,
            fontsize=7.2,
        )
        if nonpositive_notes:
            footnote_parts.append(" ".join(nonpositive_notes) + ".")

    if axis_reuse is not None:
        names = [item["display"] for item in reuse_items]
        percents = [100.0 * float(item["reuse_fraction"]) for item in reuse_items]
        colors = [COLORS.get(item["trace_id"], "#0072B2") for item in reuse_items]
        positions = np.arange(len(names))
        axis_reuse.barh(positions, percents, color=colors, height=0.62, zorder=3)
        axis_reuse.set_yticks(positions)
        axis_reuse.set_yticklabels(names)
        axis_reuse.invert_yaxis()
        xmax = max(percents) if percents else 1.0
        axis_reuse.set_xlim(0, max(xmax * 1.32, xmax + 14))
        for position, percent in zip(positions, percents):
            axis_reuse.text(
                percent + axis_reuse.get_xlim()[1] * 0.012,
                position,
                f"{percent:.1f}%",
                va="center",
                ha="left",
                fontsize=7.4,
                color="#222222",
                clip_on=True,
            )
        axis_reuse.set_xlabel("Block ids seen in an earlier request (%)")
        axis_reuse.grid(True, axis="x", color="#D0D0D0", linewidth=0.5, zorder=0)
        axis_reuse.set_axisbelow(True)
        axis_reuse.tick_params(axis="y", length=0)

    if axis_pre is not None:
        # One LMCache series is expected. If several pre_gap columns exist, plot each.
        notes = []
        for item in pre_items:
            xs, ys, n_all, n_zero, n_negative = ecdf_conditional_positive(item["pre_gap"])
            if xs.size == 0:
                notes.append(f"{item['display']}: no positive pre_gap")
                continue
            axis_pre.plot(
                xs,
                ys,
                color=COLORS.get(item["trace_id"], "#000000"),
                linestyle="-",
                linewidth=1.7,
                drawstyle="steps-post",
                label=item["display"],
            )
            zero_text = f"pre_gap = 0 on {n_zero:,} of {n_all:,} rows"
            if n_negative:
                zero_text += f"\npre_gap < 0 on {n_negative:,} rows, omitted here"
            axis_pre.text(
                0.98,
                0.06,
                zero_text,
                transform=axis_pre.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.3,
                color="#222222",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5, "alpha": 0.92},
            )
        positive_gaps = [
            item["pre_gap"][np.isfinite(item["pre_gap"]) & (item["pre_gap"] > 0)]
            for item in pre_items
            if item["pre_gap"] is not None
        ]
        positive_gaps = [values for values in positive_gaps if values.size]
        if positive_gaps and max(float(values.max()) / max(float(values.min()), 1e-12) for values in positive_gaps) >= 10:
            axis_pre.set_xscale("log")
            axis_pre.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
            axis_pre.xaxis.set_minor_formatter(NullFormatter())
        unit = next((item["pre_gap_unit"] for item in pre_items if item["pre_gap_unit"]), None)
        axis_pre.set_xlabel("pre_gap" + (f" ({unit}, log scale)" if unit else " (positive rows, log scale)"))
        axis_pre.set_ylabel("Cumulative fraction of positive pre_gap")
        axis_pre.set_ylim(-0.02, 1.04)
        axis_pre.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axis_pre.grid(True, which="major", color="#D0D0D0", linewidth=0.5)
        axis_pre.set_axisbelow(True)
        if len(pre_items) > 1:
            axis_pre.legend(frameon=False, loc="upper left")

    for letter, axis, heading in letters:
        axis.set_title(f"({letter}) {heading}", loc="left", pad=4, fontsize=10)

    fig.text(
        0.08,
        0.015,
        textwrap.fill(" ".join(footnote_parts), width=124),
        ha="left",
        va="bottom",
        fontsize=6.7,
        color="#333333",
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PNG, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(FIG_PDF, format="pdf", facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    if not FIG_PNG.exists() or FIG_PNG.stat().st_size < 10000:
        raise RuntimeError("PNG figure was not written")
    with FIG_PDF.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise RuntimeError("PDF figure is not a PDF")


def self_test() -> None:
    first = np.array([46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 2111, 2112], dtype=np.int64)
    second = np.array([46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 2124], dtype=np.int64)
    stats = reuse_from_pieces([first, second], 46, 2124, 10_000_000)
    if stats["n_block_occurrences"] != 27 or stats["n_reused_occurrences"] != 12 or stats["n_unique_block_ids"] != 15:
        raise AssertionError(f"mooncake example reuse failed: {stats}")
    within = reuse_from_pieces(
        [np.array([1, 1, 2], dtype=np.int64), np.array([1], dtype=np.int64)],
        1,
        2,
        10_000_000,
    )
    if within["n_reused_occurrences"] != 1 or within["n_block_occurrences"] != 4 or within["n_unique_block_ids"] != 2:
        raise AssertionError(f"within-request reuse failed: {within}")
    shifted = reuse_from_pieces(
        [np.array([-2, -2, 3], dtype=np.int64), np.array([-2, 4], dtype=np.int64)],
        -2,
        4,
        10_000_000,
    )
    if shifted["n_reused_occurrences"] != 1 or shifted["n_unique_block_ids"] != 3 or shifted["n_block_occurrences"] != 5:
        raise AssertionError(f"shifted reuse failed: {shifted}")
    times = np.array([5, 1, 1, 9], dtype=np.int64)
    nulls = np.array([False, False, False, False])
    order = causal_order(times, nulls)
    if list(map(int, order)) != [1, 2, 0, 3]:
        raise AssertionError(f"causal order failed: {order}")
    n_train = (4 * 7) // 10
    if n_train != 2:
        raise AssertionError(n_train)


def print_brief(records: list[dict]) -> None:
    for record in records:
        reuse = (record.get("prefix_reuse") or {}).get("reuse_fraction")
        in_p50 = stat_get(record.get("input_length"), "p50")
        out_p50 = stat_get(record.get("output_length"), "p50")
        pre_p50 = stat_get(record.get("pre_gap"), "p50")
        print(
            f"{record['trace_id']}: n={record['n_requests']} sessions={record['n_sessions']} "
            f"input_p50={in_p50} output_p50={out_p50} reuse={reuse} pre_gap_p50={pre_p50}",
            flush=True,
        )


def main() -> int:
    _stdout_utf8()
    self_test()
    if "--self-test" in sys.argv:
        print("self-test ok", flush=True)
        return 0
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    jobs = discover()
    records = []
    payloads = []
    split_doc = None
    errors = []
    for job in jobs:
        print(f"Characterizing {job['trace_id']} ({len(job['files'])} files)", flush=True)
        try:
            record, payload, split = characterize_job(job)
            records.append(record)
            payloads.append(payload)
            if split is not None:
                split_doc = split
        except Exception as exc:
            traceback.print_exc()
            errors.append({"trace_id": job["trace_id"], "error": f"{type(exc).__name__}: {exc}"})
    figure_info = None
    figure_error = None
    if payloads and any(
        item["input"] is not None or item["reuse_fraction"] is not None or item["pre_gap"] is not None
        for item in payloads
    ):
        try:
            make_figure(payloads)
            figure_info = {
                "png": rel(FIG_PNG),
                "pdf": rel(FIG_PDF),
                "png_bytes": FIG_PNG.stat().st_size,
                "pdf_bytes": FIG_PDF.stat().st_size,
                "dpi": 300,
            }
        except Exception as exc:
            traceback.print_exc()
            figure_error = f"{type(exc).__name__}: {exc}"
    elif not jobs:
        errors.append({"trace_id": None, "error": "No parquet traces found under datasets/."})
    if split_doc is not None:
        SPLIT_JSON.write_text(json.dumps(sanitize(split_doc), indent=2), encoding="utf-8")
    summary = {
        "generated_at_utc": now_utc(),
        "definitions": {
            "prefix_reuse": PREFIX_DEFINITION,
            "percentile_method": "numpy.percentile linear",
            "freeinference_split": (
                "First (n_rows * 7) // 10 rows after sorting by (timestamp is null, timestamp, original row index)."
            ),
            "lengths": "Input and output percentiles use only numeric columns that exist. Message text is not tokenized.",
        },
        "traces": records,
        "errors": errors,
        "figure": figure_info,
        "figure_error": figure_error,
        "freeinference_split_file": rel(SPLIT_JSON) if split_doc is not None else None,
        "download_status_file": rel(DATASETS / "DOWNLOAD_STATUS.md") if (DATASETS / "DOWNLOAD_STATUS.md").exists() else None,
    }
    SUMMARY_JSON.write_text(json.dumps(sanitize(summary), indent=2), encoding="utf-8")
    write_csv(records)
    print(f"Wrote {rel(SUMMARY_JSON)}", flush=True)
    print(f"Wrote {rel(SUMMARY_CSV)}", flush=True)
    if split_doc is not None:
        print(f"Wrote {rel(SPLIT_JSON)}", flush=True)
    if figure_info:
        print(f"Wrote {rel(FIG_PNG)} and {rel(FIG_PDF)}", flush=True)
    print_brief(records)
    for error in errors:
        print(f"ERROR {error}", flush=True)
    if figure_error:
        print(f"FIGURE ERROR {figure_error}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
