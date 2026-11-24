"""Request traces. Cache objects keep the trace's native block IDs.

Mooncake blocks are 512 tokens. FreeInference blocks are 16 tokens. This
loader never invents extra block IDs and never resamples a block to another size.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from optikv.modelspec import ModelSpec

MOONCAKE_BLOCK_TOKENS = 512
FREEINFERENCE_BLOCK_TOKENS = 16


@dataclass(frozen=True)
class BlockRef:
    block_id: str
    num_tokens: int


@dataclass(frozen=True)
class Request:
    request_id: str
    arrival_s: float
    session_id: str
    turn_index: int
    pre_gap_s: float
    blocks: tuple[BlockRef, ...]
    output_tokens: int = 1
    assigned_worker: int | None = None

    @property
    def input_tokens(self) -> int:
        return sum(block.num_tokens for block in self.blocks)


@dataclass
class Trace:
    name: str
    kind: str
    block_tokens: int
    requests: list[Request] = field(default_factory=list)
    seed: int = 0
    skipped: int = 0
    block_note: str = ""


def stable_hash(text: str) -> int:
    """Process-stable FNV-1a. Do not use Python's salted hash()."""
    value = 2166136261
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def split_native_blocks(
    block_ids: list,
    block_tokens: int,
    input_tokens: int | None,
) -> list[BlockRef]:
    """Assign tokens onto the given IDs. Refuse to create IDs the trace did not list."""
    ids = [str(block_id) for block_id in block_ids]
    if not ids:
        if input_tokens not in (None, 0):
            raise ValueError("input tokens are present but the trace has no block ids")
        return []
    if input_tokens is None or input_tokens == len(ids) * block_tokens:
        return [BlockRef(block_id, block_tokens) for block_id in ids]
    if input_tokens < 0:
        raise ValueError("negative input length")
    full = block_tokens * (len(ids) - 1)
    last = input_tokens - full
    if full < input_tokens <= full + block_tokens and last > 0:
        blocks = [BlockRef(block_id, block_tokens) for block_id in ids[:-1]]
        blocks.append(BlockRef(ids[-1], last))
        return blocks
    raise ValueError(
        f"input_tokens={input_tokens} is inconsistent with {len(ids)} native "
        f"blocks of {block_tokens} tokens; refusing to resample"
    )


def generate_synthetic(
    model: ModelSpec,
    *,
    name: str = "synthetic",
    n_sessions: int = 2,
    turns_per_session: int = 2,
    prefix_blocks: int = 2,
    suffix_blocks: int = 1,
    block_tokens: int = 64,
    gap_s: float = 0.0,
    inter_arrival_s: float = 1000.0,
    output_tokens: int = 1,
    seed: int = 0,
    pin_workers: bool = False,
    worker_count: int = 1,
) -> Trace:
    """Build a deterministic trace and reject contexts above the model cap.

    Prefix block IDs are stable across turns of a session. Suffix IDs are
    per turn. `seed` selects the RNG stream; the layout itself is a pure
    function of the arguments, so a fixed seed repeats exactly.
    """
    per_request = (prefix_blocks + suffix_blocks) * block_tokens
    model.check_context(per_request)
    rng = random.Random(seed)
    _ = rng.random()
    requests: list[Request] = []
    clock = 0.0
    for session in range(n_sessions):
        prefix = tuple(
            BlockRef(f"s{session}-p{index}", block_tokens) for index in range(prefix_blocks)
        )
        for turn in range(turns_per_session):
            suffix = tuple(
                BlockRef(f"s{session}-t{turn}-x{index}", block_tokens)
                for index in range(suffix_blocks)
            )
            blocks = prefix + suffix
            model.check_context(sum(block.num_tokens for block in blocks))
            assigned = (session % worker_count) if pin_workers else None
            requests.append(
                Request(
                    request_id=f"s{session}-t{turn}",
                    arrival_s=clock,
                    session_id=f"session-{session}",
                    turn_index=turn,
                    pre_gap_s=0.0 if turn == 0 else gap_s,
                    blocks=blocks,
                    output_tokens=output_tokens,
                    assigned_worker=assigned,
                )
            )
            clock += inter_arrival_s if turn == 0 else max(inter_arrival_s, gap_s)
    return Trace(name=name, kind="synthetic", block_tokens=block_tokens, requests=requests, seed=seed)


def load_trace(
    path: Path | str,
    kind: str = "auto",
    block_tokens: int | None = None,
    namespace: str | None = None,
    name: str | None = None,
) -> Trace:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"trace not found: {path}. The simulator does not download datasets."
        )
    detected = kind
    if path.suffix == ".parquet" and detected in {"auto", "mooncake", "freeinference", "lmcache"}:
        return _load_parquet(path, detected, block_tokens, namespace, name or path.stem)
    if detected == "auto":
        detected = _detect_kind(path)
    if detected == "mooncake":
        return _load_mooncake(path, name or path.stem)
    if detected == "freeinference":
        tokens = block_tokens or FREEINFERENCE_BLOCK_TOKENS
        return _load_freeinference(path, tokens, namespace, name or path.stem)
    if detected == "normalized":
        if block_tokens is None:
            raise ValueError("normalized traces require block_tokens")
        return _load_normalized(path, block_tokens, name or path.stem)
    raise ValueError(f"unknown trace kind {detected}")


def _detect_kind(path: Path) -> str:
    if path.suffix == ".parquet":
        return "freeinference"
    if path.suffix == ".csv":
        return "normalized"
    first = ""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                first = line
                break
    if "hash_ids" in first:
        return "mooncake"
    return "normalized"


def _load_parquet(path: Path, kind: str, block_tokens: int | None, namespace: str | None, name: str) -> Trace:
    import pyarrow.parquet as pq

    names = set(pq.read_schema(path).names)
    if kind in {"auto", "mooncake"} and "hash_ids" in names and "block_ids" not in names:
        return _load_mooncake_parquet(path, name)
    if kind in {"auto", "lmcache"} and "pre_gap" in names and "input" in names and "block_ids" not in names:
        return _load_lmcache_parquet(path, name)
    if "block_ids" not in names:
        raise ValueError(f"{path} has columns {sorted(names)} and no block_ids or hash_ids")
    tokens = block_tokens or FREEINFERENCE_BLOCK_TOKENS
    return _load_freeinference(path, tokens, namespace, name)


def _load_mooncake_parquet(path: Path, name: str) -> Trace:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["timestamp", "input_length", "output_length", "hash_ids"])
    requests: list[Request] = []
    skipped = 0
    timestamps = table.column("timestamp")
    inputs = table.column("input_length")
    outputs = table.column("output_length")
    hashes = table.column("hash_ids")
    for index in range(table.num_rows):
        try:
            blocks = split_native_blocks(
                hashes[index].as_py() or [],
                MOONCAKE_BLOCK_TOKENS,
                int(inputs[index].as_py()),
            )
        except ValueError:
            skipped += 1
            continue
        requests.append(
            Request(
                request_id=str(index),
                arrival_s=float(timestamps[index].as_py()) / 1000.0,
                session_id=f"mooncake-{index}",
                turn_index=0,
                pre_gap_s=0.0,
                blocks=tuple(blocks),
                output_tokens=int(outputs[index].as_py() or 1),
            )
        )
    requests.sort(key=lambda request: (request.arrival_s, request.request_id))
    return Trace(
        name=name,
        kind="mooncake",
        block_tokens=MOONCAKE_BLOCK_TOKENS,
        requests=requests,
        skipped=skipped,
        block_note="native hash_ids, 512-token blocks; partial last block keeps the trace length",
    )


def _render_messages(messages) -> bytes:
    if not isinstance(messages, list):
        return str(messages).encode("utf-8")
    chunks: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, sort_keys=True, ensure_ascii=False)
            chunks.append(f"{message.get('role', '')}\n{content}")
        else:
            chunks.append(str(message))
    return "\n".join(chunks).encode("utf-8")


def _block_id_for_token_chunk(chunk: list[int]) -> str:
    payload = ",".join(str(token) for token in chunk).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _tokenizer_chunk_blocks(messages, tokenizer, block_tokens: int = MOONCAKE_BLOCK_TOKENS) -> list[BlockRef]:
    """512-token blocks from the Qwen tokenizer; block ids hash each chunk's token ids."""
    text = _render_messages(messages).decode("utf-8")
    if not text.strip():
        return []
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []
    blocks: list[BlockRef] = []
    for start in range(0, len(token_ids), block_tokens):
        chunk = token_ids[start : start + block_tokens]
        blocks.append(BlockRef(f"tok:{_block_id_for_token_chunk(chunk)}", len(chunk)))
    return blocks


def _derived_prefix_blocks(text: bytes, block_tokens: int = 512) -> list[BlockRef]:
    """SHA1 of each cumulative byte prefix. Not a provider tokenizer."""
    if not text:
        return []
    step = block_tokens * 4
    blocks: list[BlockRef] = []
    end = 0
    while end < len(text):
        end = min(len(text), end + step)
        digest = hashlib.sha1(text[:end]).hexdigest()[:16]
        nbytes = end - (len(blocks) * step)
        tokens = max(1, int(round(nbytes / 4.0)))
        blocks.append(BlockRef(f"lm:{digest}", tokens))
    return blocks


def _load_lmcache_parquet(
    path: Path,
    name: str,
    *,
    tokenizer=None,
    block_builder=None,
) -> Trace:
    import pyarrow.parquet as pq

    if block_builder is None:
        if tokenizer is not None:
            block_builder = lambda message: _tokenizer_chunk_blocks(message, tokenizer)
        else:
            block_builder = lambda message: _derived_prefix_blocks(_render_messages(message))

    parquet = pq.ParquetFile(path)
    requests: list[Request] = []
    ready: dict[str, float] = {}
    turns: dict[str, int] = {}
    new_session_clock = 0.0
    for batch in parquet.iter_batches(batch_size=32, columns=["session_id", "pre_gap", "output_length", "input"]):
        sessions = batch.column(0).to_pylist()
        gaps = batch.column(1).to_pylist()
        outputs = batch.column(2).to_pylist()
        messages = batch.column(3).to_pylist()
        for session, gap, output, message in zip(sessions, gaps, outputs, messages):
            session_id = str(session)
            blocks = block_builder(message)
            if session_id in ready:
                arrival = ready[session_id] + float(gap or 0.0)
            else:
                arrival = new_session_clock
                new_session_clock += 0.05
            turn = turns.get(session_id, 0)
            requests.append(
                Request(
                    request_id=f"{session_id}-{turn}",
                    arrival_s=arrival,
                    session_id=session_id,
                    turn_index=turn,
                    pre_gap_s=0.0 if turn == 0 else float(gap or 0.0),
                    blocks=tuple(blocks),
                    output_tokens=int(output or 1),
                )
            )
            ready[session_id] = arrival
            turns[session_id] = turn + 1
    requests.sort(key=lambda request: (request.arrival_s, request.request_id))
    note = (
        "tokenizer-derived 512-token blocks from rendered messages; "
        "block ids hash each chunk's token ids. The release has no hash_ids."
        if tokenizer is not None
        else (
            "derived SHA1 prefixes of rendered messages in 2048-byte chunks "
            "(about 512 tokens at 4 bytes/token). The release has no hash_ids."
        )
    )
    return Trace(
        name=name,
        kind="lmcache",
        block_tokens=MOONCAKE_BLOCK_TOKENS,
        requests=requests,
        block_note=note,
    )


def load_lmcache_directory(lm_dir: Path, tokenizer_path: Path | str, name: str = "lmcache-tokenized") -> Trace:
    """Load every LMCache parquet shard with tokenizer-derived 512-token blocks."""
    from transformers import AutoTokenizer

    lm_dir = Path(lm_dir)
    shards = sorted(lm_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no LMCache parquet shards in {lm_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=False)
    requests: list[Request] = []
    skipped = 0
    for shard in shards:
        partial = _load_lmcache_parquet(shard, shard.stem, tokenizer=tokenizer)
        requests.extend(partial.requests)
        skipped += partial.skipped
    requests.sort(key=lambda request: (request.arrival_s, request.request_id))
    note = (
        "tokenizer-derived 512-token blocks from rendered messages; "
        "block ids hash each chunk's token ids. The release has no hash_ids."
    )
    return Trace(
        name=name,
        kind="lmcache",
        block_tokens=MOONCAKE_BLOCK_TOKENS,
        requests=requests,
        skipped=skipped,
        block_note=note,
    )


def _load_mooncake(path: Path, name: str) -> Trace:
    requests: list[Request] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            blocks = split_native_blocks(
                row["hash_ids"],
                MOONCAKE_BLOCK_TOKENS,
                int(row["input_length"]),
            )
            requests.append(
                Request(
                    request_id=str(row.get("request_id", index)),
                    arrival_s=float(row["timestamp"]) / 1000.0,
                    session_id=str(row.get("session_id", f"mooncake-{index}")),
                    turn_index=int(row.get("turn_index", 0)),
                    pre_gap_s=float(row.get("pre_gap_s", 0.0)),
                    blocks=tuple(blocks),
                    output_tokens=int(row.get("output_length", 1)),
                    assigned_worker=row.get("assigned_worker"),
                )
            )
    requests.sort(key=lambda request: (request.arrival_s, request.request_id))
    return Trace(name=name, kind="mooncake", block_tokens=MOONCAKE_BLOCK_TOKENS, requests=requests)


def _load_freeinference(
    path: Path,
    block_tokens: int,
    namespace: str | None,
    name: str,
) -> Trace:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    columns = set(table.column_names)
    if "block_ids" not in columns:
        raise ValueError("freeinference parquet is missing block_ids; refusing to synthesize ids")
    n = table.num_rows
    pending: list[tuple[float, dict]] = []
    for index in range(n):
        raw_ids = table.column("block_ids")[index].as_py() or []
        week = None
        if "week" in columns:
            week = table.column("week")[index].as_py()
        prefix = namespace or (str(week) if week else None)
        ids = [f"{prefix}:{block_id}" if prefix else str(block_id) for block_id in raw_ids]
        prompt = _column_int(table, index, ("input_tokens", "prompt_tokens", "n_input_tokens"))
        if prompt is None and "provider_tokens" in columns:
            payload = table.column("provider_tokens")[index].as_py()
            if isinstance(payload, dict) and payload.get("prompt") is not None:
                prompt = int(payload["prompt"])
        blocks = split_native_blocks(ids, block_tokens, prompt)
        start_ms = _column_float(table, index, ("start_ms", "timestamp_ms"))
        end_ms = _column_float(table, index, ("end_ms",))
        session = _column_str(table, index, ("session_id", "session")) or f"free-{index}"
        turn = _column_int(table, index, ("turn_index", "step")) or 0
        output = _column_int(table, index, ("output_tokens", "completion_tokens")) or 1
        pending.append(
            (
                start_ms or 0.0,
                {
                    "request_id": _column_str(table, index, ("request_id",)) or str(index),
                    "session_id": session,
                    "turn_index": int(turn),
                    "start_ms": start_ms or 0.0,
                    "end_ms": end_ms,
                    "blocks": tuple(blocks),
                    "output_tokens": int(output),
                },
            )
        )
    pending.sort(key=lambda item: (item[1]["session_id"], item[0], item[1]["request_id"]))
    last_end: dict[str, float] = {}
    requests: list[Request] = []
    for _start, row in pending:
        previous = last_end.get(row["session_id"])
        gap_s = 0.0
        if previous is not None:
            gap_s = max(0.0, (row["start_ms"] - previous) / 1000.0)
        if row["end_ms"] is not None:
            last_end[row["session_id"]] = row["end_ms"]
        requests.append(
            Request(
                request_id=row["request_id"],
                arrival_s=row["start_ms"] / 1000.0,
                session_id=row["session_id"],
                turn_index=row["turn_index"],
                pre_gap_s=gap_s,
                blocks=row["blocks"],
                output_tokens=row["output_tokens"],
            )
        )
    requests.sort(key=lambda request: (request.arrival_s, request.request_id))
    return Trace(name=name, kind="freeinference", block_tokens=block_tokens, requests=requests)


def _load_normalized(path: Path, block_tokens: int, name: str) -> Trace:
    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    requests: list[Request] = []
    for index, row in enumerate(rows):
        raw_ids = row.get("block_ids", "")
        if isinstance(raw_ids, str):
            ids = [part for part in raw_ids.replace(",", ";").split(";") if part]
        else:
            ids = list(raw_ids)
        input_tokens = row.get("input_tokens")
        input_tokens = int(input_tokens) if input_tokens not in (None, "") else None
        per_block = int(row.get("block_tokens") or block_tokens)
        blocks = split_native_blocks(ids, per_block, input_tokens)
        if "arrival_s" in row and row["arrival_s"] not in (None, ""):
            arrival = float(row["arrival_s"])
        else:
            arrival = float(row.get("timestamp_ms", 0.0)) / 1000.0
        assigned = row.get("assigned_worker")
        assigned = int(assigned) if assigned not in (None, "") else None
        requests.append(
            Request(
                request_id=str(row.get("request_id") or index),
                arrival_s=arrival,
                session_id=str(row.get("session_id") or f"norm-{index}"),
                turn_index=int(row.get("turn_index") or 0),
                pre_gap_s=float(row.get("pre_gap_s") or 0.0),
                blocks=tuple(blocks),
                output_tokens=int(row.get("output_tokens") or 1),
                assigned_worker=assigned,
            )
        )
    requests.sort(key=lambda request: (request.arrival_s, request.request_id))
    return Trace(name=name, kind="normalized", block_tokens=block_tokens, requests=requests)


def _column_int(table, index: int, names: tuple[str, ...]) -> int | None:
    for name in names:
        if name in table.column_names:
            value = table.column(name)[index].as_py()
            if value is not None:
                return int(value)
    return None


def _column_float(table, index: int, names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in table.column_names:
            value = table.column(name)[index].as_py()
            if value is not None:
                return float(value)
    return None


def _column_str(table, index: int, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in table.column_names:
            value = table.column(name)[index].as_py()
            if value is not None:
                return str(value)
    return None


def trace_from_config(raw: dict, model: ModelSpec) -> Trace:
    kind = raw.get("kind", "synthetic")
    if kind == "synthetic":
        return generate_synthetic(
            model,
            name=str(raw.get("name", "synthetic")),
            n_sessions=int(raw.get("n_sessions", 2)),
            turns_per_session=int(raw.get("turns_per_session", raw.get("turns", 2))),
            prefix_blocks=int(raw.get("prefix_blocks", 2)),
            suffix_blocks=int(raw.get("suffix_blocks", 1)),
            block_tokens=int(raw.get("block_tokens", 64)),
            gap_s=float(raw.get("gap_s", 0.0)),
            inter_arrival_s=float(raw.get("inter_arrival_s", 1000.0)),
            output_tokens=int(raw.get("output_tokens", 1)),
            seed=int(raw.get("seed", 0)),
            pin_workers=bool(raw.get("pin_workers", False)),
            worker_count=int(raw.get("worker_count", 1)),
        )
    if "path" not in raw or not raw["path"]:
        raise FileNotFoundError("trace.path is required for non-synthetic traces")
    return load_trace(
        raw["path"],
        kind=kind,
        block_tokens=raw.get("block_tokens"),
        namespace=raw.get("namespace"),
        name=raw.get("name"),
    )
