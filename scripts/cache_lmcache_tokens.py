#!/usr/bin/env python3
"""Pre-tokenize LMCache agentic traces once with Qwen2.5-7B-Instruct into 512-token block IDs.

Writes datasets/lmcache-agentic-traces/tokenized_512.parquet so downstream simulations
load all 24,880 multi-turn requests in <0.5s with exact deterministic SHA256 block hashes.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
LM_DIR = ROOT / "datasets" / "lmcache-agentic-traces" / "data"
TOK_DIR = ROOT / "models" / "Qwen2.5-7B-Instruct"
OUT_PATH = ROOT / "datasets" / "lmcache-agentic-traces" / "tokenized_512.parquet"


def _render_messages(messages) -> str:
    if not isinstance(messages, list):
        return str(messages)
    chunks: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, sort_keys=True, ensure_ascii=False)
            chunks.append(f"{message.get('role', '')}\n{content}")
        else:
            chunks.append(str(message))
    return "\n".join(chunks)


def _process_shard(shard_str: str) -> list[dict]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(TOK_DIR), trust_remote_code=False)
    parquet = pq.ParquetFile(shard_str)
    rows: list[dict] = []
    block_tokens = 512
    for batch in parquet.iter_batches(batch_size=64, columns=["session_id", "pre_gap", "output_length", "input"]):
        sessions = batch.column(0).to_pylist()
        gaps = batch.column(1).to_pylist()
        outputs = batch.column(2).to_pylist()
        messages = batch.column(3).to_pylist()
        for session, gap, output, message in zip(sessions, gaps, outputs, messages):
            text = _render_messages(message)
            if not text.strip():
                tids = []
            else:
                tids = tokenizer.encode(text, add_special_tokens=False)
            b_ids: list[str] = []
            b_lens: list[int] = []
            for start in range(0, len(tids), block_tokens):
                chunk = tids[start : start + block_tokens]
                payload = ",".join(str(t) for t in chunk).encode("utf-8")
                h = hashlib.sha256(payload).hexdigest()[:16]
                b_ids.append(f"tok:{h}")
                b_lens.append(len(chunk))
            rows.append(
                {
                    "session_id": str(session),
                    "pre_gap": 0.0 if gap is None else float(gap),
                    "output_length": int(output or 1),
                    "input_tokens": len(tids),
                    "block_ids": b_ids,
                    "block_lens": b_lens,
                }
            )
    return rows


def main() -> int:
    t0 = time.perf_counter()
    shards = sorted(LM_DIR.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No shards in {LM_DIR}")
    print(f"Tokenizing {len(shards)} LMCache shards in parallel...", flush=True)
    all_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=min(5, len(shards))) as pool:
        for shard_rows in pool.map(_process_shard, [str(s) for s in shards]):
            all_rows.extend(shard_rows)
    table = pa.Table.from_pylist(all_rows)
    pq.write_table(table, OUT_PATH, compression="zstd")
    print(f"Wrote {len(all_rows)} rows to {OUT_PATH} in {time.perf_counter()-t0:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
