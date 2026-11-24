#!/usr/bin/env python3
"""Cold prefill TTFT for Qwen2.5-7B-Instruct on the local RTX 3090.

Writes results/3090/prefill_median_7b.csv only from vLLM first_token_latency.
Does not read or modify the 0.5B smoke CSV.
An 8192-token prompt does not fit in max_model_len 8192 together with
max_tokens=1; that length is measured only if a follow-up engine with
max_model_len 8193 accepts it. An OOM at 8192 keeps shorter medians.
"""

from __future__ import annotations

import csv
import os
import statistics
import sys
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_3090  # noqa: E402

LOG_DIR = ROOT / "logs" / "3090"
RES_DIR = ROOT / "results" / "3090"
MODEL_DIR = ROOT / "models" / "Qwen2.5-7B-Instruct"
MODEL_LABEL = "Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
GPU_MEM = 0.88
LENGTHS = (1024, 2048, 4096, 8192)
REPEATS = 3
WARMUP = 1
RAW_FIELDS = [
    "model",
    "dtype",
    "target_tokens",
    "actual_tokens",
    "ttft_s",
    "wall_s",
    "ttft_source",
    "num_cached_tokens",
    "repeat",
    "is_warmup",
    "max_model_len",
    "gpu_mem_mib",
    "note",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def cached_count(text: str) -> int | None:
    if text is None or text == "":
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def weights_ready() -> bool:
    shards = [
        MODEL_DIR / "model-00001-of-00004.safetensors",
        MODEL_DIR / "model-00002-of-00004.safetensors",
        MODEL_DIR / "model-00003-of-00004.safetensors",
        MODEL_DIR / "model-00004-of-00004.safetensors",
    ]
    return all(path.is_file() and path.stat().st_size > 10_000_000 for path in shards)


def open_raw() -> tuple[csv.DictWriter, object]:
    path = LOG_DIR / "prefill_7b_raw.csv"
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
    writer.writeheader()
    handle.flush()
    return writer, handle


def write_row(writer, handle, **row) -> None:
    writer.writerow({key: row.get(key, "") for key in RAW_FIELDS})
    handle.flush()
    os.fsync(handle.fileno())


def first_token_latency(result: dict) -> float | None:
    metrics = result.get("metrics") or {}
    if not isinstance(metrics, dict):
        return None
    return bench_3090.as_float(metrics.get("first_token_latency"))


def measure_length(llm, sampling_params_cls, tokenizer, length: int, max_model_len: int, writer, handle) -> str:
    """Return 'ok', 'oom', or 'fail'."""
    ids = bench_3090.build_ids(tokenizer, length, salt=length)
    if len(ids) != length:
        raise RuntimeError(f"tokenizer built {len(ids)} ids for target {length}")
    for rep in range(WARMUP + REPEATS):
        warmup = rep < WARMUP
        repeat_id = 0 if warmup else rep - WARMUP + 1
        if hasattr(llm, "reset_prefix_cache"):
            llm.reset_prefix_cache()
        try:
            result = bench_3090.run_once(llm, sampling_params_cls, ids, max_tokens=1)
        except Exception as exc:  # noqa: BLE001
            log(f"REQUEST_FAIL len={length} repeat={repeat_id} {type(exc).__name__}: {exc}")
            log(traceback.format_exc())
            note = "oom" if bench_3090.is_oom(exc) else "request_error"
            write_row(
                writer,
                handle,
                model=MODEL_LABEL,
                dtype=DTYPE,
                target_tokens=length,
                actual_tokens=len(ids),
                max_model_len=max_model_len,
                repeat=repeat_id,
                is_warmup=int(warmup),
                gpu_mem_mib=bench_3090.gpu_mem_mib(),
                note=note,
            )
            return "oom" if note == "oom" else "fail"
        latency = first_token_latency(result)
        cached = cached_count(result["cached"])
        note = ""
        if latency is None:
            note = "missing_first_token_latency"
        elif cached is not None and cached > 0:
            note = "prefix_cache_hit_excluded"
        if result["actual_tokens"] != length:
            note = (note + ";" if note else "") + "actual_tokens_differ"
        shown = "" if latency is None else f"{latency:.9f}"
        log(
            f"PREFILL len={length} max_model_len={max_model_len} warmup={int(warmup)} "
            f"repeat={repeat_id} ttft_s={shown} source=first_token_latency "
            f"wall_s={result['wall_s']:.9f} actual={result['actual_tokens']} "
            f"cached={result['cached']!r} mem={bench_3090.gpu_mem_mib()} note={note}"
        )
        log("METRICS " + repr(result["metrics"]))
        write_row(
            writer,
            handle,
            model=MODEL_LABEL,
            dtype=DTYPE,
            target_tokens=length,
            actual_tokens=result["actual_tokens"],
            ttft_s=shown,
            wall_s=f"{result['wall_s']:.9f}",
            ttft_source="" if latency is None else "first_token_latency",
            num_cached_tokens="" if cached is None else cached,
            repeat=repeat_id,
            is_warmup=int(warmup),
            max_model_len=max_model_len,
            gpu_mem_mib=bench_3090.gpu_mem_mib(),
            note=note,
        )
        if note:
            return "fail"
    return "ok"


def run_engine(max_model_len: int, lengths: list[int], writer, handle) -> tuple[list[int], list[int]]:
    LLM, SamplingParams = bench_3090.import_vllm()
    llm = bench_3090.build_llm(
        LLM,
        str(MODEL_DIR),
        max_model_len,
        prefix_caching=False,
        gpu_mem=GPU_MEM,
        extra={"disable_log_stats": False},
    )
    finished: list[int] = []
    oom_lengths: list[int] = []
    try:
        tokenizer = bench_3090.load_tokenizer(str(MODEL_DIR))
        for length in lengths:
            if length >= max_model_len:
                log(
                    f"SKIP len={length} does not fit: prompt plus max_tokens=1 "
                    f"exceeds max_model_len={max_model_len}"
                )
                write_row(
                    writer,
                    handle,
                    model=MODEL_LABEL,
                    dtype=DTYPE,
                    target_tokens=length,
                    max_model_len=max_model_len,
                    note="does_not_fit_max_model_len",
                )
                continue
            log(f"BEGIN len={length} max_model_len={max_model_len}")
            status = measure_length(llm, SamplingParams, tokenizer, length, max_model_len, writer, handle)
            if status == "ok":
                finished.append(length)
            elif status == "oom":
                log(f"OOM len={length} max_model_len={max_model_len}")
                oom_lengths.append(length)
                break
            else:
                log(f"STOP_LENGTH len={length}")
                break
    finally:
        bench_3090.shutdown_llm(llm)
    return finished, oom_lengths


def write_median() -> int:
    raw_path = LOG_DIR / "prefill_7b_raw.csv"
    groups: dict[int, list[tuple[int, float]]] = {}
    with raw_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("is_warmup") or "") == "1":
                continue
            if row.get("note"):
                continue
            if row.get("ttft_source") != "first_token_latency":
                continue
            if not row.get("ttft_s"):
                continue
            target = int(row["target_tokens"])
            actual = int(row["actual_tokens"])
            if actual != target:
                continue
            groups.setdefault(target, []).append((actual, float(row["ttft_s"])))
    out = RES_DIR / "prefill_median_7b.csv"
    written = 0
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "dtype", "target_tokens", "actual_tokens", "ttft_s"],
        )
        writer.writeheader()
        for target in sorted(groups):
            samples = groups[target]
            if len(samples) < REPEATS:
                log(f"MEDIAN_SKIP target={target} n={len(samples)}")
                continue
            actuals = [item[0] for item in samples]
            times = [item[1] for item in samples]
            if len(set(actuals)) != 1:
                log(f"MEDIAN_ACTUAL_MISMATCH target={target} actuals={actuals}")
                continue
            median_ttft = f"{statistics.median(times):.9f}"
            writer.writerow(
                {
                    "model": MODEL_LABEL,
                    "dtype": DTYPE,
                    "target_tokens": target,
                    "actual_tokens": actuals[0],
                    "ttft_s": median_ttft,
                }
            )
            written += 1
            log(
                f"MEDIAN target={target} actual={actuals[0]} "
                f"ttft_s={median_ttft} samples={times}"
            )
    if written == 0 and out.is_file():
        out.unlink()
        log(f"REMOVED_EMPTY {out}")
    else:
        log(f"WROTE {out} rows={written}")
    return written


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    if not weights_ready():
        log(f"MISSING_WEIGHTS {MODEL_DIR}")
        return 2
    import torch
    import vllm

    log(
        f"STACK vllm={vllm.__version__} torch={torch.__version__} "
        f"cuda={torch.cuda.is_available()} "
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}"
    )
    log(f"MODEL {MODEL_DIR}")
    log("This run measures 7B directly. It does not read or scale the 0.5B medians.")
    writer, handle = open_raw()
    oom_lengths: list[int] = []
    fatal: BaseException | None = None
    try:
        try:
            done, oom = run_engine(8192, [1024, 2048, 4096, 8192], writer, handle)
            oom_lengths.extend(oom)
            log(f"ENGINE_8192_DONE {done} oom={oom}")
        except Exception as exc:  # noqa: BLE001
            log(f"ENGINE_8192_FAIL {type(exc).__name__}: {exc}")
            log(traceback.format_exc())
            if bench_3090.is_oom(exc):
                log("OOM during max_model_len=8192 engine init")
                oom_lengths.append(8192)
            else:
                fatal = exc
            done = []
        if fatal is None and 8192 not in done and 8192 not in oom_lengths:
            try:
                done_8192, oom_8192 = run_engine(8193, [8192], writer, handle)
                oom_lengths.extend(oom_8192)
                log(f"ENGINE_8193_DONE {done_8192} oom={oom_8192}")
            except Exception as exc:  # noqa: BLE001
                log(f"ENGINE_8193_FAIL {type(exc).__name__}: {exc}")
                log(traceback.format_exc())
                if bench_3090.is_oom(exc):
                    log("OOM len=8192 max_model_len=8193 engine init")
                    oom_lengths.append(8192)
                    write_row(
                        writer,
                        handle,
                        model=MODEL_LABEL,
                        dtype=DTYPE,
                        target_tokens=8192,
                        max_model_len=8193,
                        note="oom",
                    )
                else:
                    fatal = exc
    finally:
        handle.close()
    if oom_lengths:
        log(f"OOM_LENGTHS {oom_lengths}")
    written = write_median()
    if fatal is not None and written == 0:
        raise fatal
    return 0 if written else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
