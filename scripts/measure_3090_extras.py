#!/usr/bin/env python3
"""Extra RTX 3090 measurements: long 7B prefill, decode sweep, prefix-cache timing."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_3090  # noqa: E402
import measure_3090_prefill_7b as prefill7b  # noqa: E402

MODEL_DIR = ROOT / "models" / "Qwen2.5-7B-Instruct"
RES = ROOT / "results" / "3090"
LOG = ROOT / "logs" / "3090"
PARTS = RES / "parts"
RAW = LOG / "prefill_7b_raw.csv"
MEDIAN = RES / "prefill_median_7b.csv"
DECODE = RES / "decode_sweep.csv"
PREFIX_MERGED = RES / "prefix_cache_experiment.csv"
GPU_MEM = 0.88
EXTENDED = (12288, 16384, 24576, 32768)
MAX_MODEL_LEN_CAP = 32768


def log(msg: str) -> None:
    print(msg, flush=True)


def existing_targets(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            int(row["target_tokens"])
            for row in csv.DictReader(handle)
            if row.get("target_tokens") and row.get("note", "") == ""
        }


def open_raw_append() -> tuple[csv.DictWriter, object]:
    exists = RAW.is_file()
    handle = RAW.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=prefill7b.RAW_FIELDS)
    if not exists:
        writer.writeheader()
        handle.flush()
    return writer, handle


def run_extended_prefill() -> dict:
    done = existing_targets(MEDIAN)
    todo = [length for length in EXTENDED if length not in done]
    if not todo:
        log("EXTENDED_PREFILL_SKIP all lengths measured")
        return {"status": "skip", "measured": [], "oom": []}
    if not prefill7b.weights_ready():
        return {"status": "missing_weights", "measured": [], "oom": todo}
    writer, handle = open_raw_append()
    measured: list[int] = []
    oom: list[int] = []
    try:
        for length in todo:
            mlen = length + 1
            if mlen > MAX_MODEL_LEN_CAP + 1:
                log(f"EXTENDED_SKIP length={length} exceeds model context cap")
                prefill7b.write_row(
                    writer,
                    handle,
                    model=prefill7b.MODEL_LABEL,
                    dtype=prefill7b.DTYPE,
                    target_tokens=length,
                    max_model_len=MAX_MODEL_LEN_CAP,
                    note="does_not_fit_max_model_len",
                )
                continue
            if mlen > MAX_MODEL_LEN_CAP:
                os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
            else:
                os.environ.pop("VLLM_ALLOW_LONG_MAX_MODEL_LEN", None)
            mlen = min(mlen, MAX_MODEL_LEN_CAP + 1)
            if length >= mlen:
                log(f"EXTENDED_SKIP length={length} does not fit max_model_len={mlen}")
                continue
            log(f"EXTENDED_ENGINE max_model_len={mlen} length={length}")
            try:
                finished, hit_oom = prefill7b.run_engine(mlen, [length], writer, handle)
            except Exception as exc:  # noqa: BLE001
                log(f"EXTENDED_FAIL len={length} {type(exc).__name__}: {exc}")
                if bench_3090.is_oom(exc):
                    oom.append(length)
                break
            if length in finished:
                measured.append(length)
            if hit_oom:
                oom.extend(hit_oom)
                break
    finally:
        handle.close()
    prefill7b.write_median()
    return {"status": "ok", "measured": measured, "oom": oom}


def run_decode() -> dict:
    if DECODE.is_file() and DECODE.stat().st_size > 200:
        rows = bench_3090.read_csv(DECODE)
        good = [
            row
            for row in rows
            if str(row.get("oom") or "0") == "0" and str(row.get("is_warmup") or "0") != "1"
        ]
        if len(good) >= 9:
            log("DECODE_SKIP existing decode_sweep.csv")
            return {"status": "skip"}
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "bench_3090.py"),
        "decode",
        "--model",
        str(MODEL_DIR),
        "--max-model-len",
        "8320",
        "--inputs",
        "4096,8192",
        "--outputs",
        "16,64,128",
        "--out",
        str(DECODE),
        "--gpu-memory-utilization",
        str(GPU_MEM),
    ]
    log("DECODE_RUN " + " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return {"status": "ok" if proc.returncode == 0 else f"rc_{proc.returncode}"}


def run_prefix() -> dict:
    off = PARTS / "prefix_off.csv"
    on = PARTS / "prefix_on.csv"
    if off.is_file() and on.is_file():
        log("PREFIX_SKIP parts exist")
        _merge_prefix()
        return {"status": "skip"}
    PARTS.mkdir(parents=True, exist_ok=True)
    for caching, out in ((0, off), (1, on)):
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "bench_3090.py"),
            "prefix",
            "--model",
            str(MODEL_DIR),
            "--max-model-len",
            "8192",
            "--prefix-len",
            "4096",
            "--suffix-len",
            "128",
            "--enable-prefix-caching",
            str(caching),
            "--out",
            str(out),
            "--gpu-memory-utilization",
            str(GPU_MEM),
        ]
        log("PREFIX_RUN " + " ".join(cmd))
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            return {"status": f"rc_{proc.returncode}", "caching": caching}
    _merge_prefix()
    return {"status": "ok"}


def _merge_prefix() -> None:
    off = PARTS / "prefix_off.csv"
    on = PARTS / "prefix_on.csv"
    rows = bench_3090.read_csv(off) + bench_3090.read_csv(on)
    if not rows:
        return
    with PREFIX_MERGED.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=bench_3090.PREFIX_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in bench_3090.PREFIX_FIELDS})
    log(f"PREFIX_MERGED {PREFIX_MERGED} rows={len(rows)}")


def decode_summary() -> list[dict]:
    rows = bench_3090.read_csv(DECODE)
    keys: dict[tuple, list[float]] = {}
    for row in rows:
        if str(row.get("is_warmup") or "0") == "1" or str(row.get("oom") or "0") != "0":
            continue
        tpot = bench_3090.fnum(row, "tpot_s")
        if tpot is None:
            continue
        key = (int(float(row["input_tokens"])), int(float(row["output_tokens"])))
        keys.setdefault(key, []).append(tpot)
    summary = []
    for key in sorted(keys):
        if len(keys[key]) < 3:
            continue
        summary.append(
            {
                "input_tokens": key[0],
                "output_tokens": key[1],
                "tpot_s": statistics.median(keys[key]),
                "n": len(keys[key]),
            }
        )
    out = RES / "decode_median.csv"
    if summary:
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
        log(f"DECODE_MEDIAN {out}")
    return summary


def prefix_summary() -> dict:
    rows = bench_3090.read_csv(PREFIX_MERGED) if PREFIX_MERGED.is_file() else []
    if not rows:
        return {}
    def med(caching: str, request: str) -> float | None:
        vals = []
        for row in rows:
            if str(row.get("caching")) != caching or str(row.get("request")) != request:
                continue
            if str(row.get("oom") or "0") != "0":
                continue
            val = bench_3090.fnum(row, "ttft_s")
            if val is not None:
                vals.append(val)
        return statistics.median(vals) if vals else None

    off2 = med("0", "2")
    on2 = med("1", "2")
    on1 = med("1", "1")
    speedup = (off2 / on2) if off2 and on2 else None
    return {
        "request2_ttft_off": off2,
        "request2_ttft_on": on2,
        "request1_ttft_on": on1,
        "speedup_off_over_on": speedup,
    }


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("prefill", "decode", "prefix"), action="append")
    args = parser.parse_args()
    only = set(args.only or ("prefill", "decode", "prefix"))
    report = {}
    if "prefill" in only:
        report["prefill"] = run_extended_prefill()
    if "decode" in only:
        report["decode"] = run_decode()
        report["decode_summary"] = decode_summary()
    if "prefix" in only:
        report["prefix"] = run_prefix()
        report["prefix_summary"] = prefix_summary()
    log("EXTRAS_REPORT " + repr(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())