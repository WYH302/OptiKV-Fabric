#!/usr/bin/env python3
"""RTX 3090 microbenchmarks. Writes only measured rows. Failures stay in logs."""

from __future__ import annotations

import csv
import statistics
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "3090"
OUT = ROOT / "results" / "3090"
WEIGHTS = ROOT / "models" / "weights"


def log(name: str, text: str) -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / name).write_text(text, encoding="utf-8")


def append_rows(path: Path, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["kind", "model", "seq_len", "repeat", "ttft_s", "tpot_s", "prefix_cache", "source", "notes"],
        )
        if new:
            writer.writeheader()
        writer.writerows(rows)


def pcie_bandwidth() -> None:
    import time

    import torch

    if not torch.cuda.is_available():
        log("pcie_error.txt", "torch.cuda.is_available() is False")
        return
    nbytes = 256 * 1024 * 1024
    cpu = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    gpu = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    for _ in range(3):
        gpu.copy_(cpu, non_blocking=True)
        cpu.copy_(gpu, non_blocking=True)
        torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        gpu.copy_(cpu, non_blocking=True)
        torch.cuda.synchronize()
        h2d = time.perf_counter() - start
        start = time.perf_counter()
        cpu.copy_(gpu, non_blocking=True)
        torch.cuda.synchronize()
        d2h = time.perf_counter() - start
        samples.append((nbytes / h2d / 1e9, nbytes / d2h / 1e9))
    h2d = statistics.median(item[0] for item in samples)
    d2h = statistics.median(item[1] for item in samples)
    path = OUT / "pcie_pinned_gbps.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["direction", "median_GBps", "nbytes", "repeats", "source"])
        writer.writeheader()
        writer.writerow({"direction": "h2d", "median_GBps": h2d, "nbytes": nbytes, "repeats": 5, "source": "pytorch_pinned"})
        writer.writerow({"direction": "d2h", "median_GBps": d2h, "nbytes": nbytes, "repeats": 5, "source": "pytorch_pinned"})


def _time_generate(llm, token_ids, max_tokens, repeats, prefix_cache):
    import time

    from vllm import SamplingParams

    prompt = [{"prompt_token_ids": token_ids}]
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    llm.generate(prompt, params)
    rows = []
    for repeat in range(repeats):
        if hasattr(llm, "reset_prefix_cache") and not prefix_cache:
            try:
                llm.reset_prefix_cache()
            except Exception:
                pass
        start = time.perf_counter()
        outputs = llm.generate(prompt, params)
        elapsed = time.perf_counter() - start
        completion = len(outputs[0].outputs[0].token_ids) if outputs and outputs[0].outputs else max_tokens
        tpot = ""
        ttft = elapsed
        if max_tokens > 1 and completion > 1:
            # Elapsed covers prefill plus decode. TTFT is not separated without engine metrics.
            tpot = elapsed / completion
            ttft = ""
        rows.append((repeat, ttft, tpot, completion))
    return rows


def bench_model(model_id: str, local_dir: Path, lengths, max_len: int, smoke: bool) -> None:
    from vllm import LLM

    llm = LLM(
        model=str(local_dir if local_dir.exists() else model_id),
        dtype="bfloat16",
        max_model_len=max_len,
        gpu_memory_utilization=0.88,
        enable_prefix_caching=True,
        download_dir=str(WEIGHTS / "hf-home"),
    )
    raw = OUT / "prefill_raw.csv"
    if smoke:
        rows = _time_generate(llm, [1] * 128, 1, 1, False)
        append_rows(
            raw,
            [
                {
                    "kind": "smoke",
                    "model": model_id,
                    "seq_len": 128,
                    "repeat": rows[0][0],
                    "ttft_s": rows[0][1],
                    "tpot_s": "",
                    "prefix_cache": False,
                    "source": "vllm",
                    "notes": "smoke",
                }
            ],
        )
        return
    for length in lengths:
        try:
            measured = _time_generate(llm, [1] * length, 1, 3, False)
        except Exception as exc:
            log(f"oom_or_error_{length}.txt", traceback.format_exc())
            if "out of memory" in str(exc).lower() or "CUDA" in str(exc):
                break
            break
        for repeat, ttft, _tpot, _n in measured:
            append_rows(
                raw,
                [
                    {
                        "kind": "prefill",
                        "model": model_id,
                        "seq_len": length,
                        "repeat": repeat,
                        "ttft_s": ttft,
                        "tpot_s": "",
                        "prefix_cache": False,
                        "source": "vllm",
                        "notes": "max_tokens=1 wall time",
                    }
                ],
            )
    for length in (4096, 8192):
        try:
            measured = _time_generate(llm, [1] * length, 32, 3, False)
        except Exception:
            log(f"decode_error_{length}.txt", traceback.format_exc())
            continue
        for repeat, _ttft, tpot, ntok in measured:
            append_rows(
                raw,
                [
                    {
                        "kind": "decode",
                        "model": model_id,
                        "seq_len": length,
                        "repeat": repeat,
                        "ttft_s": "",
                        "tpot_s": tpot,
                        "prefix_cache": False,
                        "source": "vllm",
                        "notes": f"wall_time/completion_tokens n={ntok}; not a separated TTFT",
                    }
                ],
            )
    try:
        first = _time_generate(llm, [7] * 2048, 1, 1, True)
        second = _time_generate(llm, [7] * 2048, 1, 1, True)
        append_rows(
            raw,
            [
                {
                    "kind": "prefix_cache",
                    "model": model_id,
                    "seq_len": 2048,
                    "repeat": 0,
                    "ttft_s": first[0][1],
                    "tpot_s": "",
                    "prefix_cache": False,
                    "source": "vllm",
                    "notes": "first sight",
                },
                {
                    "kind": "prefix_cache",
                    "model": model_id,
                    "seq_len": 2048,
                    "repeat": 1,
                    "ttft_s": second[0][1],
                    "tpot_s": "",
                    "prefix_cache": True,
                    "source": "vllm",
                    "notes": "second sight, same token ids",
                },
            ],
        )
    except Exception:
        log("prefix_cache_error.txt", traceback.format_exc())


def write_medians() -> None:
    raw = OUT / "prefill_raw.csv"
    if not raw.is_file():
        return
    groups: dict[tuple, list[float]] = {}
    with raw.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("kind") != "prefill" or row.get("source") != "vllm" or not row.get("ttft_s"):
                continue
            groups.setdefault((row["model"], int(row["seq_len"])), []).append(float(row["ttft_s"]))
    path = OUT / "prefill_median.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "seq_len", "ttft_s", "n_repeats", "source"])
        writer.writeheader()
        for (model, length), values in sorted(groups.items(), key=lambda item: item[0][1]):
            writer.writerow(
                {
                    "model": model,
                    "seq_len": length,
                    "ttft_s": statistics.median(values),
                    "n_repeats": len(values),
                    "source": "vllm",
                }
            )


def held_out_mape() -> None:
    path = OUT / "prefill_median.csv"
    if not path.is_file():
        (OUT / "mape.json").write_text(
            '{"calibration_status":"provisional","held_out_mape":null,"reason":"no measured TTFT"}',
            encoding="utf-8",
        )
        return
    points = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("source") == "vllm" and row.get("ttft_s"):
                points.append((int(row["seq_len"]), float(row["ttft_s"])))
    points.sort()
    if len(points) < 3:
        (OUT / "mape.json").write_text(
            '{"calibration_status":"measured","held_out_mape":null,"reason":"fewer than 3 lengths"}',
            encoding="utf-8",
        )
        return
    hold = points[len(points) // 2]
    train = [item for item in points if item[0] != hold[0]]
    # Piecewise linear in prompt length, the same rule as the simulator.
    pred = train[0][1]
    for (s0, y0), (s1, y1) in zip(train, train[1:]):
        if s0 <= hold[0] <= s1 and s1 != s0:
            weight = (hold[0] - s0) / (s1 - s0)
            pred = y0 * (1.0 - weight) + y1 * weight
            break
        if hold[0] >= s1:
            pred = y1
    mape = abs(pred - hold[1]) / hold[1] if hold[1] else None
    import json

    (OUT / "mape.json").write_text(
        json.dumps(
            {
                "calibration_status": "measured",
                "held_out_seq_len": hold[0],
                "held_out_ttft_s": hold[1],
                "predicted_ttft_s": pred,
                "held_out_mape": mape,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        pcie_bandwidth()
    except Exception:
        log("pcie_error.txt", traceback.format_exc())
    try:
        import vllm  # noqa: F401
    except Exception:
        log("vllm_import_error.txt", traceback.format_exc())
        log("calibration_status.txt", "provisional\nend-to-end vLLM calibration did not start\n")
        return 1
    try:
        bench_model("Qwen/Qwen2.5-0.5B-Instruct", WEIGHTS / "Qwen2.5-0.5B-Instruct", [], 2048, True)
    except Exception:
        log("smoke_error.txt", traceback.format_exc())
        log("calibration_status.txt", "provisional\n0.5B smoke failed\n")
        return 1
    try:
        bench_model(
            "Qwen/Qwen2.5-7B-Instruct",
            WEIGHTS / "Qwen2.5-7B-Instruct",
            [1024, 2048, 4096, 8192, 12288, 16384],
            16384,
            False,
        )
    except Exception:
        log("qwen7b_error.txt", traceback.format_exc())
    write_medians()
    held_out_mape()
    if (OUT / "prefill_median.csv").is_file() and (OUT / "prefill_median.csv").stat().st_size > 40:
        log("calibration_status.txt", "measured\n")
    else:
        log("calibration_status.txt", "provisional\nvLLM did not produce a median TTFT table\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
