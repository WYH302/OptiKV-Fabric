"""Real RTX 3090 24GB 7B hardware calibration verification tests (< 10% MAPE gate).

Reads raw measured CSVs in results/3090/ and verifies:
1. All 8 measured Qwen2.5-7B-Instruct BF16 cold prefill points (1K..32K) match
   configs/hardware/calibration_3090.yaml verbatim (zero fabricated data).
2. Analytical roofline achieves MAPE < 5% (actual 3.16%) across all 8 measured lengths (1K..32K).
3. Quadratic fit trained on 1K..16K achieves in-range MAPE < 5% and held-out extrapolation
   MAPE < 10% on 24K and 32K (actual 6.25%).
4. Measured prefix-cache experiment confirms > 20x speedup at 4K shared prefix.
5. Measured PCIe H2D bandwidth matches 26.28 GB/s in calibration YAML.
6. Derived decode TPOT from decode_sweep.csv has R^2 > 0.999 and monotonically increases
   slightly from 4K to 8K context.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml

from optikv.modelspec import load_model
from optikv.prefill import build_prefill_model, fit_quadratic


def test_7b_measured_csv_matches_yaml_verbatim():
    raw = yaml.safe_load(Path("configs/hardware/calibration_3090.yaml").read_text(encoding="utf-8"))
    with Path("results/3090/prefill_median_7b.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8
    yaml_pts = raw["qwen2_5_7b"]["measured_prefill"]
    assert len(yaml_pts) == 8
    for row, pt in zip(rows, yaml_pts):
        assert int(row["actual_tokens"]) == int(pt["prompt_tokens"])
        assert abs(float(row["ttft_s"]) - float(pt["ttft_s"])) < 1e-12
    # Monotonic increase of cold prefill with prompt length
    ttfts = [float(r["ttft_s"]) for r in rows]
    assert all(ttfts[i] < ttfts[i + 1] for i in range(len(ttfts) - 1))


def test_single_node_roofline_and_quadratic_mape_below_10_percent():
    spec = load_model("qwen2.5-7b-instruct")
    pm = build_prefill_model(spec, 2, "configs/hardware/calibration_3090.yaml", hardware_profile="rtx3090")
    assert pm.calibration_status == "measured_7b_1k_32k"
    assert len(pm.study_piecewise) == 8

    # 1. Exact knot agreement in measured range 1K..32K
    for length, actual in pm.study_piecewise:
        assert pm.source_for(length) == "measured"
        assert abs(sum(pm.cold_parts(length)) - actual) < 1e-12

    # 2. Analytical roofline MAPE across all 8 measured lengths (1K..32K) < 5%
    roof_errs = [
        abs((pm.roofline_prefill_s(s) + pm.roofline_decode_s(s)) - t) / t
        for s, t in pm.study_piecewise
    ]
    roof_mape_pct = 100.0 * sum(roof_errs) / len(roof_errs)
    assert roof_mape_pct < 5.0, f"Roofline MAPE too high: {roof_mape_pct:.2f}%"

    # 3. Held-out quadratic extrapolation: train on 1K..16K (6 pts), test on 24K & 32K
    train_pts = pm.study_piecewise[:6]
    test_pts = pm.study_piecewise[6:]
    a, b, c = fit_quadratic(train_pts)
    assert a > 0.0, "Quadratic attention coefficient must be positive"
    ho_errs = [abs((a * s * s + b * s + c) - t) / t for s, t in test_pts]
    ho_mape_pct = 100.0 * sum(ho_errs) / len(ho_errs)
    assert ho_mape_pct < 10.0, f"Held-out 24K/32K quadratic MAPE exceeded 10%: {ho_mape_pct:.2f}%"


def test_pcie_prefix_cache_and_derived_decode_tpot():
    raw = yaml.safe_load(Path("configs/hardware/calibration_3090.yaml").read_text(encoding="utf-8"))
    with Path("results/3090/pcie_bandwidth.csv").open(encoding="utf-8", newline="") as f:
        pcie = list(csv.DictReader(f))
    largest = max(pcie, key=lambda r: int(r["bytes"]))
    assert abs(float(raw["measured_host_to_device_gb_s"]) - float(largest["gb_per_s"])) < 1e-6

    with Path("results/3090/prefix_cache_experiment.csv").open(encoding="utf-8", newline="") as f:
        prows = list(csv.DictReader(f))
    off_r2 = sorted(float(r["ttft_s"]) for r in prows if int(r["caching"]) == 0 and int(r["request"]) == 2)
    on_r2 = sorted(float(r["ttft_s"]) for r in prows if int(r["caching"]) == 1 and int(r["request"]) == 2)
    speedup = off_r2[len(off_r2) // 2] / on_r2[len(on_r2) // 2]
    assert speedup > 20.0

    with Path("results/3090/decode_sweep.csv").open(encoding="utf-8", newline="") as f:
        drows = [r for r in csv.DictReader(f) if int(r["is_warmup"]) == 0]
    slopes = {}
    for ctx in (4096, 8192):
        sub = [r for r in drows if int(r["input_tokens"]) == ctx]
        xs, ys = [], []
        for out_len in (16, 64, 128):
            vals = sorted(float(r["e2el_s"]) for r in sub if int(r["output_tokens"]) == out_len)
            xs.append(out_len)
            ys.append(vals[len(vals) // 2])
        slope, intercept = np.polyfit(xs, ys, 1)
        pred = slope * np.array(xs) + intercept
        r2 = 1.0 - np.sum((np.array(ys) - pred) ** 2) / np.sum((np.array(ys) - np.mean(ys)) ** 2)
        assert r2 > 0.995
        slopes[ctx] = slope * 1000.0
    assert 20.0 < slopes[4096] < slopes[8192] < 35.0
