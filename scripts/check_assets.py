#!/usr/bin/env python3
"""Automated asset, calibration, simulation, figure, and paper audit script."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECKS = {
    "Real Datasets": [
        "datasets/mooncake-traces/mooncake/train-00000-of-00001.parquet",
        "datasets/mooncake-traces/conversation/train-00000-of-00001.parquet",
        "datasets/mooncake-traces/toolagent/train-00000-of-00001.parquet",
        "datasets/lmcache-agentic-traces/tokenized_512.parquet",
        "datasets/freeinference-agentic-trace/week_2026_08_30/release/2026-08-30_2026-09-05/processed/requests.parquet",
        "datasets/freeinference-agentic-trace/long_sessions_40.parquet",
    ],
    "Real RTX 3090 Calibration Logs & CSVs": [
        "results/3090/prefill_median_7b.csv",
        "results/3090/prefix_cache_experiment.csv",
        "results/3090/decode_sweep.csv",
        "results/3090/pcie_bandwidth.csv",
        "configs/hardware/calibration_3090.yaml",
    ],
    "Model Configs (8 Profiles)": [
        "configs/models/qwen2.5-7b-instruct.yaml",
        "configs/models/qwen2.5-7b-instruct-1m.yaml",
        "configs/models/qwen3-8b.yaml",
        "configs/models/llama-3.1-8b.yaml",
        "configs/models/llama-3.1-70b.yaml",
        "configs/models/deepseek-v3.yaml",
        "configs/models/qwen3.5-9b.yaml",
        "configs/models/qwen3-next-80b-a3b.yaml",
    ],
    "Core Simulation Result CSVs": [
        "results/sim/fig1_kv_capacity.csv",
        "results/sim/fig1_teaser_128k.csv",
        "results/sim/fig3_calibration_points.csv",
        "results/sim/exp_main_mooncake_h100_siph.csv",
        "results/sim/exp_main_mooncake_h100_mems.csv",
        "results/sim/exp_main_mooncake_3090_calibrated.csv",
        "results/sim/exp_main_lmcache_h100_mems.csv",
        "results/sim/exp_main_lmcache_full24k_h100.csv",
        "results/sim/exp_main_freeinference_h100_mems.csv",
        "results/sim/exp_breakeven_heatmap.csv",
        "results/sim/exp_sensitivity_reconfig.csv",
        "results/sim/exp_sensitivity_bandwidth.csv",
        "results/sim/exp_sensitivity_links.csv",
        "results/sim/exp_sensitivity_arch_fp8.csv",
        "results/sim/exp_ablation.csv",
        "results/sim/exp_eviction_comparison.csv",
        "results/sim/exp_noise_robustness.csv",
        "results/sim/exp_load_sweep.csv",
    ],
    "Publication Figures (PDF + PNG)": [
        "figures/fig1_teaser.pdf",
        "figures/fig2_trace_characterization.pdf",
        "figures/fig3_prefill_model.pdf",
        "figures/fig4_main_comparison.pdf",
        "figures/fig5_heatmap.pdf",
        "figures/fig6_reconfig_and_cited_cost.pdf",
        "figures/fig7_ablation.pdf",
        "figures/fig8_p99_load.pdf",
    ],
    "Paper Manuscript & PDF": [
        "paper/main.tex",
        "paper/refs.bib",
        "paper/main.pdf",
        "results/tables/paper_numbers.json",
        "results/tables/table_main_results.tex",
    ],
}


def main() -> int:
    missing = []
    for group, paths in CHECKS.items():
        print(f"\n[{group}]")
        for rel in paths:
            p = ROOT / rel
            if p.is_file() and p.stat().st_size > 0:
                print(f"  OK   {rel} ({p.stat().st_size:,} B)")
            else:
                print(f"  MISS {rel}")
                missing.append(rel)
    print("\n" + ("ALL CHECKS PASSED!" if not missing else f"MISSING {len(missing)} FILES: {missing}"))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
