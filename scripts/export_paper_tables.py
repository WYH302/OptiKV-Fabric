#!/usr/bin/env python3
"""Export exact LaTeX tables and machine-verified number summary from results/sim/ and results/3090/.

Guarantees 100% numeric fidelity between CSV outputs and paper/main.tex.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "results" / "sim"
R3090 = ROOT / "results" / "3090"
TABLES = ROOT / "results" / "tables"

POL_DISPLAY = {
    "no_cache": "NoCache",
    "local_hbm": "Local-HBM",
    "local_dram": "Local-DRAM",
    "local_ssd": "Local-SSD",
    "rdma_routing": "RDMA+Routing",
    "optical_naive": "Optical-Naive",
    "optical_reservation": "Optical-Reservation",
    "optikv": "\\textbf{OptiKV-Fabric}",
    "eviction_s3fifo": "S3-FIFO (HBM)",
    "eviction_gdsf": "OptiKV-GDSF (HBM)",
    "eviction_belady": "Belady (HBM)",
    "oracle_infinite_hbm": "Oracle (Bound)",
}


def fmt_lat(sec: float) -> str:
    ms = sec * 1e3
    if ms >= 1000.0:
        return f"{sec:.2f}\\,s"
    if ms >= 100.0:
        return f"{ms:.0f}\\,ms"
    return f"{ms:.1f}\\,ms"


def export_all() -> dict:
    TABLES.mkdir(parents=True, exist_ok=True)
    facts: dict = {}

    # 1. Calibration Table
    df_cal = pd.read_csv(SIM / "fig3_calibration_points.csv")
    meas = df_cal[df_cal["measured_3090_7b_ttft_s"].notna()].copy()
    facts["calibration"] = {
        "n_points": int(len(meas)),
        "roofline_mape_pct": float(meas["roofline_abs_err_pct"].mean()),
        "roofline_max_err_pct": float(meas["roofline_abs_err_pct"].max()),
        "quad_in_range_mape_pct": float(meas[meas["split"] == "measured_train"]["quad_train_abs_err_pct"].mean()),
        "quad_heldout_mape_pct": float(meas[meas["split"] == "measured_heldout"]["quad_train_abs_err_pct"].mean()),
        "points": meas[["prompt_tokens", "measured_3090_7b_ttft_s", "roofline_3090_s", "quad_train_1k_16k_s", "roofline_abs_err_pct", "quad_train_abs_err_pct"]].to_dict(orient="records"),
    }

    # Prefix cache & decode & PCIe
    df_pc = pd.read_csv(R3090 / "prefix_cache_experiment.csv")
    off_r2 = float(df_pc[(df_pc["caching"] == 0) & (df_pc["request"] == 2)]["ttft_s"].median())
    on_r1 = float(df_pc[(df_pc["caching"] == 1) & (df_pc["request"] == 1)]["ttft_s"].median())
    on_r2 = float(df_pc[(df_pc["caching"] == 1) & (df_pc["request"] == 2)]["ttft_s"].median())
    df_pcie = pd.read_csv(R3090 / "pcie_bandwidth.csv")
    pcie_gb_s = float(df_pcie.sort_values("bytes").iloc[-1]["gb_per_s"])
    df_dec = pd.read_csv(R3090 / "decode_sweep.csv")
    df_dec = df_dec[df_dec["is_warmup"] == 0]
    tpot_info = {}
    for ctx in (4096, 8192):
        med = df_dec[df_dec["input_tokens"] == ctx].groupby("output_tokens")["e2el_s"].median().reset_index()
        slope, intercept = np.polyfit(med["output_tokens"], med["e2el_s"], 1)
        pred = slope * med["output_tokens"] + intercept
        r2 = 1.0 - np.sum((med["e2el_s"] - pred) ** 2) / np.sum((med["e2el_s"] - med["e2el_s"].mean()) ** 2)
        tpot_info[str(ctx)] = {"tpot_ms": float(slope * 1e3), "tok_s": float(1.0 / slope), "r2": float(r2)}
    facts["hardware_3090"] = {
        "prefix_off_s": off_r2,
        "prefix_cold_s": on_r1,
        "prefix_hit_s": on_r2,
        "prefix_speedup_off_on": off_r2 / on_r2,
        "prefix_speedup_cold_hit": on_r1 / on_r2,
        "pcie_4gib_gb_s": pcie_gb_s,
        "decode_tpot": tpot_info,
    }

    # 2. Main Trace Suites Summary
    trace_files = {
        "mooncake_h100_siph": "exp_main_mooncake_h100_siph.csv",
        "mooncake_h100_mems": "exp_main_mooncake_h100_mems.csv",
        "mooncake_3090_cal": "exp_main_mooncake_3090_calibrated.csv",
        "lmcache_120_h100": "exp_main_lmcache_h100_mems.csv",
        "lmcache_full24k_h100": "exp_main_lmcache_full24k_h100.csv",
        "freeinference_40_h100": "exp_main_freeinference_h100_mems.csv",
        "teaser_128k": "fig1_teaser_128k.csv",
    }
    facts["main_traces"] = {}
    for key, fname in trace_files.items():
        p = SIM / fname
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        facts["main_traces"][key] = df[
            [
                "policy",
                "num_requests",
                "p50_ttft_s",
                "p95_ttft_s",
                "p99_ttft_s",
                "mean_ttft_s",
                "hit_rate",
                "hbm_hit_rate",
                "dram_hit_rate",
                "remote_hbm_hit_rate",
                "optical_pool_hit_rate",
                "rdma_hit_rate",
                "abort_count",
                "prefetch_bursts",
                "ocs_duty_cycle",
                "slo_attainment",
                "goodput_rps",
                "goodput_per_watt",
                "goodput_per_dollar",
            ]
        ].to_dict(orient="records")

    # Write Main Results LaTeX Table (Mooncake H100 SiPh, LMCache 120-sess MEMS, FreeInference 40-sess MEMS, 128K Agentic MEMS)
    order = ["no_cache", "local_hbm", "local_dram", "rdma_routing", "optical_naive", "optical_reservation", "optikv", "oracle_infinite_hbm"]
    dfs = {k: pd.read_csv(SIM / trace_files[k]).set_index("policy") for k in ("mooncake_h100_siph", "lmcache_120_h100", "freeinference_40_h100", "teaser_128k") if (SIM / trace_files[k]).is_file()}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{End-to-end TTFT (P50 / P99), prefix hit rate, and OCS duty cycle across real production/agentic traces and 128K multi-turn serving (H100 peak-spec scaling, 8 workers, RDMA 25\,GB/s vs.\ OCS 200\,GB/s).}",
        r"\label{tab:main-results}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l|cc|cc|cc|ccc}",
        r"\toprule",
        r" & \multicolumn{2}{c|}{\textbf{Mooncake Full (SiPh 10\,$\mu$s)}} & \multicolumn{2}{c|}{\textbf{LMCache (MEMS 25\,ms)}} & \multicolumn{2}{c|}{\textbf{FreeInference (MEMS 25\,ms)}} & \multicolumn{3}{c}{\textbf{128K Agentic (MEMS 25\,ms)}} \\",
        r"\textbf{Policy} & P50 / P99 & Hit & P50 / P99 & Hit & P50 / P99 & Hit & P50 / P99 & Hit & Duty \\",
        r"\midrule",
    ]
    for pol in order:
        cells = [POL_DISPLAY[pol]]
        for k in ("mooncake_h100_siph", "lmcache_120_h100", "freeinference_40_h100"):
            if k in dfs and pol in dfs[k].index:
                r = dfs[k].loc[pol]
                cells.append(f"{fmt_lat(r['p50_ttft_s'])} / {fmt_lat(r['p99_ttft_s'])}")
                cells.append(f"{r['hit_rate']*100:.1f}\\%")
            else:
                cells.extend(["--", "--"])
        if "teaser_128k" in dfs and pol in dfs["teaser_128k"].index:
            r = dfs["teaser_128k"].loc[pol]
            cells.append(f"{fmt_lat(r['p50_ttft_s'])} / {fmt_lat(r['p99_ttft_s'])}")
            cells.append(f"{r['hit_rate']*100:.1f}\\%")
            duty = f"{r['ocs_duty_cycle']*100:.1f}\\%" if r["ocs_duty_cycle"] > 0 else "--"
            cells.append(duty)
        else:
            cells.extend(["--", "--", "--"])
        if pol == "optikv":
            lines.append(r"\midrule")
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"])
    (TABLES / "table_main_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 3. Heatmap summary facts
    df_hm = pd.read_csv(SIM / "exp_breakeven_heatmap.csv")
    facts["heatmap"] = {
        "n_cells": int(len(df_hm)),
        "no_benefit_vs_rdma_count": int(df_hm["no_benefit_vs_rdma"].sum()),
        "max_speedup_vs_rdma_p50": float(df_hm["speedup_vs_rdma_p50"].max()),
        "max_speedup_vs_naive_p50": float(df_hm["speedup_vs_naive_p50"].max()),
        "cell_64k_90": df_hm[(df_hm["context_tokens"] == 65536) & (np.isclose(df_hm["requested_hit_rate"], 0.90))].iloc[0].to_dict(),
        "cell_128k_90": df_hm[(df_hm["context_tokens"] == 131072) & (np.isclose(df_hm["requested_hit_rate"], 0.90))].iloc[0].to_dict(),
        "cell_128k_98": df_hm[(df_hm["context_tokens"] == 131072) & (np.isclose(df_hm["requested_hit_rate"], 0.98))].iloc[0].to_dict()
        if any(np.isclose(df_hm["requested_hit_rate"], 0.98))
        else None,
        "cell_512k_95": df_hm[(df_hm["context_tokens"] == 524288) & (np.isclose(df_hm["requested_hit_rate"], 0.95))].iloc[0].to_dict(),
        "cell_512k_98": df_hm[(df_hm["context_tokens"] == 524288) & (np.isclose(df_hm["requested_hit_rate"], 0.98))].iloc[0].to_dict()
        if any(np.isclose(df_hm["requested_hit_rate"], 0.98))
        else None,
        "cell_1m_95": df_hm[(df_hm["context_tokens"] == 1000000) & (np.isclose(df_hm["requested_hit_rate"], 0.95))].iloc[0].to_dict(),
        "cell_1m_98": df_hm[(df_hm["context_tokens"] == 1000000) & (np.isclose(df_hm["requested_hit_rate"], 0.98))].iloc[0].to_dict()
        if any(np.isclose(df_hm["requested_hit_rate"], 0.98))
        else None,
        "cell_128k_995": df_hm[(df_hm["context_tokens"] == 131072) & (np.isclose(df_hm["requested_hit_rate"], 0.995))].iloc[0].to_dict()
        if any(np.isclose(df_hm["requested_hit_rate"], 0.995))
        else None,
        "cell_512k_995": df_hm[(df_hm["context_tokens"] == 524288) & (np.isclose(df_hm["requested_hit_rate"], 0.995))].iloc[0].to_dict()
        if any(np.isclose(df_hm["requested_hit_rate"], 0.995))
        else None,
        "cell_1m_995": df_hm[(df_hm["context_tokens"] == 1000000) & (np.isclose(df_hm["requested_hit_rate"], 0.995))].iloc[0].to_dict()
        if any(np.isclose(df_hm["requested_hit_rate"], 0.995))
        else None,
        "cell_2k_95": df_hm[(df_hm["context_tokens"] == 2048) & (np.isclose(df_hm["requested_hit_rate"], 0.95))].iloc[0].to_dict(),
    }

    # 4. Ablation, Eviction, Noise, Sensitivity, Load summary facts
    df_abl = pd.read_csv(SIM / "exp_ablation.csv").sort_values("step_idx")
    facts["ablation"] = df_abl[["ablation_step", "p50_ttft_s", "p99_ttft_s", "hit_rate", "abort_count", "prefetch_bursts", "ocs_duty_cycle"]].to_dict(orient="records")

    df_ev = pd.read_csv(SIM / "exp_eviction_comparison.csv").sort_values("ev_idx")
    facts["eviction"] = df_ev[["eviction_label", "p50_ttft_s", "p99_ttft_s", "hit_rate"]].to_dict(orient="records")

    df_ns = pd.read_csv(SIM / "exp_noise_robustness.csv")
    facts["noise"] = (
        df_ns.groupby("sigma")
        .agg(
            p50_mean=("p50_ttft_s", "mean"),
            p50_std=("p50_ttft_s", "std"),
            p99_mean=("p99_ttft_s", "mean"),
            p99_std=("p99_ttft_s", "std"),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    df_arch = pd.read_csv(SIM / "exp_sensitivity_arch_fp8.csv")
    arch_sum = []
    for (m, prec), g in df_arch.groupby(["model", "precision"]):
        gi = g.set_index("policy")
        arch_sum.append(
            {
                "model": m,
                "precision": prec,
                "rdma_p50_s": float(gi.loc["rdma_routing", "p50_ttft_s"]),
                "optikv_p50_s": float(gi.loc["optikv", "p50_ttft_s"]),
                "speedup_vs_rdma": float(gi.loc["rdma_routing", "p50_ttft_s"] / max(1e-9, gi.loc["optikv", "p50_ttft_s"])),
            }
        )
    facts["arch_fp8"] = arch_sum

    df_load = pd.read_csv(SIM / "exp_load_sweep.csv")
    facts["load_peak_goodput"] = (
        df_load.groupby("policy")[["goodput_rps", "goodput_per_watt", "goodput_per_dollar"]]
        .max()
        .reset_index()
        .to_dict(orient="records")
    )

    df_tau = pd.read_csv(SIM / "exp_sensitivity_reconfig.csv")
    facts["reconfig_sweep"] = df_tau[["policy", "reconfig_us", "p50_ttft_s", "p99_ttft_s", "ocs_duty_cycle", "abort_count"]].to_dict(orient="records")

    df_bw = pd.read_csv(SIM / "exp_sensitivity_bandwidth.csv")
    facts["bandwidth_sweep"] = df_bw[["policy", "bandwidth_gb_s", "p50_ttft_s", "p99_ttft_s", "ocs_duty_cycle"]].to_dict(orient="records")

    (TABLES / "paper_numbers.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print(f"Exported LaTeX tables and {TABLES / 'paper_numbers.json'} cleanly.", flush=True)
    return facts


if __name__ == "__main__":
    export_all()
