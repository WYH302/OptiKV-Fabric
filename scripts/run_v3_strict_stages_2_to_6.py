#!/usr/bin/env python3
"""Run remaining Stages [2/6]..[6/6] of the V3 Strict-Bipartite-Matching Suite.
Zero pickled Trace objects across Windows multiprocessing pipes — each worker builds
tiny synthetic traces locally from lightweight spec dicts (100% crash-free & fast!).
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from optikv.metrics import SUMMARY_COLUMNS, write_csv
from optikv.simulator import run_simulation
from run_full_suite import SIM
from run_suites_3_to_6_refined import make_agentic_closedloop
from run_v3_strict_matching_suite import ALL_POLS, _get_trace, _strict_cluster

_LOCAL_SYN_CACHE: dict[tuple, object] = {}


def _run_light_job(job: dict) -> dict:
    if "syn_spec" in job:
        spec = job["syn_spec"]
        k = tuple(sorted(spec.items()))
        if k not in _LOCAL_SYN_CACHE:
            _LOCAL_SYN_CACHE[k] = make_agentic_closedloop(**spec)
        tr = _LOCAL_SYN_CACHE[k]
    else:
        tr = _get_trace(job["trace_key"])
    t0 = time.perf_counter()
    res = run_simulation(job["config"], trace=tr)
    wall_s = time.perf_counter() - t0
    row = dict(res.summary_rows[0])
    if "policy_override" in job:
        row["policy"] = job["policy_override"]
    row.update(job.get("meta", {}))
    row["sim_wall_s"] = wall_s
    row["ctrl_ms_per_req"] = (wall_s * 1e3) / max(1, int(row["num_requests"]))
    return row


def run_batch(pool: ProcessPoolExecutor, jobs: list[dict]) -> list[dict]:
    return list(pool.map(_run_light_job, jobs))


def main() -> int:
    t_all = time.perf_counter()
    print("=== V3 STRICT BIPARTITE MATCHING STAGES [2/6]..[6/6] (workers=8, zero-pickle) ===", flush=True)

    with ProcessPoolExecutor(max_workers=8) as pool:
        # ------------------------------------------------------------------
        # [2/6] Corrected 80-Cell Break-Even Heatmap (2K..1M x 0%..99.5%)
        # ------------------------------------------------------------------
        print("\n[2/6] 80-Cell Break-Even Heatmap (strict matching, fixed Optical Pool admission)...", flush=True)
        t0 = time.perf_counter()
        contexts = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1000000]
        hit_rates = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.995]
        hm_pols = ["no_cache", "rdma_routing", "optical_naive", "optical_reservation", "optikv"]
        cl_hm = _strict_cluster(workers=8, hbm_gib=12.0, dram_gib=24.0, pool_gib=512.0, ocs_bw=200.0, tau_us=25000.0, links=8)
        hm_jobs = []
        for ctx in contexts:
            n_sess = 24 if ctx <= 131072 else (16 if ctx <= 262144 else (12 if ctx <= 524288 else 8))
            gap = 1.5 if ctx <= 262144 else (4.0 if ctx <= 524288 else 8.0)
            for hr in hit_rates:
                syn = dict(name=f"hm-{ctx}-{int(round(hr*1000))}", context_tokens=ctx, hit_rate=hr, n_sessions=n_sess, turns_per_session=8, gap_s=gap, workers=8)
                for pol in hm_pols:
                    hm_jobs.append({
                        "syn_spec": syn,
                        "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_hm, "policies": [pol], "slo_ttft_s": 1.0, "seed": 0},
                        "meta": {"context_tokens": ctx, "requested_hit_rate": hr},
                    })
        hm_rows = run_batch(pool, hm_jobs)
        by_cell: dict[tuple[int, float], dict[str, dict]] = {}
        for r in hm_rows:
            by_cell.setdefault((r["context_tokens"], r["requested_hit_rate"]), {})[r["policy"]] = r
        cells = []
        for (ctx, hr), by_p in sorted(by_cell.items()):
            rdma, opt, naive, resv, nc = by_p["rdma_routing"], by_p["optikv"], by_p["optical_naive"], by_p["optical_reservation"], by_p["no_cache"]
            sp_rdma = rdma["p50_ttft_s"] / max(1e-9, opt["p50_ttft_s"])
            cells.append({
                "context_tokens": ctx,
                "requested_hit_rate": hr,
                "nocache_p50_s": nc["p50_ttft_s"],
                "rdma_p50_s": rdma["p50_ttft_s"],
                "naive_p50_s": naive["p50_ttft_s"],
                "reservation_p50_s": resv["p50_ttft_s"],
                "optikv_p50_s": opt["p50_ttft_s"],
                "nocache_p99_s": nc["p99_ttft_s"],
                "rdma_p99_s": rdma["p99_ttft_s"],
                "naive_p99_s": naive["p99_ttft_s"],
                "optikv_p99_s": opt["p99_ttft_s"],
                "speedup_vs_rdma_p50": sp_rdma,
                "speedup_vs_rdma_p99": rdma["p99_ttft_s"] / max(1e-9, opt["p99_ttft_s"]),
                "speedup_vs_naive_p50": naive["p50_ttft_s"] / max(1e-9, opt["p50_ttft_s"]),
                "speedup_vs_nocache_p50": nc["p50_ttft_s"] / max(1e-9, opt["p50_ttft_s"]),
                "optikv_hit_rate": opt["hit_rate"],
                "optikv_ocs_duty": opt["ocs_duty_cycle"],
                "optikv_aborts": opt["abort_count"],
                "optikv_prefetches": opt["prefetch_bursts"],
                "no_benefit_vs_rdma": sp_rdma <= 1.05,
            })
        write_csv(SIM / "exp_breakeven_heatmap.csv", cells, list(cells[0].keys()))
        min_sp = min(c["speedup_vs_rdma_p50"] for c in cells)
        max_sp = max(c["speedup_vs_rdma_p50"] for c in cells)
        gains = [(c["context_tokens"], c["requested_hit_rate"], round(c["speedup_vs_rdma_p50"], 2)) for c in cells if c["speedup_vs_rdma_p50"] > 1.05]
        print(f"[2/6] Heatmap done in {time.perf_counter()-t0:.1f}s | min={min_sp:.3f}x max={max_sp:.2f}x | gain_cells({len(gains)})={gains}", flush=True)

        # ------------------------------------------------------------------
        # [3/6] Sensitivity (tau, BW, Multi-Arch x FP8), Ablation, Noise, Load Sweep
        # ------------------------------------------------------------------
        print("\n[3/6] Sensitivity, Multi-Arch/FP8, Ablation, Noise, and 1-32 req/s Load Sweep...", flush=True)
        t0 = time.perf_counter()
        sens_pols = ["optikv", "rdma_prefetch", "rdma_routing", "optical_reservation", "optical_naive"]

        j_tau = []
        for tau_us in [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0, 25000.0, 50000.0]:
            cl = _strict_cluster(workers=8, hbm_gib=10.0, dram_gib=16.0, pool_gib=512.0, ocs_bw=200.0, tau_us=tau_us, links=8)
            for p in sens_pols:
                j_tau.append({"trace_key": "bimodal_128k_probes", "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl, "policies": [p], "slo_ttft_s": 0.25, "seed": 0}, "meta": {"reconfig_us": tau_us}})
        r_tau = run_batch(pool, j_tau)
        write_csv(SIM / "exp_sensitivity_reconfig.csv", r_tau, ["reconfig_us"] + SUMMARY_COLUMNS)

        j_bw = []
        for bw in [45.0, 50.0, 100.0, 200.0, 400.0, 800.0]:
            cl = _strict_cluster(workers=8, hbm_gib=10.0, dram_gib=16.0, pool_gib=512.0, ocs_bw=bw, tau_us=25000.0, links=8)
            for p in sens_pols:
                j_bw.append({"trace_key": "bimodal_128k_probes", "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl, "policies": [p], "slo_ttft_s": 0.25, "seed": 0}, "meta": {"bandwidth_gb_s": bw}})
        r_bw = run_batch(pool, j_bw)
        write_csv(SIM / "exp_sensitivity_bandwidth.csv", r_bw, ["bandwidth_gb_s"] + SUMMARY_COLUMNS)

        j_ma = []
        for mname in ["qwen3.5-9b", "qwen2.5-7b-instruct-1m", "deepseek-v3", "qwen3-8b", "llama-3.1-70b"]:
            for db in (2, 1):
                hbm_g = 4.0 if mname == "qwen3.5-9b" else (10.0 if mname == "qwen2.5-7b-instruct-1m" else (12.0 if mname == "deepseek-v3" else (24.0 if mname == "qwen3-8b" else 48.0)))
                dram_g = hbm_g * 1.6
                pool_g = max(512.0, hbm_g * 40.0)
                syn = dict(name=f"ma-{mname}-b{db}", model_name=mname, dtype_bytes=db, context_tokens=131072, hit_rate=255.0 / 256.0, n_sessions=24, turns_per_session=8, gap_s=1.5, bimodal_gaps=True, workers=8)
                cl = _strict_cluster(workers=8, hbm_gib=hbm_g, dram_gib=dram_g, pool_gib=pool_g, ocs_bw=200.0, tau_us=25000.0, links=8)
                for p in ("rdma_routing", "optical_naive", "optikv"):
                    j_ma.append({"syn_spec": syn, "config": {"model": mname, "dtype_bytes": db, "hardware_profile": "h100_peak_spec", "cluster": cl, "policies": [p], "slo_ttft_s": 0.5, "seed": 0}, "meta": {"model_name": mname, "dtype_bytes": db}})
        r_ma = run_batch(pool, j_ma)
        write_csv(SIM / "exp_sensitivity_arch_fp8.csv", r_ma, ["model_name", "dtype_bytes"] + SUMMARY_COLUMNS)

        abl_steps = [
            ("Optical-Naive", "optical_naive", None),
            ("+StandingCircuit", "optikv", {"enable_routing": False, "enable_recompute_fallback": False, "enable_placement": False, "enable_prefetch": False, "enable_eviction": False}),
            ("+OpticalRouting", "optikv", {"enable_routing": True, "enable_recompute_fallback": False, "enable_placement": False, "enable_prefetch": False, "enable_eviction": False}),
            ("+RecomputeAbort", "optikv", {"enable_routing": True, "enable_recompute_fallback": True, "enable_placement": False, "enable_prefetch": False, "enable_eviction": False}),
            ("+TierPlacement", "optikv", {"enable_routing": True, "enable_recompute_fallback": True, "enable_placement": True, "enable_prefetch": False, "enable_eviction": False}),
            ("+WindowedPrefetch", "optikv", {"enable_routing": True, "enable_recompute_fallback": True, "enable_placement": True, "enable_prefetch": True, "enable_eviction": False}),
            ("+PrefixGDSF (Full)", "optikv", {"enable_routing": True, "enable_recompute_fallback": True, "enable_placement": True, "enable_prefetch": True, "enable_eviction": True}),
        ]
        cl_abl = _strict_cluster(workers=8, hbm_gib=10.0, dram_gib=16.0, pool_gib=512.0, ocs_bw=200.0, tau_us=25000.0, links=8)
        j_abl = []
        for idx, (lbl, p, flg) in enumerate(abl_steps):
            cfg = {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_abl, "policies": [p], "slo_ttft_s": 0.25, "seed": 0}
            if flg is not None:
                cfg["optikv"] = flg
            j_abl.append({"trace_key": "bimodal_128k_probes", "config": cfg, "meta": {"order": idx, "ablation_step": lbl}})
        r_abl = sorted(run_batch(pool, j_abl), key=lambda r: r["order"])
        write_csv(SIM / "exp_ablation.csv", r_abl, ["ablation_step"] + SUMMARY_COLUMNS)

        j_ns = []
        for sig in [0.0, 0.05, 0.10, 0.20]:
            for sd in (0, 1, 2):
                j_ns.append({"trace_key": "bimodal_128k_probes", "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_abl, "policies": ["optikv"], "slo_ttft_s": 0.25, "seed": sd, "reuse_noise_std": sig}, "meta": {"sigma": sig, "seed_id": sd}})
        r_ns = run_batch(pool, j_ns)
        write_csv(SIM / "exp_noise_robustness.csv", r_ns, ["sigma", "seed_id"] + SUMMARY_COLUMNS)

        j_ld = []
        cl_ld = _strict_cluster(workers=8, hbm_gib=10.0, dram_gib=16.0, pool_gib=512.0, ocs_bw=200.0, tau_us=25000.0, links=8)
        for rps in [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]:
            syn = dict(name=f"load-128k-r{int(rps)}", context_tokens=131072, hit_rate=255.0 / 256.0, n_sessions=36, turns_per_session=6, workers=8, steady_rps=rps)
            for p in ALL_POLS:
                j_ld.append({"syn_spec": syn, "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_ld, "policies": [p], "slo_ttft_s": 0.25, "seed": 0}, "meta": {"target_rps": rps}})
        r_ld = run_batch(pool, j_ld)
        write_csv(SIM / "exp_load_sweep.csv", r_ld, ["target_rps"] + SUMMARY_COLUMNS)
        print(f"[3/6] Sensitivity + Ablation + Noise + Load done in {time.perf_counter()-t0:.1f}s", flush=True)

        # ------------------------------------------------------------------
        # [4/6] Strict Physics Multi-Node Scale-Out (4-32 GPUs: 1:1 AND 4:1!) + W=16 Port Sweep (P=1..16)
        # ------------------------------------------------------------------
        print("\n[4/6] Strict-Matching 4-32 GPU Weak Scaling (1:1 vs 4:1) + W=16 Port Sweep (P=1..16)...", flush=True)
        t0 = time.perf_counter()
        scale_pols = ["rdma_routing", "concur_rdma", "rdma_prefetch", "optical_naive", "optikv", "oracle_infinite_hbm"]
        j_sc = []
        for w in [4, 8, 16, 32]:
            syn_sc = dict(name=f"scale-128k-w{w}", context_tokens=131072, hit_rate=0.996, n_sessions=4 * w, turns_per_session=6, workers=w, steady_rps=1.5 * w)
            cl_11 = _strict_cluster(workers=w, hbm_gib=10.0, dram_gib=16.0, pool_gib=64.0 * w, ocs_bw=200.0, tau_us=25000.0, links=w, couple_decode=True)
            for pol in scale_pols:
                j_sc.append({"syn_spec": syn_sc, "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_11, "policies": [pol], "slo_ttft_s": 0.25, "seed": 0}, "meta": {"sweep_type": "weak_scale_1to1", "workers": w, "ocs_ports": w, "oversub_ratio": 1.0, "offered_rps": 1.5 * w}})
            p4 = max(1, w // 4)
            cl_41 = _strict_cluster(workers=w, hbm_gib=10.0, dram_gib=16.0, pool_gib=64.0 * w, ocs_bw=200.0, tau_us=25000.0, links=p4, couple_decode=True)
            j_sc.append({"syn_spec": syn_sc, "policy_override": "optikv_4to1", "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_41, "policies": ["optikv"], "slo_ttft_s": 0.25, "seed": 0}, "meta": {"sweep_type": "weak_scale_4to1", "workers": w, "ocs_ports": p4, "oversub_ratio": w / p4, "offered_rps": 1.5 * w}})

        syn_p16 = dict(name="port-128k-w16", context_tokens=131072, hit_rate=0.996, n_sessions=64, turns_per_session=6, workers=16, steady_rps=64.0)
        for p in [1, 2, 4, 8, 16]:
            cl_p = _strict_cluster(workers=16, hbm_gib=10.0, dram_gib=16.0, pool_gib=1024.0, ocs_bw=200.0, tau_us=25000.0, links=p, couple_decode=True)
            for pol in ("optikv", "rdma_prefetch", "rdma_routing"):
                j_sc.append({"syn_spec": syn_p16, "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_p, "policies": [pol], "slo_ttft_s": 0.25, "seed": 0}, "meta": {"sweep_type": "port_ratio_w16", "workers": 16, "ocs_ports": p, "oversub_ratio": 16.0 / p, "offered_rps": 64.0}})

        r_sc = run_batch(pool, j_sc)
        extra_sc = ["sweep_type", "workers", "ocs_ports", "oversub_ratio", "offered_rps", "sim_wall_s", "ctrl_ms_per_req"]
        write_csv(SIM / "exp_v3_cluster_scale.csv", r_sc, extra_sc + SUMMARY_COLUMNS)
        print(f"[4/6] Scale-out + Port Sweep done in {time.perf_counter()-t0:.1f}s:", flush=True)
        for r in r_sc:
            if r["policy"] in ("optikv", "optikv_4to1", "rdma_routing", "rdma_prefetch", "oracle_infinite_hbm"):
                print(f"  {r['sweep_type']:16s} W={r['workers']:2d} P={r['ocs_ports']:2d} ({r['oversub_ratio']:4.1f}:1) | {r['policy']:19s} | P50={r['p50_ttft_s']*1e3:6.1f}ms | GP={r['goodput_rps']:5.2f} | Duty={r['ocs_duty_cycle']*100:5.1f}%", flush=True)

        # ------------------------------------------------------------------
        # [5/6] Strict Physics Fidelity & WDM Striping (1x, 2x, 4x)
        # ------------------------------------------------------------------
        print("\n[5/6] Strict Fidelity (1:1 vs 4:1 Matching + Decode + WDM 1x/2x/4x)...", flush=True)
        t0 = time.perf_counter()
        j_fid = []
        fid_wls = [
            ("lmcache-120", "lm_120", 4.0, 32.0, 512.0, 0.25),
            ("freeinference-50", "fi_50", 16.0, 64.0, 1024.0, 1.0),
            ("agentic-128k-bimodal", "bimodal_128k_probes", 10.0, 16.0, 512.0, 0.25),
        ]
        fid_regimes = [
            ("Match-1:1 (Base)", 8, False, 1),
            ("Match-4:1 (Oversub)", 2, False, 1),
            ("Match-1:1 + Decode", 8, True, 1),
            ("Match-4:1 + Decode", 2, True, 1),
            ("Strict 1:1 + WDM 2x", 8, True, 2),
            ("Strict 1:1 + WDM 4x", 8, True, 4),
        ]
        for wlbl, tkey, hbm_g, dram_g, pool_g, slo in fid_wls:
            for rlbl, links, dec, wdm in fid_regimes:
                cl = _strict_cluster(workers=8, hbm_gib=hbm_g, dram_gib=dram_g, pool_gib=pool_g, ocs_bw=200.0, tau_us=25000.0, links=links, wdm=wdm, couple_decode=dec)
                for pol in ("rdma_routing", "concur_rdma", "rdma_prefetch", "optical_naive", "optikv"):
                    j_fid.append({"trace_key": tkey, "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl, "policies": [pol], "slo_ttft_s": slo, "seed": 0}, "meta": {"workload": wlbl, "regime": rlbl, "ocs_ports": links, "wdm_wavelengths": wdm, "couple_decode": dec}})
        r_fid = run_batch(pool, j_fid)
        write_csv(SIM / "exp_v3_fidelity.csv", r_fid, ["workload", "regime", "ocs_ports", "wdm_wavelengths", "couple_decode"] + SUMMARY_COLUMNS)
        print(f"[5/6] Fidelity done in {time.perf_counter()-t0:.1f}s", flush=True)

        # ------------------------------------------------------------------
        # [6/6] 5-Fold Disjoint Cross-Validation (All 767 LMCache + 250 FreeInference)
        # ------------------------------------------------------------------
        print("\n[6/6] 5-Fold Disjoint Session CV (767 LMCache + 250 FreeInference)...", flush=True)
        t0 = time.perf_counter()
        cv_pols = ["rdma_routing", "concur_rdma", "rdma_prefetch", "optical_naive", "optikv", "oracle_infinite_hbm"]
        j_cv_lm, j_cv_fi = [], []
        for fold in range(5):
            cl_lm = _strict_cluster(workers=8, hbm_gib=4.0, dram_gib=32.0, pool_gib=512.0, ocs_bw=200.0, tau_us=25000.0, links=8)
            cl_fi = _strict_cluster(workers=8, hbm_gib=16.0, dram_gib=64.0, pool_gib=1024.0, ocs_bw=200.0, tau_us=25000.0, links=8)
            for pol in cv_pols:
                j_cv_lm.append({"trace_key": f"lm_fold_{fold}", "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_lm, "policies": [pol], "slo_ttft_s": 0.25, "seed": 0}, "meta": {"fold": fold}})
                j_cv_fi.append({"trace_key": f"fi_fold_{fold}", "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_fi, "policies": [pol], "slo_ttft_s": 1.0, "seed": 0}, "meta": {"fold": fold}})
        r_cv_lm = run_batch(pool, j_cv_lm)
        r_cv_fi = run_batch(pool, j_cv_fi)
        write_csv(SIM / "exp_5fold_cv_lmcache.csv", r_cv_lm, ["fold"] + SUMMARY_COLUMNS)
        write_csv(SIM / "exp_5fold_cv_freeinference.csv", r_cv_fi, ["fold"] + SUMMARY_COLUMNS)
        print(f"[6/6] 5-Fold CV done in {time.perf_counter()-t0:.1f}s", flush=True)

    print(f"\n=== ALL STAGES [2/6]..[6/6] COMPLETED IN {time.perf_counter()-t_all:.1f}s ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
