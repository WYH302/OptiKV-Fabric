# OptiKV-Fabric

**Physics-Aware Lossless KV Cache Scheduling over Reconfigurable Optical Fabrics for Agentic LLM Serving**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](requirements.txt)
[![Tests](https://img.shields.io/badge/Invariants-23%2F23%20Passed-2ea44f.svg)](tests/)
[![Release](https://img.shields.io/badge/Release-v3.1%20IEEE%20TPDS-0072B2.svg)](https://github.com/WYH302/OptiKV-Fabric/releases)

[English](#overview) | [简体中文](#简体中文简介)

---

## Overview

**OptiKV-Fabric** is a trace-driven, GPU-calibrated simulation and lossless scheduling framework for Key-Value (KV) cache management over Reconfigurable Optical Circuit Switches (OCS) in multi-turn, long-context Large Language Model (LLM) agent serving.

Multi-turn agentic workloads exhibit high strictly causal prefix reuse (55.3%–98.4% across 255,056 unique requests in Mooncake, tokenized LMCache, and FreeInference) and expanding prompt lengths (6K → 24K → 170K+ tokens), overflowing single-GPU HBM. While optical circuit switches provide 45–800 GB/s exclusive circuits via wavelength-agnostic MEMS mirrors without electrical switch-silicon power walls, mechanical switching latencies (25–50 ms) and strict bipartite port exclusivity degrade packet-switched schedulers (*Optical-Naive*) to 3.8%–13.1% duty cycles. Guided by three closed-form roofline crossover laws, OptiKV-Fabric anchors **three tightly coupled modules** on one unified state primitive—a strictly bipartite-conserved **OCS Port Calendar** ($|\mathcal{L}_{\text{pool}}(t)| \le \min(W, P)$)—backed by formal duty-cycle and competitive bounds across both non-oversubscribed ($P \ge W$) and oversubscribed ($P < W$) matchings (**Theorem 1**) plus per-request complexity bounds (**Proposition 1**):

1. **OptiRoute (Matching-Aware Circuit Co-Routing & Exact Abort)**: Rides latched worker–pool circuits ($\tau_{\text{eff}} = 0$), concentrates remote transfers onto latched workers under oversubscription, coalesces remote prefixes pool-first, co-steers against PCIe DRAM queue wait, and triggers exact overlapped recompute abort (Law 1).
2. **GapFetch (Strictly Causal Pause-Windowed Prefetching)**: Hides 25–50 ms MEMS switching off the critical path without future lookahead via a causal EWMA estimator over natural 0.71–2.22 s inter-turn agent pauses, congestion-guarded chronological bursts, and Strata-style in-flight join.
3. **PrefixKeep (Probation-Filtered Lossless Retention)**: Protects sequential prefix chains inside finite HBM via second-observation anchor probation, **Bounded Prefix-GDSF** eviction, strict request-unit pinning, and five-tier lossless spill.

![OptiKV-Fabric Overview, Control Plane, and Honest End-to-End Performance-Power Pareto Frontier](figures/fig1_teaser.png)

---

## Key Verified Results (`v3.1` IEEE TPDS Submission)

All metrics below are directly reproducible from the included RTX 3090 calibration logs (`results/3090/`) and strict bipartite-matching simulation CSVs (`results/sim/`):

- **Live NVIDIA RTX 3090 24 GB Calibration & Unified 45 GB/s HIL Replay (`Qwen2.5-7B-Instruct` BF16, vLLM 0.34.0 & PyTorch 2.14.1+cu126)**:
  - **3.08% analytical roofline MAPE** across 1K–32K cold prefills (6.25% held-out error on 24K–32K).
  - **21.1×–94.5×** multi-length Automatic Prefix Caching speedups across 4K–30K prefixes (`94.5×` at 24K+128, `40.9×` at 30K+512).
  - **40.7× live 24 GB HBM capacity cliff** ($N=1\dots20 \times 8\text{K}$): Turn-2 hit rate collapses **100.0% → 0.0%** and P50 TTFT jumps **43.9 ms → 1,788.1 ms** past vLLM's 4.70 GiB (88K-token) ceiling.
  - **25.91 / 25.93 GB/s** bare-metal 28-layer PCIe Gen4 H2D/D2H KV offloading (`18.21 ms` across 28 interleaved layer copies at 8K) and **$\alpha_{\text{ovl}} = 0.93$** median dual-CUDA-stream overlap (`0.86–0.98`).
  - **Live RTX 3090 vLLM 0.34.0 HIL Replay (`N=12 × 8K`, 5.25 vs. 3.50 GiB HBM, unified 45 GB/s 400G optical link = 10.44 ms wire time, exact pure-prefill Eq. (4) overlap shadow = $0.70 \times 36.38 = 25.5\text{ ms}$)**: recovers **38.04×** from live LRU thrashing (**1,709.0 ms → 44.9 ms** P50), eliminates `CONCUR (W=6)`'s **10.83 s** Wave-2 admission stall (**0.00 s**), eliminates `Optical-Naive`'s **+21.2%** Law 3 breakout (**54.5 ms**), lifts MEMS duty **3.09×** (**29.5% → 90.9%**), measures **16.59 ms (28.32 GB/s decimal / 26.37 GiB/s)** single contiguous 448 MiB async-stream CUDA DMA, and verifies **100.0% bit-exact greedy parity** (128/128 tokens).

- **Full-Scale Multi-Trace & Strict-Matching Cluster Evaluation (H100 Scaled, 8–32 GPUs, 25 ms MEMS vs. Both Contended 25 GB/s & Full 400G Line-Rate 50 GB/s RDMA)**:

| Workload / Benchmark | RDMA-25G+Rt / Pf | RDMA-50G+Rt / Pf (400G Line) | **OptiKV 4:1 ($P{=}W/4$)** | **OptiKV 1:1 ($P{=}W$)** | Key Verified Gain |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Mooncake Full** (23,608 reqs, SiPh 10 µs) | 27.3 / 27.2 ms | 26.5 / 26.5 ms | **25.8 ms (99.7% duty)** | **25.8 ms (99.9% duty)** | Beats both 25 GB/s and 50 GB/s RDMA (62.9% / 17.7% duty on 25 ms MEMS) |
| **LMCache Full** (767 sess / 24,880 reqs, MEMS) | 24.3 / 22.5 ms | 22.7 / 21.1 ms | **18.1 ms (43.2% duty)** | **16.4 ms (84.0% duty)** | **-32.5% vs. 25G Rt / -27.8% vs. 50G Rt (-22.3% vs. 50G Pf)** |
| **FreeInference Stratified** (250 sess / 21,230 reqs) | 344 / 317 ms (1.24–1.28) | 265 / 248 ms (1.38–1.37) | **124 ms / 1.49 req/s (83.4%)** | **121 ms / 1.47 req/s (96.2%)** | **-64.8% vs. 25G / -54.3% vs. 50G P50**, **+6.5%–8.8% raw GP vs. 50G**, **1.31×–2.12× E2E GP/W vs. 50G** |
| **128K Agentic Load** (250 ms SLO, 1–32 req/s) | 0.94 / 1.16 req/s (312 ms) | 4.01 / 4.01 req/s (162 ms) | **4.02 req/s (50 ms, 404 W)** | **4.02 req/s (50 ms, 646 W)** | **3.25× lower P50 & 1.23×–1.96× E2E GP/W vs. 50G** (**5.2×–8.4× vs. 25G**) |
| **5-Fold Disjoint CV (LMCache / FreeInference)** | 19.35 / 108.07 ms | — | — | **17.40 / 81.03 ms** | Paired **-10.06% ± 0.80%** (LMCache) / **-24.21% ± 4.52%** (FreeInference) |
| **Multi-Node Scale-Out** (4→32 GPUs, Strict) | 1.54 / 1.95 req/s (32G) | 12.38 req/s (162 ms P50) | **12.41 req/s (88.5%, 108 sw)** | **12.41 req/s (97.5%, 21 sw)** | **8.04× vs. 25G Rt; 3.25× lower P50 vs. 50G** at 32 GPUs; **4:1 knee** |
| **Single-GPU Eviction** (2 GiB HBM, LMCache) | 26.33% (LRU) / 27.43% (RR) | 30.23% (SIEVE) / 30.24% (S3) | — | **32.87% (Prefix-GDSF)** | **+6.54 pp over LRU / +2.63 pp over S3-FIFO/SIEVE** (94.7% of Belady) |

> **Honest Bandwidth-Matched End-to-End Power Accounting:** All E2E Goodput/Watt metrics charge both switch ports (39.4 W per 400G electrical port vs. 0.39 W per wavelength-agnostic MEMS mirror, Yoon & Yu 2026) **and** pluggable 400G-DR4/FR4 NIC optical transceivers (10 W per transceiver end at matched 4×400G WDM lane counts: 646 W E2E for 200 GB/s 1:1 OCS and 404 W E2E for 200 GB/s 4:1 OCS vs. 790 W E2E for 400G electrical evaluated at both contended 25 GB/s and full 50 GB/s line rate, and 3,162 W E2E for bandwidth-matched 200 GB/s electrical).

---

## Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/WYH302/OptiKV-Fabric.git
cd OptiKV-Fabric
pip install -r requirements.txt
pip install -e .
```

### 2. Run Automated Invariant & Conservation Tests

```bash
pytest tests/ -v
```

Expected output: **23 passed** in <1 second (verifies GQA/MLA/Hybrid KV formulas, RTX 3090 calibration MAPE <10%, zero pinned evictions, strict Oracle & Belady dominance, standing-circuit $\tau = 0$, recompute abort safety, strict causal chronological JIT prefetching, Strata in-flight join, strict bipartite port-matching conservation $|\mathcal{L}_{\text{pool}}(t)| \le \min(W, P)$ including 32-GPU/8-port round-robin port stealing, WDM striping, continuous-batching decode coupling, and Theorem 1's non-oversubscribed and oversubscribed bounds).

### 3. Regenerate All 10 Publication Figures

```bash
python scripts/make_publication_figures.py
```

Figures are written to `figures/fig1_teaser.{pdf,png}` through `figures/fig10_live_3090_hil_replay.{pdf,png}`.

### 4. Re-Run Full `v3.1` Strict-Matching Simulations

Place the public trace datasets under `datasets/`:
- **Mooncake Traces**: [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) (`datasets/mooncake-traces/`)
- **LMCache Agentic Traces**: [LMCache](https://github.com/LMCache/LMCache) (`datasets/lmcache-agentic-traces/`, tokenized via `python scripts/cache_lmcache_tokens.py`)
- **FreeInference Traces**: [Zenodo 10.5281/zenodo.23903440](https://doi.org/10.5281/zenodo.23903440) (`datasets/freeinference-agentic-trace/`)

Then run:

```bash
python scripts/run_v3_strict_matching_suite.py
python scripts/run_v3_strict_stages_2_to_6.py
python scripts/run_v31_round2_additions.py
python scripts/make_publication_figures.py
```

---

## Citation

```bibtex
@article{wu2026optikv,
  author  = {Yonghao Wu and Jiayang Wu and Wensheng Gan and Philip S. Yu},
  title   = {{OptiKV-Fabric}: Physics-Aware Lossless {KV} Cache Scheduling over Reconfigurable Optical Fabrics for Agentic {LLM} Serving},
  journal = {Submitted to IEEE Transactions on Parallel and Distributed Systems},
  year    = {2026}
}
```

**Authors:** Yonghao Wu (First Author, School of Computer Science, South China Normal University, `yonghaowu369@gmail.com`), Jiayang Wu (Second Author, School of Engineering, Westlake University, `csjywu1@gmail.com`), Wensheng Gan (Corresponding Author, Member, IEEE, College of Cyber Security, Jinan University, `wsgan001@gmail.com`), and Philip S. Yu (Life Fellow, IEEE, Department of Computer Science, University of Illinois Chicago, `psyu@uic.edu`).  
**Funding:** Supported in part by the National Natural Science Foundation of China (Grant 62272196) and the Guangzhou Basic and Applied Basic Research Foundation (Grant 2024A04J9971).

---

## 简体中文简介

**OptiKV-Fabric** 是一个面向多轮、长上下文 Agentic LLM 推理的物理感知无损光互连 KV Cache 调度与硬件校准仿真框架（`v3.1`，目标投稿 **IEEE TPDS**）。项目在底层严格贯彻**二分图光端口匹配守恒**（任意时刻同时锁存光池的 Worker 数量严格满足 $|\mathcal{L}_{\text{pool}}(t)| \le \min(W, P)$，抢占端口必断开旧连接并支付重构时延 $\tau$），系统区分非超配（$P \ge W$）与超配（$P < W$）两种物理体制并给出完整理论证明（Theorem 1 与 Proposition 1）。基于 NVIDIA RTX 3090 24 GB 实机微基准校准（3.08% 冷 prefill MAPE、40.7× HBM 容量悬崖、25.9 GB/s 28 层双流重叠 $\alpha_{\text{ovl}}{=}0.93$）与统一 45 GB/s 单口径、纯增量预填充 Eq. (4) 重叠阴影（25.5 ms）的 vLLM 0.34.0 硬件在环回放（38.04× LRU 恢复、消除 CONCUR 10.83 s 准入停顿与 Optical-Naive +21.2% 穿透、3.09× MEMS 占空比与 100.0% 逐 Token 无损一致），在全量 767 会话 LMCache（24,880 请求）与 250 会话分层抽样 FreeInference（21,230 请求）上同时对标 25 GB/s 竞争态与 50 GB/s 满速 400G 线速 RDMA：即便面对满速 50 GB/s 400G RDMA，OptiKV 仍将中位 TTFT 分别降低 **27.8%** 与 **54.3%**（FreeInference 原始 SLO Goodput 提升 +6.5%–8.8%，128K 中位 TTFT 快 **3.25×–3.43×**），计入全部光模块端到端功耗后实现 **1.23×–1.96×**（对标 50 GB/s）与 **5.2×–8.4×**（对标 25 GB/s）端到端 Goodput/Watt，并在 4–32 卡扩展中验证 **4:1 端口超配拐点**。
