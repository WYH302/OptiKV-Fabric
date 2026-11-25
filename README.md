# OptiKV-Fabric

**Interconnect-Aware Lossless KV Cache Scheduling over Reconfigurable Optical Circuit Switches for Agentic LLM Serving**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](requirements.txt)
[![Tests](https://img.shields.io/badge/Invariants-22%2F22%20Passed-2ea44f.svg)](tests/)

[English](#overview) | [简体中文](#简体中文简介)

---

## Overview

**OptiKV-Fabric** is a trace-driven, GPU-calibrated simulation and scheduling framework studying lossless Key-Value (KV) cache management over Reconfigurable Optical Circuit Switches (OCS) for multi-turn, long-context Large Language Model (LLM) agents.

Multi-turn agentic workloads exhibit high causal prefix reuse (55.3%–98.4% across Mooncake, LMCache, and FreeInference) and growing prompt lengths (24K–170K+ tokens), causing active KV caches to overflow single-GPU HBM. While optical circuit switches provide high exclusive bandwidth (45–800 GB/s) at ~100× lower switch port power than electrical 400G packet switches, mechanical MEMS switching latencies (25–50 ms) and port exclusivity degrade interconnect-blind schedulers (*Optical-Naive*). Guided by three closed-form roofline crossover laws, OptiKV-Fabric centers on a single state primitive—the **OCS Port Calendar**—to resolve three coupled scheduling questions in closed loop:

1. **Optical Transfer vs. Local DRAM vs. Exact Recomputation**: Standing-circuit pool-first coalescing exploits latched ports ($\tau_{\text{eff}} = 0$), queue-aware co-steering arbitrates optical links against local PCIe DRAM, and closed-form selective recompute abort cancels remote fetches whenever overlapped remote TTFT exceeds local delta prefill.
2. **Strictly Causal Hiding of 25–50 ms MEMS Switching**: Chronological EWMA Just-In-Time (JIT) burst prefetching schedules asynchronous optical transfers inside natural inter-turn agent pauses (0.71–2.22 s median) using strictly past session history, attaching early arrivals via Strata-style in-flight join to sustain **75%–93%** MEMS duty cycles (and **76%–81%** under strict bipartite transceiver contention).
3. **Prefix-Dependency-Aware Retention**: Probation-filtered causal reuse prediction (promoting anchors only on second observation) drives **Bounded Prefix-GDSF** eviction and five-tier lossless spill placement under request-unit block pinning.

![OptiKV-Fabric Architecture and Speedup Frontier](figures/fig1_teaser.png)

---

## Key Verified Results (V3)

All numbers below are reproducible directly from the included RTX 3090 calibration logs (`results/3090/`) and strictly causal simulation summaries (`results/sim/`):

- **Real-GPU Calibration (NVIDIA RTX 3090 24 GB, `Qwen2.5-7B-Instruct`, BF16, vLLM 0.34.0 & Bare-Metal PyTorch 2.14.1+cu126)**:
  - **3.08% analytical roofline MAPE** across 1K–32K cold prefills (6.25% held-out extrapolation MAPE on 24K/32K).
  - **21.1×–94.5×** measured multi-length Automatic Prefix Caching speedups across 4K–30K cached prefixes (`24.16×` on controlled 4K+128 ON/OFF, `94.47×` at 24K+128, `40.88×` at 30K+512).
  - **Live 40.7× single-GPU HBM capacity cliff** ($N=1\dots20 \times 8\text{K}$ concurrent sessions): Turn-2 hit rate collapses from **100.0% → 0.0%** and TTFT jumps **43.9 ms → 1,788.1 ms** past vLLM's 4.70 GiB (88K-token) KV ceiling.
  - **25.91 / 25.93 GB/s** bare-metal 28-layer PCIe Gen4 H2D/D2H KV offloading and **$\alpha_{\text{ovl}} = 0.926$** median dual-CUDA-stream overlap (range `0.860–0.980`), plus **24.68 / 25.24 ms/tok** OLS-derived decode TPOT ($R^2 > 0.999$) and **3.15×–5.13×** speedup vs. RDMA+Routing (**1.69×–2.86×** vs. causal RDMA+Prefetch) across the full $\alpha_{\text{ovl}} \in [0, 1]$ sweep.
- **End-to-End Multi-Trace & 2024–2026 Frontier Baseline Evaluation (8 Workers, H100 Peak-Spec Scaling, 25 ms MEMS OCS vs. 25 GB/s RDMA)**:

| Workload / Benchmark | RDMA+Routing | CONCUR+RDMA ('26) | Causal RDMA+Prefetch | **OptiKV-Fabric** | Key Improvement |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **LMCache Agentic** (120 sess, MEMS 25 ms) | 25.3 ms P50 | 25.1 ms P50 | 23.3 ms P50 | **18.5 ms P50 (92.6% duty)** | **-26.8% vs. RDMA+Rout / -20.3% vs. RDMA+Pref**, **101.0× GP/W** |
| **LMCache Full** (767 sess, 24,880 reqs) | 25.7 ms P50 | 25.1 ms P50 | 22.8 ms P50 | **22.3 ms P50 (85.6% duty)** | **-13.3% vs. RDMA+Rout / -2.2% vs. RDMA+Pref**, **100.7× GP/W** |
| **FreeInference Long** (25 sess, MEMS 25 ms) | 91.3 ms P50 | 87.2 ms P50 | 80.2 ms P50 | **45.1 ms P50 (91.2% duty)** | **-50.6% vs. RDMA+Rout / -43.7% vs. RDMA+Pref**, **100.9× GP/W** |
| **128K Agentic Teaser** (MEMS 25 ms) | 169.2 ms P50 | 169.2 ms P50 | 91.3 ms P50 | **36.3 ms P50 (90.3% duty)** | **4.66× vs. RDMA+Rout / 2.52× vs. RDMA+Pref** (exact Oracle floor) |
| **128K Interactive Load** (250 ms SLO, 1–32 req/s) | 1.55 req/s peak | 1.55 req/s peak | 1.83 req/s peak | **6.58 req/s (Oracle floor)** | **4.25× vs. RDMA/CONCUR**, **3.60× vs. RDMA+Pref**, **364× GP/W** |
| **5-Fold Disjoint CV (LMCache, 767 sess)** | 24.54 ± 1.22 ms | 24.84 ± 0.92 ms | 22.71 ± 0.81 ms | **18.38 ± 0.41 ms (91.6% duty)** | **25.03% ± 2.06%** vs. RDMA+Rout / **19.04% ± 1.52%** vs. RDMA+Pref |
| **5-Fold Disjoint CV (FreeInference, 40 sess)** | 74.83 ± 28.57 ms | 74.94 ± 28.70 ms | 69.81 ± 28.79 ms | **58.42 ± 26.70 ms (79.7% duty)** | **23.78% ± 8.46%** vs. RDMA+Rout / **16.86% ± 11.07%** vs. RDMA+Pref |
| **Strict Physics (Bipartite + Decode)** | 81.3 ms (FI) | 83.4 ms (FI) | 72.1 ms (FI) | **45.4 ms (76.1% duty)** | Holds **18.3 / 45.4 / 36.3 ms** across LMCache / FI / 128K; WDM 4× cuts transfer 42%–44% |
| **Multi-Node Scale-Out** (4→32 GPUs, Strict) | 1.93 req/s (32G) | 1.93 req/s (32G) | 1.95 req/s (32G) | **12.41 req/s (Oracle, 97.5% duty)** | **6.43× vs. RDMA** (650× GP/W) at 32 GPUs; verified **4:1 GPU:OCS-port provisioning knee** |
| **Single-GPU HBM Eviction** (2 GiB, LMCache) | 26.33% (LRU) | 27.43% (RR-Evict) | 30.23% (SIEVE) / 30.24% (S3-FIFO) | **32.87% (Prefix-GDSF)** | **+6.54 pp over LRU / +5.44 pp over RR-Evict / +2.63 pp over S3-FIFO** (94.7% of Belady) |

---

## Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/WYH302/OptiKV-Fabric.git
cd OptiKV-Fabric
pip install -r requirements.txt
pip install -e .
```

### 2. Run Automated Physical Conservation & Calibration Tests

```bash
pytest tests/ -v
```

Expected output: **22 passed** in <1 second (verifies GQA/MLA/Hybrid KV formulas, RTX 3090 calibration MAPE <10%, zero pinned evictions, strict Oracle & Belady dominance, standing-circuit $\tau = 0$, recompute abort safety, strict causal chronological JIT prefetching, Strata in-flight join, bipartite source–destination port exclusivity, WDM striping, decode batch-queue coupling, CONCUR admission control, and queue-aware DRAM co-steering).

### 3. Regenerate All Publication Figures from Included Results

```bash
python scripts/make_publication_figures.py
```

Figures are written to `figures/fig1_teaser.{pdf,png}` through `figures/fig9_fidelity_and_scaling.{pdf,png}`.

### 4. Re-Run Full Trace Simulations, 5-Fold CV & Strict Physics Fidelity

Place the public trace datasets under `datasets/`:
- **Mooncake Traces**: [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) (`datasets/mooncake-traces/`)
- **LMCache Agentic Traces**: [LMCache](https://github.com/LMCache/LMCache) (`datasets/lmcache-agentic-traces/`, tokenized via `python scripts/cache_lmcache_tokens.py`)
- **FreeInference Traces**: [Zenodo 10.5281/zenodo.23903440](https://doi.org/10.5281/zenodo.23903440) (`datasets/freeinference-agentic-trace/`)

Then run:

```bash
python scripts/run_full_suite.py
python scripts/run_suites_3_to_6_refined.py
python scripts/run_stage2_audit_and_cv.py
python scripts/run_overlap_sensitivity.py
python scripts/run_v3_fidelity_and_baselines.py
python scripts/run_v3_cluster_scale.py
python scripts/make_publication_figures.py
```

---

## Repository Structure

```text
OptiKV-Fabric/
├── src/
│   ├── optikv/               # Core simulator: OCS Port Calendar (bipartite + WDM), causal JIT heap, decode queue
│   └── policies/             # 15 scheduling & eviction policies (SIEVE, RR-Evict, S3-FIFO, CONCUR+RDMA, RDMA+Prefetch, OptiKV)
├── configs/                  # Model KV profiles (Qwen2.5/3/3.5, Llama-3.1, DeepSeek-V3) & RTX 3090 calibration YAML
├── scripts/                  # Calibration, live RTX 3090 vLLM replay, full simulation, 5-fold CV, fidelity & figure scripts
├── tests/                    # 22 automated physical conservation & calibration accuracy checks
├── results/
│   ├── 3090/                 # Measured RTX 3090 7B prefill, decode, APC, HBM cliff, PCIe Gen4 DMA & live replay CSVs
│   ├── sim/                  # Experiment, sensitivity, ablation, load-sweep, ROI, 5-fold CV & V3 fidelity CSVs
│   ├── tables/               # Consolidated paper metrics JSON & LaTeX tables
│   └── trace_stats/          # Profiled trace summary statistics
├── figures/                  # Rendered publication figures (PNG)
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

---

## Engineering Insights & Pitfalls Overcome

During hardware calibration and simulator verification, we encountered four non-obvious systems pitfalls that directly shaped OptiKV-Fabric's design and evaluation:

1. **Chronological JIT Event-Heap Causality & Fair Fabric-Scaled Lead Times**: Scheduling prefetches using future `pre_gap_s` leaks arrival timestamps, while launching prefetches retroactively pollutes the OCS calendar and capping lead times at a fixed 100 ms unfairly handicaps 25 GB/s electrical `RDMA+Prefetch`. Moving both optical and electrical prefetches into a single arrival-ordered `heapq` driven strictly by past-session EWMA pauses (`obs_gap >= 0.15 s`), bandwidth-scaled lead times, and Strata-style in-flight join (`ready_at_s`) eliminates all lookahead while preserving fair apples-to-apples comparisons.
2. **Windows WDDM Dual-CUDA-Stream Batching Artifact (`0.04 → 0.93` Overlap)**: In initial 28-layer KV offload microbenchmarks on bare-metal Windows CUDA, enqueueing all 28 H2D copies before kernel dispatch caused the WDDM driver to batch transfers serially before layer 0 compute began, collapsing apparent overlap to `0.04–0.07`. Switching to interleaved per-layer producer–consumer CUDA events with explicit `stream.query()` flushes exposed true hardware PCIe Gen4 pipelining (**25.91/25.93 GB/s**, median $\alpha_{\text{ovl}} = 0.926$, range `0.860–0.980`).
3. **Bounded Prefix-GDSF & Probation-Filtered Anchors**: Unbounded marginal-cost GDSF inverts prefix priority (protecting quadratic suffixes over roots), while unbounded downstream cost can hoard stale multi-session roots and first-seen anchors pollute HBM on stateless Mooncake chatbots. Bounding the root-to-leaf ratio to `[1.0, 2.0]` and promoting anchors strictly on second observation lifts Mooncake local HBM hit rate from `29.7% → 34.8%` (beating LRU `33.1%`) and achieves **32.87%** single-GPU LMCache hit rate (+6.54 pp over LRU, +5.44 pp over RR-Evict, +2.64 pp over SIEVE, **94.7%** of offline Belady).
4. **Live 4.70 GiB HBM Cyclical LRU Thrashing Cliff (`40.7×` TTFT Jump)**: Measuring round-robin multi-session 8K prompts on a real 24 GB RTX 3090 running vLLM 0.34.0 revealed that once concurrent working sets exceed the 4.70 GiB (88K-token) paged KV pool from $N=10$ to $N=12$ sessions, hit rate drops sharply from **100.0% to 0.0%**, driving a **40.7×** median TTFT spike (`43.9 ms → 1,788.1 ms`).

---

## Citation

If you find this codebase or dataset analysis helpful for your research, please consider citing:

```bibtex
@article{wu2026optikv,
  author  = {Yonghao Wu},
  title   = {{OptiKV-Fabric}: Interconnect-Aware Lossless {KV} Cache Scheduling over Reconfigurable Optical Circuit Switches for Agentic {LLM} Serving},
  journal = {Manuscript under review},
  year    = {2026}
}
```

Contact: Yonghao Wu (`yonghaowu369@gmail.com`), School of Electronic Science and Engineering (School of Microelectronics), South China Normal University.

---

## 简体中文简介

**OptiKV-Fabric** 是一个面向多轮、长上下文 Agentic LLM 推理场景的光互连感知无损 KV Cache 调度与硬件校准仿真框架。项目基于真实 NVIDIA RTX 3090 24 GB（`Qwen2.5-7B-Instruct` BF16，vLLM 0.34.0 与裸机 28 层 CUDA 卸载/重叠实测）微基准数据（1K–32K 冷 prefill 解析 Roofline 误差 **3.08% MAPE**、4K–30K 多长度热命中最高 **94.5×** 加速、24 GB HBM 显存墙 **40.7×** LRU 驱逐悬崖、28 层 PCIe Gen4 **25.9 GB/s** 与双流重叠 $\alpha_{\text{ovl}}{=}0.93$），并在 Mooncake、LMCache（24,880 请求）与 FreeInference（190,544 请求）公开真实轨迹上与涵盖 `SIEVE` (NSDI'24)、`RR-Evict` (2026)、`CONCUR+RDMA` (ICML'26) 与严格因果 `RDMA+Prefetch` 在内的 15 种策略同台对比，并通过严格源–宿二分端口冲突、Decode 连续批处理队列耦合及 WDM 多波长条带化压力验证。

### 核心设计
围绕单一状态原语 **OCS Port Calendar（光交换端口日历）**，基于三条解析 Roofline 交叉定律协同求解三个耦合问题：
- **何时用光 vs. 本地 DRAM vs. 精确重算**：驻留光路与共享光池优先聚合（$\tau_{\text{eff}} = 0$）、排队感知光–电 DRAM 协同分流与闭合形式无损选择性重算中止；
- **严格因果隐藏 25–50 ms 机械 MEMS 切换延迟**：基于历史停顿 EWMA 的时间序事件堆 JIT 突发预取与 Strata 风格在途预取合流（零未来泄露），在 25 ms MEMS 下将光交换占空比提升至 **75%–93%**（严格二分端口约束下仍达 **76.1%–80.8%**）；
- **前缀链式失效下的无损保留**：试用期过滤锚点与有界 **Prefix-GDSF** 驱逐驱动五层无损溢出（单卡 2 GiB HBM 命中率 **32.87%**，超越 LRU +6.54 pp、RR-Evict +5.44 pp、SIEVE +2.64 pp 与 S3-FIFO +2.63 pp，逼近离线最优 Belady 34.72% 的 **94.7%**）。

### 快速复现
```bash
pip install -r requirements.txt
pip install -e .
pytest tests/ -v                                # 22/22 物理守恒、严格因果与校准不变量测试
python scripts/make_publication_figures.py      # 一键重新生成全部论文图表
```

欢迎通过 Issue 或邮件（`yonghaowu369@gmail.com`）交流指正。
