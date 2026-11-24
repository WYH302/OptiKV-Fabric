# LMCache tokenizer-derived replay

Finished: 2026-11-21

## Block derivation

- Dataset: `datasets/lmcache-agentic-traces/data` (five parquet shards)
- Tokenizer: `models/Qwen2.5-7B-Instruct`
- Messages rendered with the same `_render_messages` helper as `src/optikv/trace.py`
- Each prompt split into 512-token chunks; block id is `tok:` plus the first 16 hex digits of SHA256 over comma-joined token ids in the chunk
- Scope: `results/sim/lmcache_simulation_scope.json`
- Loader: `load_lmcache_directory()` in `src/optikv/trace.py`
- Runner: `scripts/run_lmcache_tokenized.py` (`.venv-sim` on Windows)

## Simulation

- Requests: 24880 (full release), skipped 0
- Input tokens min / median / max: 330 / 24086 / 103637
- Model clock: provisional roofline (`qwen2.5-7b-instruct-1m`)
- Policies: no_cache, local_hbm, rdma_routing, optical_naive, optical_reservation, optikv, oracle_infinite_hbm

## Outputs

| file | description |
|------|-------------|
| `results/sim/lmcache_eval.csv` | P50/P99/mean TTFT and hit rate by policy |
| `results/sim/lmcache_requests.csv` | per-request rows |
| `results/sim/lmcache_summary.csv` | run summary |
| `results/sim/lmcache_component_medians.csv` | median queue/transfer/prefill/TTFT |

## P50 TTFT (seconds) from `lmcache_eval.csv`

| policy | p50_ttft_s | hit_rate |
|--------|------------|----------|
| no_cache | 7354.790958284952 | 0.0 |
| local_hbm | 3971.4702480448404 | 0.49641461152622046 |
| rdma_routing | 3967.385202574365 | 0.49558681416998435 |
| optical_naive | 4722.196529742138 | 0.500544554093084 |
| optical_reservation | 3921.1999230101146 | 0.5003742103094913 |
| optikv | 5762.787450666319 | 0.2003896155134097 |
| oracle_infinite_hbm | 0.21466451013499338 | 0.9353327808437517 |

## Notes

- Earlier `lmcache_eval.csv` from byte-prefix SHA1 blocks is superseded by this tokenizer replay.
- `lmcache_session_sample.csv` (200 sessions, no `input` text) remains a metadata-only sample; it is not the replay used for TTFT.
- OptiKV mean hit rate 0.200 on this trace is much lower than RDMA routing (0.496); P50 TTFT is worse than Optical-Reservation and RDMA routing under the roofline clock.
- Log: `logs/sim/lmcache_tokenized.log`