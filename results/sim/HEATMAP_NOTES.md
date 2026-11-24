# Heatmap notes

`fig5_heatmap.csv` and `fig5_heatmap_corrected.csv` are kept as diagnostics. Do not plot them and do not quote their speedups. The usable table is `fig5_heatmap_tier_load.csv`.

## Cause 1. The full-hit residual was not a KV load

In `fig5_heatmap.csv` the cell context 65536, hit rate 1.0 is NoCache P50 TTFT 248.50891926491767 s over OptiKV P50 TTFT 0.289193171517699 s, ratio 859.3180743540087. Context 131072, hit rate 1.0 is 856.8562030662695 / 0.48328321929142437 = 1772.9897684479233.

Those OptiKV values are not optical transfers. The generator forced one extra 512-token block even at hit rate 1.0 (`suffix_n = max(1, 0)`), so the prompt was 66048 and 131584 tokens. The roofline delta is

- `cold(66048) - cold(65536) = 24.476110 - 24.191520 = 0.284590` s, plus decode 0.004604 s, total 0.289194 s
- `cold(131584) - cold(131072) = 73.139507 - 72.665213 = 0.474294` s, plus decode 0.008989 s, total 0.483283 s

That matches the OptiKV column. The NoCache column does not match a single prefill. One cold prefill of 65536 tokens is 24.191520 s. The cell used 24 probes on one worker, 5 s apart, while NoCache service is 24.476 s, so the queue grows by about 19.48 s per probe. The median of that queue is 248.5 s, not the prefill.

`fig5_heatmap_corrected.csv` removed the queue (`optikv_p50_queue_s` is 0 on every row) and removed the extra block, then took the median of four probes. The first probe does pay a remote HBM load: at 65536 tokens, probe 0 TTFT is 0.113085 s of which transfer is 0.108515 s. Probes 1–3 find those blocks already in the probe worker's HBM and pay only decode, 0.004569 s. The median is that decode stub (0.0002 s at 64 tokens through 0.009 s at 131072). Dividing NoCache's full prefill (0.011968 s at 64 tokens, 72.674168 s at 131072) by the stub prints 63x to 8115x. The queue column is 0, so the ratio is not versus RDMA. A hit that is already in local HBM has to be compared with Local-HBM, and that ratio is not an optical speedup.

## Cause 2. The long-context hit is dropped by the HBM cap

Per-worker HBM in the heatmap is 17179869184 bytes. Qwen2.5-7B-Instruct-1M BF16 KV is 57344 bytes/token.

- 524288 tokens are 30064771072 bytes, which does not fit.
- 1000000 tokens would be 57344000000 bytes. The cell's prefix is 999936 tokens and 57340329984 bytes. Neither fits.
- The largest prefix that fits is 17179869184 // 57344 = 299593 tokens.

Above that, the requested hit rate of 1.0 is not realized. Only the anchor blocks stay on the remote GPU (1024 tokens). Realized hit rate is 0.001953125 at 524288 and 0.0010240655401945725 at 1000000. OptiKV still prefills almost the whole prompt: 873.3033101364972 s versus NoCache 873.4682554090653 s, and 3010.22049923954 s versus 3010.385444512108 s. The hit is ignored past the HBM cap. That is why those cells sit near 1.0 next to the inflated shorter-context ratios.

## What the new table measures

`fig5_heatmap_tier_load.csv` uses one probe, prefetch off, routing off. The prefix is created on worker 0. The probe runs on worker 1, so a hit that fits is a remote HBM load on that request, not a later local-HBM median.

At context 65536 and requested hit rate 1.0 the comparison is `optical_remote_load_vs_recompute`: OptiKV TTFT 0.11308464070419635 s, transfer 0.10851525297778153 s, prefill 0, NoCache TTFT 24.196089757402124 s, ratio 213.96442175284952. At 131072 and hit rate 1.0: OptiKV 0.2009834704661949 s, transfer 0.19202850595556242 s, NoCache 72.67416799052077 s, ratio 361.59276094669906. The transfer term is the 25 ms first-circuit reconfiguration plus KV bytes / 45 GB/s, then decode. It is not the decode stub.

At 64 tokens and hit rate 1.0 the remote load loses to recompute (`recompute_abort_load_exceeds_prefill`). Both TTFTs are 0.011967674548220055 s and the ratio is 1. At 524288 and hit rate 1.0 the label is `hit_dropped_hbm_capacity`: OptiKV 873.3033101364972 s versus NoCache 873.4682554090653 s, ratio 1.000188875125805. At the 1000000 column the prefix that was actually built is 999936 tokens, 57340329984 bytes: OptiKV 3010.22049923954 s versus NoCache 3010.385444512108 s, ratio 1.000054795079832.
