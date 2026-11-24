#!/usr/bin/env python3
"""Real NVIDIA RTX 3090 Microbenchmark (Linux CUDA / PyTorch 2.14+cu132):
1. Exact 28-layer Qwen2.5-7B BF16 paged KV cache CPU<->GPU PCIe Gen4 offload/reload (4K..32K).
2. True Dual-CUDA-Stream layer-wise compute-transfer overlap factor alpha_ovl measurement
   combining exact Qwen2.5-7B layer compute (QKV/O + SwiGLU MLP + native GQA SDPA) with async H2D KV reload.
"""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "results" / "3090" / "kv_offload_overlap_3090.csv"

# Exact Qwen2.5-7B-Instruct geometry (56 KiB/tok BF16)
NUM_LAYERS = 28
HIDDEN = 3584
INTERMEDIATE = 18944
NUM_Q_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
DTYPE = torch.bfloat16
DTYPE_BYTES = 2
MAX_CTX = 32768
RING_SLOTS = 3


@torch.no_grad()
def main() -> int:
    assert torch.cuda.is_available(), "CUDA GPU required!"
    dev = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)} | PyTorch: {torch.__version__}", flush=True)
    torch.manual_seed(0)

    # Exact Qwen2.5-7B dense layer weights (~0.5 GiB)
    w_q = torch.randn(NUM_Q_HEADS * HEAD_DIM, HIDDEN, dtype=DTYPE, device=dev) * 0.02
    w_kv = torch.randn(2 * NUM_KV_HEADS * HEAD_DIM, HIDDEN, dtype=DTYPE, device=dev) * 0.02
    w_o = torch.randn(HIDDEN, HIDDEN, dtype=DTYPE, device=dev) * 0.02
    w_gu = torch.randn(2 * INTERMEDIATE, HIDDEN, dtype=DTYPE, device=dev) * 0.02
    w_d = torch.randn(HIDDEN, INTERMEDIATE, dtype=DTYPE, device=dev) * 0.02

    # Allocate 4-slot pinned host KV pool (256 MiB total -> exceeds CPU L3 so every transfer hits DRAM->PCIe Gen4)
    # and full 28-layer contiguous GPU KV cache (1.75 GiB at 32K, matching real vLLM/LMCache per-layer HBM layout).
    t_alloc = time.perf_counter()
    HOST_SLOTS = 4
    host_ring = [torch.empty((2, NUM_KV_HEADS, MAX_CTX, HEAD_DIM), dtype=DTYPE, pin_memory=True) for _ in range(HOST_SLOTS)]
    for hr in host_ring:
        hr.zero_()
    gpu_kv_all = [torch.empty((2, NUM_KV_HEADS, MAX_CTX, HEAD_DIM), dtype=DTYPE, device=dev) for _ in range(NUM_LAYERS)]
    gpu_kv_all[0].normal_(0.0, 0.02)
    for l in range(1, NUM_LAYERS):
        gpu_kv_all[l].copy_(gpu_kv_all[0])
    for hr in host_ring:
        hr.copy_(gpu_kv_all[0])
    torch.cuda.synchronize()
    print(f"Allocated {HOST_SLOTS}-slot pinned host KV (256 MiB) + {NUM_LAYERS}-layer GPU KV (1.75 GiB) in {time.perf_counter()-t_alloc:.2f}s", flush=True)

    # Global GPU clock warmup so P0 boost clock is locked before first timed grid point
    x_warm = torch.randn(1, 512, HIDDEN, dtype=DTYPE, device=dev) * 0.02
    for _ in range(10):
        _ = F.linear(F.silu(F.linear(x_warm, w_gu).chunk(2, dim=-1)[0]), w_d)
    torch.cuda.synchronize()
    del x_warm

    copy_stream = torch.cuda.Stream(device=dev)
    comp_stream = torch.cuda.Stream(device=dev)
    ready_ev = [torch.cuda.Event() for _ in range(NUM_LAYERS)]

    def run_layer_compute(h: torch.Tensor, kv_gpu: torch.Tensor, ctx: int, delta: int) -> torch.Tensor:
        q = F.linear(h, w_q).view(1, delta, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
        _ = F.linear(h, w_kv)
        k = kv_gpu[0, :, :ctx, :].unsqueeze(0)  # [1, H_kv, ctx, D]
        v = kv_gpu[1, :, :ctx, :].unsqueeze(0)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(1, delta, HIDDEN)
        out = F.linear(attn_out, w_o) + h
        gu = F.linear(out, w_gu)
        gate, up = gu.chunk(2, dim=-1)
        mlp = F.linear(F.silu(gate) * up, w_d)
        return out + mlp

    grid = [
        (4096, 128),
        (4096, 512),
        (8192, 128),
        (8192, 512),
        (16384, 128),
        (16384, 512),
        (32768, 128),
        (32768, 512),
    ]
    rows = []
    t0_all = time.perf_counter()

    for ctx, delta in grid:
        layer_bytes = 2 * ctx * NUM_KV_HEADS * HEAD_DIM * DTYPE_BYTES
        total_kv_bytes = NUM_LAYERS * layer_bytes
        # Contiguous per-layer views of exact [2, H_kv, ctx, D]
        h_layers = [host_ring[l % HOST_SLOTS].view(-1)[: layer_bytes // DTYPE_BYTES].view(2, NUM_KV_HEADS, ctx, HEAD_DIM) for l in range(NUM_LAYERS)]
        g_layers = [gpu_kv_all[l].view(-1)[: layer_bytes // DTYPE_BYTES].view(2, NUM_KV_HEADS, ctx, HEAD_DIM) for l in range(NUM_LAYERS)]
        x = torch.randn(1, delta, HIDDEN, dtype=DTYPE, device=dev) * 0.02

        def copy_h2d_all():
            with torch.cuda.stream(copy_stream):
                for l in range(NUM_LAYERS):
                    g_layers[l].copy_(h_layers[l], non_blocking=True)

        def copy_d2h_all():
            with torch.cuda.stream(copy_stream):
                for l in range(NUM_LAYERS):
                    h_layers[l].copy_(g_layers[l], non_blocking=True)

        def compute_all():
            with torch.cuda.stream(comp_stream):
                h = x
                for l in range(NUM_LAYERS):
                    h = run_layer_compute(h, g_layers[l], ctx, delta)

        def overlapped_all():
            # Strict producer-consumer layer-wise KV prefetch:
            # copy_stream loads layer l (with 1-layer lookahead), comp_stream waits on ready_ev[l]
            # before computing layer l. Stream.query() flushes WDDM command batches immediately.
            with torch.cuda.stream(copy_stream):
                g_layers[0].copy_(h_layers[0], non_blocking=True)
                ready_ev[0].record(copy_stream)
                _ = copy_stream.query()
            h = x
            for l in range(NUM_LAYERS):
                nxt = l + 1
                if nxt < NUM_LAYERS:
                    with torch.cuda.stream(copy_stream):
                        g_layers[nxt].copy_(h_layers[nxt], non_blocking=True)
                        ready_ev[nxt].record(copy_stream)
                        _ = copy_stream.query()
                with torch.cuda.stream(comp_stream):
                    comp_stream.wait_event(ready_ev[l])
                    h = run_layer_compute(h, g_layers[l], ctx, delta)
                    _ = comp_stream.query()

        def time_sync(fn, warmup=2, repeats=5) -> float:
            for _ in range(warmup):
                fn()
                torch.cuda.synchronize()
            ts = []
            for _ in range(repeats):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            return statistics.median(ts)

        t_h2d = time_sync(copy_h2d_all)
        t_d2h = time_sync(copy_d2h_all)
        t_comp = time_sync(compute_all)
        t_ovl = time_sync(overlapped_all)

        h2d_gb_s = (total_kv_bytes / 1e9) / t_h2d
        d2h_gb_s = (total_kv_bytes / 1e9) / t_d2h
        saved = (t_h2d + t_comp) - t_ovl
        alpha_eff = max(0.0, min(1.0, saved / max(1e-9, min(t_h2d, t_comp))))
        row = {
            "context_tokens": ctx,
            "delta_tokens": delta,
            "kv_mib": round(total_kv_bytes / (1024**2), 2),
            "t_h2d_ms": round(t_h2d * 1e3, 2),
            "t_d2h_ms": round(t_d2h * 1e3, 2),
            "h2d_gb_s": round(h2d_gb_s, 2),
            "d2h_gb_s": round(d2h_gb_s, 2),
            "t_comp_ms": round(t_comp * 1e3, 2),
            "t_serial_ms": round((t_h2d + t_comp) * 1e3, 2),
            "t_overlapped_ms": round(t_ovl * 1e3, 2),
            "overlap_speedup": round((t_h2d + t_comp) / t_ovl, 3),
            "alpha_ovl_measured": round(alpha_eff, 4),
        }
        rows.append(row)
        print(
            f"  ctx={ctx:5d} delta={delta:3d} ({row['kv_mib']:7.1f} MiB) | "
            f"H2D={row['t_h2d_ms']:6.2f}ms ({row['h2d_gb_s']:5.2f} GB/s) "
            f"D2H={row['t_d2h_ms']:6.2f}ms ({row['d2h_gb_s']:5.2f} GB/s) | "
            f"Comp={row['t_comp_ms']:6.2f}ms Serial={row['t_serial_ms']:6.2f}ms Ovl={row['t_overlapped_ms']:6.2f}ms "
            f"alpha={row['alpha_ovl_measured']:.3f}",
            flush=True,
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    alphas = [r["alpha_ovl_measured"] for r in rows]
    h2ds = [r["h2d_gb_s"] for r in rows]
    d2hs = [r["d2h_gb_s"] for r in rows]
    print(
        f"\nSaved {OUT_CSV} in {time.perf_counter()-t0_all:.1f}s | "
        f"Median H2D={statistics.median(h2ds):.2f} GB/s, D2H={statistics.median(d2hs):.2f} GB/s, "
        f"Measured alpha_ovl={statistics.median(alphas):.3f} (range {min(alphas):.3f}-{max(alphas):.3f})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
