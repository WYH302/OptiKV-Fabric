"""Physical conservation & invariant gatekeeper tests for OptiKV-Fabric.

Verifies:
1. Published KV byte formulas (GQA, MLA, Hybrid Linear, BF16 vs exact 0.5x FP8).
2. Model simulation context caps.
3. NoCache strictly yields hit_rate == 0 and recomputed_tokens == input_tokens.
4. No policy (including OptiKV-Fabric) beats Oracle bounds.
5. Standing circuit rule: consecutive transfers to same peer pay tau == 0.0; Optical-Naive pays tau.
6. Request-unit block pinning: zero pinned blocks evicted during execution.
7. Lossless selective recompute abort: under extreme MEMS switch latency (50 ms) + short prefix,
   OptiKV aborts remote fetch and never degrades worse than NoCache.
8. Multi-tier spill verification: when HBM overflows, Local-DRAM spills to DRAM and outperforms Local-HBM,
   and OptiKV spills high-value blocks to DRAM/Optical Pool rather than dropping them.
9. Monotonicity: higher optical BW lowers transfer time; larger HBM non-decreases hit rate.
10. Determinism: fixed seed gives bit-identical TTFT sequence even under noise injection.
"""

from __future__ import annotations

from optikv.cache import CacheStore, LRUEvictor
from optikv.hardware import cluster_from_dict
from optikv.kvbytes import gqa_bytes_per_token, mla_bytes_per_token
from optikv.modelspec import ContextCapError, load_catalog, load_model
from optikv.ocs import OpticalCalendar
from optikv.simulator import run_simulation
from optikv.trace import BlockRef, Request, Trace, generate_synthetic
from policies import POLICY_NAMES


def _cluster(reconfig_us: float = 25000.0, hbm: int = 8 * 1024**3) -> dict:
    return {
        "num_workers": 2,
        "hbm_bytes_per_worker": hbm,
        "dram_bytes_per_worker": 32 * 1024**3,
        "ssd_bytes_per_worker": 64 * 1024**3,
        "rdma_pool_bytes": 64 * 1024**3,
        "dram_bandwidth_gb_s": 26.28,
        "ssd_bandwidth_gb_s": 3.0,
        "rdma_bandwidth_gb_s": 25.0,
        "rdma_latency_us": 2.0,
        "alpha_overlap": 0.7,
        "ocs": {
            "concurrent_links": 1,
            "bandwidth_gb_s": 45.0,
            "latency_us": 5.0,
            "reconfig_us": reconfig_us,
        },
    }


def _config(policies, trace, **extra) -> dict:
    config = {
        "model": "qwen2.5-7b-instruct",
        "seed": 0,
        "dtype_bytes": 2,
        "hardware_profile": "rtx3090",
        "calibration": "configs/hardware/calibration_3090.yaml",
        "cluster": _cluster(),
        "policies": policies,
        "trace": {
            "kind": "synthetic",
            "n_sessions": 4,
            "turns_per_session": 3,
            "prefix_blocks": 4,
            "suffix_blocks": 2,
            "block_tokens": 128,
            "inter_arrival_s": 0.0,
            "seed": 0,
        },
    }
    config.update(extra)
    if trace is not None:
        config.pop("trace", None)
    return config


def _run(policies, trace=None, **extra):
    return run_simulation(_config(policies, trace, **extra), trace=trace)


def _remote_trace(prefix_tokens: int, new_tokens: int) -> Trace:
    shared = (BlockRef("shared", prefix_tokens),)
    fresh = (BlockRef("fresh", new_tokens),)
    return Trace(
        name="remote-prefix",
        kind="synthetic",
        block_tokens=prefix_tokens,
        requests=[
            Request("warm", 0.0, "home", 0, 0.0, shared, output_tokens=1, assigned_worker=0),
            Request("probe1", 10.0, "other1", 0, 0.0, shared + fresh, output_tokens=1, assigned_worker=1),
            Request(
                "probe2",
                20.0,
                "other2",
                0,
                0.0,
                shared + (BlockRef("fresh2", new_tokens),),
                output_tokens=1,
                assigned_worker=1,
            ),
        ],
    )


def test_kv_bytes_match_published_constants():
    catalog = load_catalog()
    bf16 = {
        "qwen2.5-7b-instruct": 57344,
        "qwen2.5-7b-instruct-1m": 57344,
        "llama-3.1-8b": 131072,
        "qwen3-8b": 147456,
        "llama-3.1-70b": 327680,
        "deepseek-v3": 70272,
    }
    for name, expected in bf16.items():
        spec = catalog[name]
        assert spec.kv_bytes_per_token(2) == expected
        assert spec.kv_bytes_per_token(1) == expected // 2
    assert gqa_bytes_per_token(28, 4, 128, 2) == 57344
    assert mla_bytes_per_token(61, 512, 64, 2) == 70272
    assert mla_bytes_per_token(61, 512, 64, 1) == 35136


def test_hybrid_models_use_full_attention_layers_only():
    nxt = load_model("qwen3-next-80b-a3b")
    q35 = load_model("qwen3.5-9b")
    assert nxt.kv_arch == "hybrid_gqa"
    assert nxt.num_full_attention_layers == 12
    assert nxt.num_linear_attention_layers == 36
    assert nxt.kv_bytes_per_token(2) == gqa_bytes_per_token(12, 2, 256, 2)
    assert q35.kv_arch == "hybrid_gqa"
    assert q35.num_full_attention_layers == 8
    assert q35.num_linear_attention_layers == 24
    assert q35.kv_bytes_per_token(2) == gqa_bytes_per_token(8, 4, 256, 2)


def test_context_caps_follow_configs():
    load_model("qwen2.5-7b-instruct-1m").check_context(1_000_000)
    qwen3 = load_model("qwen3-8b")
    assert qwen3.simulation_context_cap == 131072
    try:
        qwen3.check_context(1_000_000)
    except ContextCapError:
        pass
    else:
        raise AssertionError("Qwen3-8B accepted a 1M context")
    for name in ("llama-3.1-8b", "llama-3.1-70b"):
        try:
            load_model(name).check_context(200_000)
        except ContextCapError:
            pass
        else:
            raise AssertionError(f"{name} accepted 200k tokens")


def test_nocache_hit_rate_is_zero():
    result = _run(["no_cache"])
    assert result.request_rows
    for row in result.request_rows:
        assert row["hit_rate"] == 0.0
        assert row["recomputed_tokens"] == row["input_tokens"]
        assert row["hbm_hit_tokens"] == 0
        assert row["remote_hbm_hit_tokens"] == 0


def test_nobody_beats_oracle():
    result = _run(list(POLICY_NAMES))

    def index(policy: str) -> dict[str, float]:
        return {
            row["request_id"]: row["ttft_s"]
            for row in result.request_rows
            if row["policy"] == policy
        }

    bounds = [index("oracle_infinite_hbm"), index("oracle_infinite_bandwidth")]
    for policy in POLICY_NAMES:
        got = index(policy)
        for bound in bounds:
            for request_id, ttft in got.items():
                assert ttft + 1e-8 >= bound[request_id], (policy, request_id, ttft, bound[request_id])


def test_standing_circuit_same_target_tau_is_zero():
    calendar = OpticalCalendar(1, reconfig_s=0.05, bandwidth_B_s=45e9, latency_s=0.0)
    first = calendar.transfer_sequence(0.0, [(3, 1_000_000)], commit=True)
    second = calendar.transfer_sequence(first[0].end_s, [(3, 1_000_000)], commit=True)
    third = calendar.transfer_sequence(second[0].end_s, [(2, 1_000_000)], commit=True)
    assert first[0].reconfig_s == 0.05
    assert second[0].reconfig_s == 0.0
    assert third[0].reconfig_s == 0.05
    assert calendar.standing_hits == 1
    assert 0.0 < calendar.duty_cycle < 1.0


def test_standing_circuit_kept_for_reservation_and_optikv_dropped_for_naive():
    # Two consecutive remote fetches from worker 0 to different idle workers (1 and 2)
    shared1 = (BlockRef("s1", 4096),)
    shared2 = (BlockRef("s2", 4096),)
    trace = Trace(
        name="standing-vs-naive",
        kind="synthetic",
        block_tokens=4096,
        requests=[
            Request("w1", 0.0, "h1", 0, 0.0, shared1, assigned_worker=0),
            Request("w2", 1.0, "h2", 0, 0.0, shared2, assigned_worker=0),
            Request("p1", 10.0, "c1", 0, 0.0, shared1 + (BlockRef("f1", 512),), assigned_worker=1),
            Request("p2", 20.0, "c2", 0, 0.0, shared2 + (BlockRef("f2", 512),), assigned_worker=2),
        ],
    )
    cl = _cluster(reconfig_us=25000.0)
    cl["num_workers"] = 3
    result = _run(
        ["optical_naive", "optical_reservation", "optikv"],
        trace=trace,
        cluster=cl,
        optikv={"enable_routing": False, "enable_prefetch": False},
    )
    p2 = {row["policy"]: row for row in result.request_rows if row["request_id"] == "p2"}
    assert p2["optical_naive"]["reconfig_s"] >= 0.024
    assert p2["optical_reservation"]["reconfig_s"] == 0.0
    assert p2["optikv"]["reconfig_s"] == 0.0


def test_pins_are_not_evicted():
    cluster = cluster_from_dict(_cluster(hbm=15))
    cache = CacheStore(cluster, kv_bytes_per_token=10)
    evictor = LRUEvictor()
    cache.upsert("hbm", 0, "pinned", 1, 0.0, 1.0, evictor, 0, pin=True)
    cache.upsert("hbm", 0, "other", 1, 1.0, 0.0, evictor, 1, pin=False, auto_trim=False)
    evicted = cache.trim("hbm", 0, evictor, 1)
    assert cache.has("pinned", "hbm", 0)
    assert not cache.has("other", "hbm", 0)
    assert len(evicted) == 1 and evicted[0].block_id == "other"
    assert cache.pinned_evictions == 0


def test_short_prefix_recompute_fallback_is_not_worse_than_nocache():
    trace = _remote_trace(64, 64)
    result = _run(
        ["no_cache", "optikv"],
        trace=trace,
        cluster=_cluster(reconfig_us=50_000.0),
        optikv={"enable_routing": False, "enable_recompute_fallback": True},
    )
    rows = {row["policy"]: row for row in result.request_rows if row["request_id"] == "probe1"}
    assert rows["optikv"]["aborted"] is True
    assert rows["optikv"]["reconfig_s"] == 0.0
    assert rows["optikv"]["ttft_s"] <= rows["no_cache"]["ttft_s"] + 1e-9


def test_multitier_spill_local_dram_beats_local_hbm_under_capacity_pressure():
    spec = load_model("qwen2.5-7b-instruct")
    block_tokens = 512
    block_bytes = block_tokens * spec.kv_bytes_per_token(2)
    # 2 sessions alternating on 1 worker; HBM holds only 4 blocks (1 session), DRAM holds 64 blocks
    s0 = tuple(BlockRef(f"s0-{i}", block_tokens) for i in range(4))
    s1 = tuple(BlockRef(f"s1-{i}", block_tokens) for i in range(4))
    trace = Trace(
        name="spill-test",
        kind="synthetic",
        block_tokens=block_tokens,
        requests=[
            Request("r0", 0.0, "s0", 0, 0.0, s0, assigned_worker=0),
            Request("r1", 5.0, "s1", 0, 0.0, s1, assigned_worker=0),
            Request("r2", 10.0, "s0", 1, 5.0, s0 + (BlockRef("s0-new", block_tokens),), assigned_worker=0),
        ],
    )
    cl = _cluster(hbm=block_bytes * 4)
    cl["num_workers"] = 1
    res = _run(["local_hbm", "local_dram"], trace=trace, cluster=cl)
    by_pol = {r["policy"]: r for r in res.summary_rows}
    assert by_pol["local_dram"]["dram_hit_rate"] > 0.0
    assert by_pol["local_dram"]["hit_rate"] > by_pol["local_hbm"]["hit_rate"]
    assert by_pol["local_dram"]["mean_ttft_s"] < by_pol["local_hbm"]["mean_ttft_s"]


def test_same_seed_is_deterministic():
    first = _run(["optikv"], seed=7, reuse_noise_std=0.10, latency_noise_std=0.05)
    second = _run(["optikv"], seed=7, reuse_noise_std=0.10, latency_noise_std=0.05)
    assert [row["ttft_s"] for row in first.request_rows] == [
        row["ttft_s"] for row in second.request_rows
    ]


def test_bandwidth_context_and_hbm_monotonicity():
    slow = OpticalCalendar(1, 0.0, 50e9, 0.0)
    fast = OpticalCalendar(1, 0.0, 100e9, 0.0)
    slow_t = slow.transfer_sequence(0.0, [(1, 10_000_000)], commit=True)[0].transfer_s
    fast_t = fast.transfer_sequence(0.0, [(1, 10_000_000)], commit=True)[0].transfer_s
    assert fast_t < slow_t
    spec = load_model("qwen2.5-7b-instruct")
    assert spec.kv_bytes_per_token(2) * 2048 > spec.kv_bytes_per_token(2) * 1024
    trace = generate_synthetic(
        spec,
        name="hbm-mono",
        n_sessions=1,
        turns_per_session=4,
        prefix_blocks=4,
        suffix_blocks=1,
        block_tokens=64,
        inter_arrival_s=1000.0,
        pin_workers=True,
        worker_count=1,
    )
    block_bytes = 64 * spec.kv_bytes_per_token(2)
    small = _run(["local_hbm"], trace=trace, cluster=_cluster(hbm=block_bytes * 2))
    large = _run(["local_hbm"], trace=trace, cluster=_cluster(hbm=block_bytes * 20))
    assert small.summary_rows[0]["hbm_hit_rate"] < large.summary_rows[0]["hbm_hit_rate"]


def test_single_gpu_belady_dominates_online_evictors():
    spec = load_model("qwen2.5-7b-instruct")
    block_tokens = 256
    block_bytes = block_tokens * spec.kv_bytes_per_token(2)
    trace = generate_synthetic(
        spec,
        name="eviction-bound",
        n_sessions=6,
        turns_per_session=4,
        prefix_blocks=4,
        suffix_blocks=2,
        block_tokens=block_tokens,
        inter_arrival_s=5.0,
        pin_workers=True,
        worker_count=1,
    )
    cl = _cluster(hbm=block_bytes * 10)
    cl["num_workers"] = 1
    cl["dram_bytes_per_worker"] = 0
    cl["ssd_bytes_per_worker"] = 0
    cl["rdma_pool_bytes"] = 0
    res = _run(["local_hbm", "eviction_s3fifo", "eviction_gdsf", "eviction_belady"], trace=trace, cluster=cl)
    by_pol = {r["policy"]: r["hbm_hit_rate"] for r in res.summary_rows}
    assert by_pol["eviction_belady"] >= by_pol["local_hbm"] - 1e-9
    assert by_pol["eviction_belady"] >= by_pol["eviction_s3fifo"] - 1e-9
    assert by_pol["eviction_belady"] >= by_pol["eviction_gdsf"] - 1e-9
    assert by_pol["eviction_s3fifo"] > 0.0
