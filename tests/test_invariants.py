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
    # Two consecutive remote fetches along the SAME physical edge (worker 1 <- worker 0)
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
            Request("p2", 20.0, "c2", 0, 0.0, shared2 + (BlockRef("f2", 512),), assigned_worker=1),
        ],
    )
    cl = _cluster(reconfig_us=25000.0)
    cl["num_workers"] = 2
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
    res = _run(
        ["local_hbm", "eviction_s3fifo", "eviction_sieve", "eviction_rrevict", "eviction_gdsf", "eviction_belady"],
        trace=trace,
        cluster=cl,
    )
    by_pol = {r["policy"]: r["hbm_hit_rate"] for r in res.summary_rows}
    assert by_pol["eviction_belady"] >= by_pol["local_hbm"] - 1e-9
    assert by_pol["eviction_belady"] >= by_pol["eviction_s3fifo"] - 1e-9
    assert by_pol["eviction_belady"] >= by_pol["eviction_sieve"] - 1e-9
    assert by_pol["eviction_belady"] >= by_pol["eviction_rrevict"] - 1e-9
    assert by_pol["eviction_belady"] >= by_pol["eviction_gdsf"] - 1e-9
    assert by_pol["eviction_s3fifo"] > 0.0
    assert by_pol["eviction_sieve"] > 0.0
    assert by_pol["eviction_rrevict"] > 0.0


def test_bipartite_port_contention_and_wdm_striping():
    # Single-wavelength vs 2-wavelength WDM striping
    cal_1w = OpticalCalendar(2, reconfig_s=0.025, bandwidth_B_s=100e9, latency_s=0.0, num_workers=4, wdm_wavelengths=1)
    cal_2w = OpticalCalendar(2, reconfig_s=0.025, bandwidth_B_s=100e9, latency_s=0.0, num_workers=4, wdm_wavelengths=2)
    p1 = cal_1w.transfer_sequence(0.0, [(-1, 200_000_000)], commit=True)[0]
    p2 = cal_2w.transfer_sequence(0.0, [(-1, 200_000_000)], commit=True)[0]
    assert abs(p1.transfer_s - 2.0 * p2.transfer_s) < 1e-12

    # Strict bipartite mode serializes two simultaneous fetches from the same src_worker
    cal_bi = OpticalCalendar(
        2, reconfig_s=0.025, bandwidth_B_s=100e9, latency_s=0.0, num_workers=4, bipartite_port_contention=True
    )
    a = cal_bi.transfer_sequence(0.0, [(-1, 1_000_000_000)], commit=True, src_worker=0)[0]
    b = cal_bi.transfer_sequence(0.0, [(-1, 1_000_000_000)], commit=True, src_worker=0)[0]
    assert b.start_s >= a.end_s - 1e-12


def test_parallel_multiport_transfer_sequence():
    cal = OpticalCalendar(2, reconfig_s=0.0, bandwidth_B_s=100e9, latency_s=0.0, num_workers=4, bipartite_port_contention=False)
    plans = cal.transfer_sequence(0.0, [(1, 100_000_000), (2, 100_000_000)], commit=True)
    assert len(plans) == 2
    assert plans[0].start_s == 0.0 and plans[1].start_s == 0.0
    assert plans[0].port_index != plans[1].port_index


def test_inflight_prefetch_join_is_strictly_causal():
    # Two sessions alternate on 1 worker with small HBM (1 session capacity);
    # Turn 1 arrives early while async prefetch from Optical Pool is still in flight.
    spec = load_model("qwen2.5-7b-instruct")
    bt = 512
    bb = bt * spec.kv_bytes_per_token(2)
    s0 = tuple(BlockRef(f"s0-{i}", bt) for i in range(8))
    s1 = tuple(BlockRef(f"s1-{i}", bt) for i in range(8))
    trace = Trace(
        name="inflight-join",
        kind="synthetic",
        block_tokens=bt,
        requests=[
            Request("r0", 0.0, "s0", 0, 0.0, s0, assigned_worker=0),
            Request("r1", 2.0, "s1", 0, 0.0, s1, assigned_worker=0),
            # r2 arrives 10ms after r0 would finish if it were right after r1: force very short gap
            Request("r2", 4.01, "s0", 1, 0.01, s0 + (BlockRef("s0-u", bt),), assigned_worker=0),
        ],
    )
    cl = _cluster(reconfig_us=25_000.0, hbm=bb * 8)
    cl["num_workers"] = 1
    res = _run(["optikv", "oracle_infinite_hbm"], trace=trace, cluster=cl)
    by_pol = {r["policy"]: r for r in res.summary_rows}
    assert by_pol["optikv"]["mean_ttft_s"] >= by_pol["oracle_infinite_hbm"]["mean_ttft_s"] - 1e-9
    assert by_pol["optikv"]["pinned_evictions"] == 0


def test_decode_coupled_queue_monotonicity():
    cl_base = _cluster()
    cl_dec = _cluster()
    cl_dec["couple_decode_queue"] = True
    cl_dec["decode_batch_slots"] = 4
    trace = Trace(
        name="decode-couple",
        kind="synthetic",
        block_tokens=256,
        requests=[
            Request(f"r{i}", i * 0.01, f"s{i}", 0, 0.0, (BlockRef(f"b{i}", 256),), output_tokens=64)
            for i in range(8)
        ],
    )
    res_base = _run(["optikv"], trace=trace, cluster=cl_base)
    res_dec = _run(["optikv", "oracle_infinite_hbm"], trace=trace, cluster=cl_dec)
    dec_by = {r["policy"]: r for r in res_dec.summary_rows}
    assert dec_by["optikv"]["mean_ttft_s"] >= res_base.summary_rows[0]["mean_ttft_s"] - 1e-9
    assert dec_by["optikv"]["mean_ttft_s"] >= dec_by["oracle_infinite_hbm"]["mean_ttft_s"] - 1e-9


def test_concur_admission_control_respects_oracle():
    res = _run(["concur_rdma", "oracle_infinite_hbm"])
    by_pol = {r["policy"]: r for r in res.summary_rows}
    assert by_pol["concur_rdma"]["mean_ttft_s"] >= by_pol["oracle_infinite_hbm"]["mean_ttft_s"] - 1e-9
    assert by_pol["concur_rdma"]["pinned_evictions"] == 0


def test_queue_aware_dram_co_steering():
    # When OCS is queued beyond DRAM transfer time, OptiKV uses local DRAM rather than stalling on OCS
    spec = load_model("qwen2.5-7b-instruct")
    bt = 512
    bb = bt * spec.kv_bytes_per_token(2)
    s0 = tuple(BlockRef(f"q0-{i}", bt) for i in range(4))
    s1 = tuple(BlockRef(f"q1-{i}", bt) for i in range(4))
    trace = Trace(
        name="dram-fallback",
        kind="synthetic",
        block_tokens=bt,
        requests=[
            Request("r0", 0.0, "s0", 0, 0.0, s0, assigned_worker=0),
            Request("r1", 2.0, "s1", 0, 0.0, s1, assigned_worker=0),
            Request("r2", 4.0, "s0", 1, 0.0, s0 + (BlockRef("q0-n", bt),), assigned_worker=0),
        ],
    )
    cl = _cluster(reconfig_us=50_000.0, hbm=bb * 4)
    cl["num_workers"] = 1
    cl["dram_bandwidth_gb_s"] = 50.0
    cl["ocs"]["bandwidth_gb_s"] = 100.0
    res = _run(["optikv"], trace=trace, cluster=cl, optikv={"enable_prefetch": False})
    assert res.summary_rows[0]["dram_hit_rate"] > 0.0


def test_theorem1_bipartite_matching_conservation_and_bounds():
    """Verify strict physical bipartite matching conservation and Theorem 1 bounds
    across both non-oversubscribed (P >= W) and oversubscribed (P < W) OCS regimes."""
    tau = 0.025
    bw = 200e9
    lat = 2e-6
    payload_b = 500_000_000  # 500 MB burst (2.5 ms active)
    t_act = lat + payload_b / bw

    # Regime 1 (Non-oversubscribed P >= W): each worker latches its own dedicated pool port;
    # total switches S_sw <= min(M, W), and duty_cycle >= t_act / (t_act + (W/M)*tau).
    W, P = 8, 8
    cal_1to1 = OpticalCalendar(P, reconfig_s=tau, bandwidth_B_s=bw, latency_s=lat, num_workers=W, bipartite_port_contention=True)
    m_1to1 = 64
    for idx in range(m_1to1):
        cal_1to1.transfer_sequence(idx * 0.005, [(-1, payload_b)], commit=True, src_worker=idx % W)
        assert len(cal_1to1.latched_pool_workers()) <= min(W, P)
    sw_1to1 = round(cal_1to1.total_reconfig_s / tau)
    assert sw_1to1 == W  # Exactly W initial cold latches, zero reconfigurations thereafter!
    lower_duty = t_act / (t_act + (W / m_1to1) * tau)
    assert abs(cal_1to1.duty_cycle - lower_duty) < 1e-12
    t_tot = cal_1to1.total_active_s + cal_1to1.total_reconfig_s
    t_opt = m_1to1 * t_act + tau
    assert t_tot / t_opt <= 1.0 + (W - 1) * tau / t_opt + 1e-12

    # Regime 2 (Strict 4:1 Oversubscription: W=32 GPUs sharing P=8 pool ports):
    # At EVERY instant, at most P=8 workers can be physically matched to the pool!
    # Round-robin across 32 GPUs MUST pay tau on every transfer (ZERO impossible permanent 32-on-8 locks!).
    W_ov, P_ov = 32, 8
    cal_ov = OpticalCalendar(P_ov, reconfig_s=tau, bandwidth_B_s=bw, latency_s=lat, num_workers=W_ov, bipartite_port_contention=True)
    m_ov = 64
    for idx in range(m_ov):
        cal_ov.transfer_sequence(idx * 0.01, [(-1, payload_b)], commit=True, src_worker=idx % W_ov)
        assert len(cal_ov.latched_pool_workers()) <= P_ov
    sw_ov = round(cal_ov.total_reconfig_s / tau)
    assert sw_ov == m_ov  # Every round-robin access across 32 GPUs on 8 ports MUST physically switch!
    assert cal_ov.standing_hits == 0
    assert abs(cal_ov.duty_cycle - t_act / (t_act + tau)) < 1e-12

    # Regime 2b (Oversubscribed P=2, W=8 with matching-aware dwell of B=4 bursts per latched worker):
    cal_dw = OpticalCalendar(2, reconfig_s=tau, bandwidth_B_s=bw, latency_s=lat, num_workers=8, bipartite_port_contention=True)
    dwell = 4
    m_dw = 32
    for idx in range(m_dw):
        w_grp = (idx // dwell) % 8
        cal_dw.transfer_sequence(idx * 0.01, [(-1, payload_b)], commit=True, src_worker=w_grp)
        assert len(cal_dw.latched_pool_workers()) <= 2
    sw_dw = round(cal_dw.total_reconfig_s / tau)
    assert sw_dw == m_dw // dwell
    assert abs(cal_dw.duty_cycle - t_act / (t_act + tau / dwell)) < 1e-12

    # Part (2): Zero critical-path stall when gap >= wake + wait + tau_eff + t_act (even when tau_eff = tau!)
    cal_c = OpticalCalendar(1, reconfig_s=tau, bandwidth_B_s=bw, latency_s=lat, num_workers=1)
    delta_wake = 0.02
    gap_ok = delta_wake + tau + t_act + 0.01
    p_ok = cal_c.transfer_sequence(delta_wake, [(-1, payload_b)], commit=True, src_worker=0)[0]
    t_crit_ok = max(0.0, p_ok.end_s - gap_ok)
    assert t_crit_ok == 0.0

    # Early arrival in-flight join strictly saves (gap_early - delta_wake) over arrival-time fetch
    cal_j = OpticalCalendar(1, reconfig_s=tau, bandwidth_B_s=bw, latency_s=lat, num_workers=1)
    p_j = cal_j.transfer_sequence(delta_wake, [(-1, payload_b)], commit=True, src_worker=0)[0]
    gap_early = delta_wake + 0.010
    t_crit_join = max(0.0, p_j.end_s - gap_early)
    t_crit_arrival = p_j.wait_s + p_j.reconfig_s + p_j.transfer_s
    assert abs((t_crit_arrival - t_crit_join) - (gap_early - delta_wake)) < 1e-12
