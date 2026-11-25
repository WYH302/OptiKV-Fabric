"""Request-level multi-tier simulator for OptiKV-Fabric (V3).

Models GPU HBM, local host DRAM, local NVMe SSD, contended electrical RDMA,
and an Optical Circuit Switch (OCS) port calendar with standing circuits,
reconfiguration dead time, strictly causal EWMA-gated inter-turn burst prefetching
with Strata-style in-flight join, queue-aware optical-vs-DRAM co-steering,
probation-filtered Agent Anchors, balanced Prefix-GDSF multi-tier retention,
Law 1+3 closed-form lossless selective recompute abort, plus optional strict
bipartite OCS port contention, WDM multi-wavelength striping, and decode
continuous-batching queue coupling.
"""

from __future__ import annotations

import datetime
import heapq
import math
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from optikv.cache import CacheStore, LRUEvictor
from optikv.hardware import Cluster, cluster_from_dict, load_cost_ledger
from optikv.metrics import layer_ttft, percentile
from optikv.modelspec import ROOT, load_model
from optikv.ocs import ElectricalNICCalendar, OpticalCalendar
from optikv.prefill import PrefillModel, build_prefill_model
from optikv.trace import BlockRef, Request, Trace, trace_from_config
from policies import (
    BeladyEvictor,
    PolicySpec,
    RREvictEvictor,
    S3FIFOEvictor,
    SIEVEEvictor,
    build_policy,
    expand_policy_names,
)
from policies.optikv import ReuseEvictor, p_reuse

OPTICAL_POOL_TARGET = -1
_CACHED_GIT_COMMIT: str | None = None


def _git_commit() -> str:
    global _CACHED_GIT_COMMIT
    if _CACHED_GIT_COMMIT is not None:
        return _CACHED_GIT_COMMIT
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        _CACHED_GIT_COMMIT = out.decode("utf-8", errors="ignore").strip() or "unknown"
    except Exception:
        _CACHED_GIT_COMMIT = "no-git"
    return _CACHED_GIT_COMMIT


@dataclass
class SimulationResult:
    summary_rows: list[dict] = field(default_factory=list)
    request_rows: list[dict] = field(default_factory=list)
    hbm_ids_after_request: dict[tuple[str, int], tuple[str, ...]] = field(default_factory=dict)


def run_simulation(
    config: dict | str | Path,
    trace: Trace | None = None,
    *,
    record_hbm_snapshots: bool = False,
    verbose: bool = False,
) -> SimulationResult:
    if not isinstance(config, dict):
        config = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    model = load_model(str(config["model"]))
    dtype_bytes = int(config.get("dtype_bytes", 2))
    if trace is None:
        trace_cfg = dict(config.get("trace") or {"kind": "synthetic"})
        trace_cfg.setdefault("seed", int(config.get("seed", 0)))
        trace = trace_from_config(trace_cfg, model)
    for request in trace.requests:
        model.check_context(request.input_tokens)
    cluster = cluster_from_dict(config["cluster"])
    prefill = build_prefill_model(
        model,
        dtype_bytes,
        config.get("calibration", "configs/hardware/calibration_3090.yaml"),
        hardware_profile=str(config.get("hardware_profile", "rtx3090")),
        quadratic_extrapolation=bool(config.get("quadratic_extrapolation", False)),
    )
    cost = load_cost_ledger(config.get("cost_ledger"))
    names = expand_policy_names(list(config.get("policies") or ["optikv"]))
    flags = dict(config.get("optikv") or {})
    result = SimulationResult()
    seed = int(config.get("seed", getattr(trace, "seed", 0)))
    slo_ttft_s = float(config.get("slo_ttft_s", 2.0))
    reuse_noise_std = float(config.get("reuse_noise_std", 0.0))
    latency_noise_std = float(config.get("latency_noise_std", 0.0))

    for name in names:
        if verbose:
            print(f"policy {name} start requests={len(trace.requests)}", flush=True)
        spec = build_policy(name, flags if name == "optikv" else None)
        summary, requests, hbm_after = _run_policy(
            spec=spec,
            trace=trace,
            model_name=model.name,
            dtype_bytes=dtype_bytes,
            kv_bytes=model.kv_bytes_per_token(dtype_bytes),
            cluster=cluster,
            prefill=prefill,
            seed=seed,
            cost_columns=cost.as_columns(),
            quadratic=bool(config.get("quadratic_extrapolation", False)),
            noise_std=latency_noise_std,
            reuse_noise_std=reuse_noise_std,
            slo_ttft_s=slo_ttft_s,
            record_hbm_snapshots=record_hbm_snapshots,
        )
        result.summary_rows.append(summary)
        result.request_rows.extend(requests)
        if record_hbm_snapshots:
            result.hbm_ids_after_request.update(hbm_after)
        if verbose:
            print(f"policy {name} done requests={summary['num_requests']}", flush=True)
    return result


def _make_evictor(spec: PolicySpec, requests: list[Request]):
    if spec.eviction == "belady":
        return BeladyEvictor(requests)
    if spec.eviction == "s3fifo":
        return S3FIFOEvictor()
    if spec.eviction == "sieve":
        return SIEVEEvictor()
    if spec.eviction == "rrevict":
        return RREvictEvictor()
    if spec.eviction == "reuse":
        return ReuseEvictor()
    return LRUEvictor()


def _prefix_gdsf_score(sc: float, protected_saved: float, nbytes: int, total_cold: float = 0.0) -> float:
    """Bounded-ratio Prefix-GDSF eviction score.

    Preserves strict root > leaf monotonicity within every session (suffix leaves
    always evict before prefix roots) while bounding the root-to-leaf ratio to
    ~[1.0, 2.0] via normalized protected-cost weighting so aging cleanly retires
    idle historical roots without amputating freshly prefetched active sessions.
    """
    if total_cold > 1e-9:
        prefix_weight = 1.0 + (max(0.0, protected_saved) / total_cold)
    else:
        prefix_weight = 1.0 + math.log1p(max(0.0, protected_saved))
    # Multiply by mild cost sensitivity log1p(total_cold * 10) to favor longer reused prefixes
    cost_factor = 1.0 + 0.25 * math.log1p(max(0.0, total_cold) * 10.0)
    return (sc * prefix_weight * cost_factor) / float(max(1, nbytes))


def _run_policy(
    spec: PolicySpec,
    trace: Trace,
    model_name: str,
    dtype_bytes: int,
    kv_bytes: int,
    cluster: Cluster,
    prefill: PrefillModel,
    seed: int,
    cost_columns: dict,
    quadratic: bool,
    noise_std: float = 0.0,
    reuse_noise_std: float = 0.0,
    slo_ttft_s: float = 2.0,
    record_hbm_snapshots: bool = False,
) -> tuple[dict, list[dict], dict[tuple[str, int], tuple[str, ...]]]:
    requests = sorted(trace.requests, key=lambda item: (item.arrival_s, item.request_id))
    cache = CacheStore(cluster, kv_bytes)
    evictor = _make_evictor(spec, requests)
    calendar = OpticalCalendar(
        cluster.ocs_concurrent_links,
        cluster.reconfig_s,
        cluster.ocs_bandwidth_B_s,
        cluster.ocs_latency_s,
        num_workers=cluster.num_workers,
        bipartite_port_contention=cluster.bipartite_port_contention,
        wdm_wavelengths=cluster.wdm_wavelengths,
    )
    nic = ElectricalNICCalendar(
        cluster.num_workers,
        cluster.rdma_bandwidth_gb_s * 1e9,
        cluster.rdma_latency_s,
    )
    free = [0.0] * cluster.num_workers
    seen: set[str] = set()
    session_prev: dict[str, tuple[BlockRef, ...]] = {}
    session_last_worker: dict[str, int] = {}
    session_last_done: dict[str, float] = {}
    session_last_turn: dict[str, int] = {}
    session_ewma_gap: dict[str, float] = {}
    # Strictly causal chronological prefetch event queue: (wake_t, seq, sid, prev_turn_idx)
    prefetch_heap: list[tuple[float, int, str, int]] = []
    pf_event_seq = 0
    # Pending prefetches fired for each session since its last turn: sid -> (target_worker, prefetched_ids)
    pending_prefetch: dict[str, tuple[int, set[str]]] = {}
    # CONCUR (ICML '26) AIMD session-level concurrency window state (used only when spec.admission_control)
    active_sessions: dict[str, float] = {}  # sid -> last_completion_s
    concur_cwnd: float = float(max(2, cluster.num_workers))
    concur_max_cwnd: float = float(max(4, cluster.num_workers * 4))

    rows: list[dict] = []
    hbm_after: dict[tuple[str, int], tuple[str, ...]] = {}
    rng = random.Random(seed)
    run_id = (
        f"{spec.name}|{model_name}|{trace.name}|bw{cluster.ocs_bandwidth_gb_s:g}"
        f"|tau{cluster.ocs_reconfig_us:g}|seed{seed}"
    )
    status = prefill.calibration_status
    total_prefetch_hit_tokens = 0
    total_spurious_prefetch_bytes = 0
    total_inflight_joins = 0

    for index, request in enumerate(requests):
        # 1. Fire all strictly causal scheduled prefetch events whose wake_t <= request.arrival_s
        #    in TRUE chronological order (ZERO future peeking into request.arrival_s or pre_gap_s!).
        while prefetch_heap and prefetch_heap[0][0] <= request.arrival_s:
            wake_t, _, pf_sid, pf_turn_idx = heapq.heappop(prefetch_heap)
            # Skip stale event if session already advanced to a newer turn
            if pf_sid not in session_prev or session_last_turn.get(pf_sid, -1) != pf_turn_idx:
                continue
            pred_gap = session_ewma_gap.get(pf_sid, 0.50)
            pf_worker = _choose_prefetch_worker(
                spec=spec,
                index=index,
                cache=cache,
                free=free,
                launch_t=wake_t,
                prev_worker=session_last_worker.get(pf_sid),
                session_prev=session_prev[pf_sid],
            )
            ok_pf, pf_ids = _prefetch(
                spec=spec,
                session_id=pf_sid,
                worker=pf_worker,
                index=index,
                launch_t=wake_t,
                pred_gap=pred_gap,
                cache=cache,
                calendar=calendar,
                nic=nic,
                evictor=evictor,
                session_prev=session_prev,
                prefill=prefill,
                cluster=cluster,
                kv_bytes=kv_bytes,
                reuse_noise_std=reuse_noise_std,
                rng=rng,
            )
            if ok_pf and pf_ids:
                pending_prefetch[pf_sid] = (pf_worker, pf_ids)

        # Probation-filtered Agent Anchors (strictly causal): promote early prefix
        # blocks ONLY on their >=2nd observation so one-shot prompts never pollute HBM.
        if spec.anchor_prefix_blocks > 0:
            for block in request.blocks[: spec.anchor_prefix_blocks]:
                if cache.obs.get(block.block_id, 0) >= 1:
                    cache.anchors.add(block.block_id)

        # CONCUR AIMD session admission control baseline (strictly causal)
        admission_delay_s = 0.0
        if spec.admission_control and not spec.oracle:
            expired = [s for s, t_end in active_sessions.items() if request.arrival_s - t_end > 2.0]
            for s in expired:
                del active_sessions[s]
            if request.session_id not in active_sessions and len(active_sessions) >= int(max(2, concur_cwnd)):
                earliest_slot = min(active_sessions.values())
                admission_delay_s = max(0.0, earliest_slot - request.arrival_s)
                if admission_delay_s == 0.0:
                    admission_delay_s = min(0.08, max(0.005, max(0.0, min(free) - request.arrival_s)))

        sid = request.session_id
        prev_done = session_last_done.get(sid)
        # Update causal session EWMA pause estimator using non-trivial inter-turn pauses
        # (>= 0.15s) so sub-100ms automated fast steps (still hot in HBM) do not skew the
        # JIT wake timer for tool/user pauses into firing prematurely.
        if prev_done is not None:
            obs_gap = max(0.0, request.arrival_s - prev_done)
            if obs_gap >= 0.15:
                if sid in session_ewma_gap:
                    session_ewma_gap[sid] = 0.5 * session_ewma_gap[sid] + 0.5 * obs_gap
                else:
                    session_ewma_gap[sid] = obs_gap

        pf_record = pending_prefetch.pop(sid, None)
        prefetched = pf_record is not None
        prefetch_target_worker = pf_record[0] if pf_record is not None else None
        prefetched_ids = pf_record[1] if pf_record is not None else set()

        worker = _choose_worker(
            spec=spec,
            request=request,
            index=index,
            cache=cache,
            calendar=calendar,
            nic=nic,
            free=free,
            prefill=prefill,
            cluster=cluster,
            kv_bytes=kv_bytes,
            preferred_worker=prefetch_target_worker if prefetched else None,
            admission_delay_s=admission_delay_s,
        )

        # Track prefetch precision vs spurious prefetching (SYMPHONY metric)
        if prefetched and prefetched_ids:
            req_id_set = {b.block_id for b in request.blocks}
            for block in session_prev.get(sid, ()):
                if block.block_id in prefetched_ids:
                    if worker == prefetch_target_worker and block.block_id in req_id_set:
                        total_prefetch_hit_tokens += block.num_tokens
                    else:
                        total_spurious_prefetch_bytes += block.num_tokens * kv_bytes

        row, joined_inflight = _serve(
            spec=spec,
            request=request,
            index=index,
            worker=worker,
            cache=cache,
            calendar=calendar,
            nic=nic,
            evictor=evictor,
            free=free,
            seen=seen,
            prefill=prefill,
            cluster=cluster,
            kv_bytes=kv_bytes,
            prefetched=prefetched,
            run_id=run_id,
            status=status,
            noise_std=noise_std,
            rng=rng,
            admission_delay_s=admission_delay_s,
        )
        if joined_inflight:
            total_inflight_joins += 1
        rows.append(row)

        evicted_count = 0
        if spec.store and not spec.oracle:
            evicted_count = _retain(
                spec=spec,
                request=request,
                index=index,
                worker=worker,
                cache=cache,
                evictor=evictor,
                prefill=prefill,
                kv_bytes=kv_bytes,
                reuse_noise_std=reuse_noise_std,
                rng=rng,
                obs_gap_s=session_ewma_gap.get(sid, 0.0),
            )
        elif spec.oracle:
            for block in request.blocks:
                seen.add(block.block_id)

        if spec.admission_control and not spec.oracle:
            req_done = request.arrival_s + float(row["ttft_s"])
            active_sessions[sid] = req_done
            if evicted_count > 0 and float(row["hbm_hit_tokens"]) < 0.5 * max(1, request.input_tokens):
                concur_cwnd = max(2.0, concur_cwnd * 0.75)
            else:
                concur_cwnd = min(concur_max_cwnd, concur_cwnd + 0.5)

        if spec.store and not spec.oracle:
            for block_id in {block.block_id for block in request.blocks}:
                cache.obs[block_id] = cache.obs.get(block_id, 0) + 1
                cache.last_obs_s[block_id] = request.arrival_s
            session_prev[sid] = request.blocks
            session_last_worker[sid] = worker
            session_last_turn[sid] = request.turn_index
            done_t = request.arrival_s + float(row["ttft_s"])
            session_last_done[sid] = done_t
            # Arm strictly causal Just-In-Time (JIT) chronological prefetch for confirmed
            # multi-turn sessions (turn >= 1): wake up `lead_s` before predicted EWMA arrival,
            # scaled fairly to each policy's own fabric bandwidth (so 25 GB/s RDMA gets full
            # bytes/BW_rdma lead time while 200 GB/s OCS uses switch + optical burst lead time).
            if spec.prefetch and request.turn_index >= 1:
                est_gap = session_ewma_gap.get(sid, 0.50)
                if est_gap >= 0.03:
                    if spec.optical != "none":
                        lead_s = min(0.10, max(0.055, est_gap * 0.15))
                    else:
                        est_bytes = sum(b.num_tokens for b in request.blocks) * kv_bytes
                        est_xfer = est_bytes / max(1.0, cluster.rdma_bandwidth_gb_s * 1e9)
                        lead_s = min(est_gap * 0.45, max(0.055, est_xfer + 0.025))
                    wake_t = done_t + max(0.015, est_gap - lead_s)
                    pf_event_seq += 1
                    heapq.heappush(prefetch_heap, (wake_t, pf_event_seq, sid, request.turn_index))
            if record_hbm_snapshots:
                hbm_after[(spec.name, index)] = cache.ids("hbm", worker)

    # Account honestly for any speculative prefetches armed on final turns that never returned
    for pf_sid, (_, pf_ids) in pending_prefetch.items():
        for block in session_prev.get(pf_sid, ()):
            if block.block_id in pf_ids:
                total_spurious_prefetch_bytes += block.num_tokens * kv_bytes

    summary = _summarize(
        rows=rows,
        spec=spec,
        trace=trace,
        model_name=model_name,
        dtype_bytes=dtype_bytes,
        kv_bytes=kv_bytes,
        cluster=cluster,
        prefill=prefill,
        seed=seed,
        cache=cache,
        calendar=calendar,
        cost=cost_columns,
        quadratic=quadratic,
        run_id=run_id,
        slo_ttft_s=slo_ttft_s,
        reuse_noise_std=reuse_noise_std,
        prefetch_hit_tokens=total_prefetch_hit_tokens,
        spurious_prefetch_bytes=total_spurious_prefetch_bytes,
    )
    summary["inflight_joins"] = total_inflight_joins
    return summary, rows, hbm_after


def _prefer_optical_over_dram(
    spec: PolicySpec,
    cluster: Cluster,
    missing_tokens: int,
    kv_bytes: int,
    calendar: OpticalCalendar,
    now: float,
    src_worker: int | None = None,
) -> bool:
    """Queue-aware analytical fabric crossover: prefer OCS over local PCIe DRAM
    only when total optical latency (port wait + effective switch + transfer) is
    strictly lower than local DRAM transfer time."""
    if spec.optical == "none" or cluster.ocs_bandwidth_gb_s <= 0.0:
        return False
    missing_bytes = missing_tokens * kv_bytes
    if missing_bytes <= 0:
        return False
    t_dram = missing_bytes / (cluster.dram_bandwidth_gb_s * 1e9) if cluster.dram_bandwidth_gb_s > 0 else float("inf")
    active = calendar.active_targets(src_worker=src_worker)
    tau_eff = 0.0 if OPTICAL_POOL_TARGET in active else cluster.reconfig_s
    ocs_wait = calendar.min_wait_s(now, src_worker=src_worker)
    eff_ocs_bw = cluster.ocs_bandwidth_gb_s * max(1, cluster.wdm_wavelengths) * 1e9
    t_ocs = ocs_wait + tau_eff + cluster.ocs_latency_s + missing_bytes / eff_ocs_bw
    return t_ocs < t_dram


def _choose_prefetch_worker(
    *,
    spec: PolicySpec,
    index: int,
    cache: CacheStore,
    free: list[float],
    launch_t: float,
    prev_worker: int | None,
    session_prev: tuple[BlockRef, ...],
) -> int:
    """Pick the best worker for async prefetching: maximize existing prefix locality
    (so we only fetch the delta instead of duplicating entire prefixes), respect
    pinned-memory headroom, and keep low worker queue."""
    if not spec.routing:
        return min(
            range(cache.n_workers),
            key=lambda w: (max(0.0, free[w] - launch_t), (w - index) % cache.n_workers),
        )
    best_w = prev_worker if (prev_worker is not None and 0 <= prev_worker < cache.n_workers) else 0
    best_key = None
    for w in range(cache.n_workers):
        cap = cache.capacity[("hbm", w)]
        pinned = sum(e.nbytes for e in cache.rows[("hbm", w)].values() if e.pin_count > 0)
        pin_util = pinned / cap if cap > 0 else 1.0
        over_cap = 1 if pin_util >= spec.prefetch_utilization_cap else 0
        q = max(0.0, free[w] - launch_t)
        # Count leading blocks already hot in HBM on worker w
        hbm_w = cache.rows[("hbm", w)]
        local_hit_tok = 0
        for b in session_prev:
            if b.block_id in hbm_w:
                local_hit_tok += b.num_tokens
            else:
                break
        not_prev = 0 if (prev_worker is not None and w == prev_worker) else 1
        # Prioritize: (1) unpinned headroom, (2) low queue bucket (10ms), (3) highest local HBM hits, (4) prev_worker stickiness
        key = (over_cap, int(q / 0.01), -local_hit_tok, not_prev, pin_util, w)
        if best_key is None or key < best_key:
            best_key = key
            best_w = w
    return best_w


def _choose_worker(
    *,
    spec: PolicySpec,
    request: Request,
    index: int,
    cache: CacheStore,
    calendar: OpticalCalendar,
    nic: ElectricalNICCalendar,
    free: list[float],
    prefill: PrefillModel,
    cluster: Cluster,
    kv_bytes: int,
    preferred_worker: int | None = None,
    admission_delay_s: float = 0.0,
) -> int:
    workers = cache.n_workers
    if spec.oracle:
        return min(range(workers), key=lambda w: (free[w], w))
    if not spec.routing:
        if request.assigned_worker is not None:
            return int(request.assigned_worker) % workers
        return min(
            range(workers),
            key=lambda w: (max(0.0, free[w] - request.arrival_s), (w - index) % workers),
        )

    eff_arrival = request.arrival_s + admission_delay_s

    # Candidate worker pruning for very long 16-token block requests (exact fast path)
    if len(request.blocks) > 512 and workers > 2:
        b0 = request.blocks[0].block_id
        cands = {
            w
            for w in range(workers)
            if (b0 in cache.rows[("hbm", w)]) or (spec.use_dram and b0 in cache.rows[("dram", w)])
        }
        if preferred_worker is not None:
            cands.add(preferred_worker)
        by_q = sorted(range(workers), key=lambda w: (max(0.0, free[w] - eff_arrival), w))
        cands.add(by_q[0])
        cands.add(by_q[1])
        candidate_workers = sorted(cands)
    else:
        candidate_workers = list(range(workers))

    alpha = cluster.alpha_overlap
    best_w = candidate_workers[0]
    best_ttft = float("inf")
    best_q = float("inf")

    for w in candidate_workers:
        start = max(eff_arrival, free[w])
        queue_s = start - request.arrival_s
        local_tokens, offset, inflight_s = _leading_local(request.blocks, cache, w, start)
        active_peers = calendar.active_targets(src_worker=w) if spec.optical != "none" else None
        prefer_opt = _prefer_optical_over_dram(
            spec, cluster, request.input_tokens - local_tokens, kv_bytes, calendar, eff_arrival, src_worker=w
        )
        extension = _extension(
            request.blocks,
            offset,
            cache,
            w,
            spec,
            preferred_peers=active_peers,
            prefer_optical_over_dram=prefer_opt,
        )
        dram_b, ssd_b, rdma_b, segments, ext_tokens, _ = _segment_bytes(extension, w, kv_bytes, spec)
        lower_s = _tier_seconds(dram_b, cluster.dram_bandwidth_gb_s, 0.0) + _tier_seconds(
            ssd_b, cluster.ssd_bandwidth_gb_s, 0.0
        )
        if rdma_b > 0:
            lower_s += nic.transfer(start, w, rdma_b, commit=False)
        book_at = eff_arrival if spec.book_at_arrival else start
        plans = _transfer_plans(calendar, book_at, segments, spec, commit=False, src_worker=w)
        residual = max(0.0, max(p.end_s for p in plans) - start) if plans else 0.0
        fetch_transfer_s = max(inflight_s, residual + lower_s) if (inflight_s > 0 and (residual + lower_s) > 0) else (inflight_s + residual + lower_s)

        prefill_ext, dec_ext = prefill.request_times(local_tokens + ext_tokens, request.input_tokens - local_tokens - ext_tokens)
        est_fetch_ttft = layer_ttft(queue_s, fetch_transfer_s, prefill_ext, dec_ext, alpha)

        if spec.recompute_fallback and ext_tokens > 0:
            prefill_loc, dec_loc = prefill.request_times(local_tokens, request.input_tokens - local_tokens)
            est_abort_ttft = layer_ttft(queue_s, inflight_s, prefill_loc, dec_loc, alpha)
            est_ttft = min(est_fetch_ttft, est_abort_ttft)
        else:
            est_ttft = est_fetch_ttft

        # Sticky preference (2ms tolerance) for the prefetched worker to avoid spurious churn
        sticky_bonus = -0.002 if (preferred_worker is not None and w == preferred_worker) else 0.0
        eff_score = est_ttft + sticky_bonus
        if (eff_score < best_ttft - 1e-12) or (abs(eff_score - best_ttft) <= 1e-12 and queue_s < best_q):
            best_ttft = eff_score
            best_q = queue_s
            best_w = w
    return best_w


def _leading_local(
    blocks: tuple[BlockRef, ...],
    cache: CacheStore,
    worker: int,
    start_s: float = 0.0,
) -> tuple[int, int, float]:
    """Return (leading_hbm_tokens, first_missing_index, inflight_residual_wait_s)."""
    hbm = cache.rows[("hbm", worker)]
    tokens = 0
    max_inflight_wait = 0.0
    for index, block in enumerate(blocks):
        entry = hbm.get(block.block_id)
        if entry is None:
            return tokens, index, max_inflight_wait
        if entry.ready_at_s > start_s:
            max_inflight_wait = max(max_inflight_wait, entry.ready_at_s - start_s)
        tokens += block.num_tokens
    return tokens, len(blocks), max_inflight_wait


def _extension(
    blocks: tuple[BlockRef, ...],
    start: int,
    cache: CacheStore,
    worker: int,
    spec: PolicySpec,
    preferred_peers: set[int] | None = None,
    prefer_optical_over_dram: bool = False,
):
    found = []
    optical_target = None
    for block in blocks[start:]:
        location = cache.best_location(
            block.block_id,
            worker,
            spec,
            preferred_peers=preferred_peers,
            prefer_optical_over_dram=prefer_optical_over_dram,
        )
        if location is None:
            break
        if spec.optical == "reservation" and location.tier == "remote_hbm":
            if optical_target is None:
                optical_target = location.worker
            elif location.worker != optical_target:
                break
        found.append((block, location))
    return found


def _score(
    cache: CacheStore,
    block_id: str,
    obs_gap_s: float,
    spec: PolicySpec,
    noise: float = 0.0,
) -> float:
    return p_reuse(
        cache.obs.get(block_id, 0),
        block_id in cache.anchors,
        obs_gap_s,
        spec.recency_tau_s,
        noise=noise,
    )


def _prefetch(
    *,
    spec: PolicySpec,
    session_id: str,
    worker: int,
    index: int,
    launch_t: float,
    pred_gap: float,
    cache: CacheStore,
    calendar: OpticalCalendar,
    nic: ElectricalNICCalendar,
    evictor,
    session_prev: dict[str, tuple[BlockRef, ...]],
    prefill: PrefillModel,
    cluster: Cluster,
    kv_bytes: int,
    reuse_noise_std: float,
    rng: random.Random,
) -> tuple[bool, set[str]]:
    """Strictly causal async inter-turn prefetch launched chronologically at `launch_t`.

    Never peeks at Turn t's arrival time or future gap. Uses past EWMA `pred_gap`
    and real-time port congestion at `launch_t`; if Turn t arrives early before
    completion, blocks carry `ready_at_s = pf_end_s` for Strata-style in-flight join.
    """
    if not spec.prefetch or pred_gap < 0.015:
        return False, set()
    previous = session_prev.get(session_id)
    if not previous:
        return False, set()
    capacity = cache.capacity[("hbm", worker)]
    if capacity <= 0:
        return False, set()
    pinned_bytes = sum(e.nbytes for e in cache.rows[("hbm", worker)].values() if e.pin_count > 0)
    if pinned_bytes / capacity >= spec.prefetch_utilization_cap:
        return False, set()
    budget = int(spec.prefetch_utilization_cap * capacity - pinned_bytes)

    is_optical = spec.optical != "none"

    # Strictly causal congestion gate at `launch_t` (never uses future pre_gap_s)
    if is_optical:
        if calendar.min_wait_s(launch_t, src_worker=worker) > max(0.05, pred_gap * 0.40):
            return False, set()
    elif spec.use_rdma:
        if nic.min_wait_s(launch_t, worker) > max(0.05, pred_gap * 0.40):
            return False, set()

    hbm_local = cache.rows[("hbm", worker)]
    dram_local = cache.rows[("dram", worker)]
    pool_rows = cache.rows[("optical_pool", None)]
    rdma_rows = cache.rows[("rdma", None)]
    active_peers = calendar.active_targets(src_worker=worker) if is_optical else set()

    prev_total_tokens = sum(b.num_tokens for b in previous)
    prev_total_cold = prefill.cold_parts(prev_total_tokens)[0] if prev_total_tokens > 0 else 0.0

    # Collect missing blocks from previous turn (strictly causal: session_prev only)
    dram_blocks: list[tuple[int, BlockRef, float, float]] = []
    remote_groups: dict[int, list[tuple[int, BlockRef, float, float]]] = {}
    rdma_blocks: list[tuple[int, BlockRef, float, float]] = []
    cum_tok = 0
    for pos, block in enumerate(previous):
        prev_cold = prefill.cold_parts(cum_tok)[0] if cum_tok > 0 else 0.0
        protected_saved = max(1e-9, prev_total_cold - prev_cold)
        cum_tok += block.num_tokens
        if block.block_id in hbm_local:
            continue
        noise = rng.gauss(0.0, reuse_noise_std) if reuse_noise_std > 0.0 else 0.0
        sc = _score(cache, block.block_id, pred_gap, spec, noise=noise)
        if sc < 0.35:
            continue
        # Freshly prefetched imminent-turn blocks get a 2.0x imminent-reuse grace multiplier
        # so they are retained across the JIT window before Turn t arrives.
        ev_score = _prefix_gdsf_score(min(2.0, sc * 2.0), protected_saved, block.num_tokens * kv_bytes, prev_total_cold)
        if spec.use_dram and block.block_id in dram_local:
            dram_blocks.append((pos, block, sc, ev_score))
            continue
        if is_optical:
            # Always coalesce async background prefetches through the unified Optical KV Pool
            # first to keep standing circuits latched (tau_eff = 0, maximizing MEMS duty cycle!).
            target: int | None = None
            owners = cache._hbm_owners.get(block.block_id)
            if spec.use_optical_pool and block.block_id in pool_rows:
                target = OPTICAL_POOL_TARGET
            elif active_peers and owners:
                for peer in active_peers:
                    if peer != worker and peer in owners:
                        target = peer
                        break
            if target is None and owners:
                for other in sorted(owners):
                    if other != worker:
                        target = other
                        break
            if target is not None:
                remote_groups.setdefault(target, []).append((pos, block, sc, ev_score))
        elif spec.use_rdma:
            owners = cache._hbm_owners.get(block.block_id)
            has_peer = bool(owners and any(other != worker for other in owners))
            if has_peer or block.block_id in rdma_rows:
                rdma_blocks.append((pos, block, sc, ev_score))

    prefetched_ids: set[str] = set()
    applied = False
    remaining_budget = budget
    sid = session_id

    # Path A: Local DRAM prefetch over PCIe during inter-turn gap
    if dram_blocks and remaining_budget > 0:
        chosen_dram: list[tuple[int, BlockRef, float, float]] = []
        dram_bytes = 0
        for pos, block, sc, ev_sc in dram_blocks:
            nb = block.num_tokens * kv_bytes
            if dram_bytes + nb > remaining_budget:
                break
            chosen_dram.append((pos, block, sc, ev_sc))
            dram_bytes += nb
        dram_s = _tier_seconds(dram_bytes, cluster.dram_bandwidth_gb_s, 0.0)
        if chosen_dram and dram_s <= pred_gap * 1.5:
            ready_t = launch_t + dram_s
            for pos, block, sc, ev_sc in chosen_dram:
                cache.upsert(
                    "hbm",
                    worker,
                    block.block_id,
                    block.num_tokens,
                    launch_t,
                    sc,
                    evictor,
                    index,
                    pin=False,
                    evict_score=ev_sc,
                    auto_trim=False,
                    session_id=sid,
                    block_pos=pos,
                    ready_at_s=ready_t,
                )
                prefetched_ids.add(block.block_id)
            remaining_budget -= dram_bytes
            applied = True

    # Path B (Optical): Strictly causal off-critical-path Windowed Burst Prefetch
    if is_optical and remote_groups and remaining_budget > 0:
        target = max(
            remote_groups,
            key=lambda peer: (
                sum(b.num_tokens for _, b, _, _ in remote_groups[peer])
                * (1.20 if peer in active_peers else 1.0),
                -peer,
            ),
        )
        chosen: list[tuple[int, BlockRef, float, float]] = []
        total_bytes = 0
        weighted_tokens = 0.0
        for pos, block, sc, ev_sc in remote_groups[target]:
            nb = block.num_tokens * kv_bytes
            if total_bytes + nb > remaining_budget:
                break
            chosen.append((pos, block, sc, ev_sc))
            total_bytes += nb
            weighted_tokens += sc * block.num_tokens
        if total_bytes > 0:
            preview = calendar.transfer_sequence(launch_t, [(target, total_bytes)], commit=False, src_worker=worker)
            if preview:
                pf_end_s = preview[-1].end_s
                burst_dur = pf_end_s - launch_t
                reconfig_s = sum(p.reconfig_s for p in preview)
                expected_saved = prefill.cold_parts(int(round(weighted_tokens)))[0]
                # Asynchronous off-path benefit gate:
                # Standing circuits (reconfig_s == 0) are always worth prefetching when
                # port is idle; unlatched MEMS switches require burst to amortize switch
                # overhead (avoid mirror churn on tiny 1K bursts while admitting 8K-170K!).
                worth_switching = (reconfig_s <= 1e-9) or (expected_saved > 0.25 * reconfig_s) or (total_bytes >= 64 * 1024 * 1024)
                fits_causal_window = burst_dur <= max(0.08, pred_gap * 1.25)
                if worth_switching and fits_causal_window:
                    committed = calendar.transfer_sequence(
                        launch_t, [(target, total_bytes)], commit=True, src_worker=worker
                    )
                    ready_t = committed[-1].end_s
                    for pos, block, sc, ev_sc in chosen:
                        cache.upsert(
                            "hbm",
                            worker,
                            block.block_id,
                            block.num_tokens,
                            launch_t,
                            sc,
                            evictor,
                            index,
                            pin=False,
                            evict_score=ev_sc,
                            auto_trim=False,
                            session_id=sid,
                            block_pos=pos,
                            ready_at_s=ready_t,
                        )
                        prefetched_ids.add(block.block_id)
                    applied = True

    # Path B (Electrical RDMA SOTA): Identical strictly causal NIC prefetch + in-flight join
    if (not is_optical) and rdma_blocks and remaining_budget > 0:
        chosen_r: list[tuple[int, BlockRef, float, float]] = []
        rdma_bytes = 0
        for pos, block, sc, ev_sc in rdma_blocks:
            nb = block.num_tokens * kv_bytes
            if rdma_bytes + nb > remaining_budget:
                break
            chosen_r.append((pos, block, sc, ev_sc))
            rdma_bytes += nb
        if chosen_r and rdma_bytes > 0:
            preview_s = nic.transfer(launch_t, worker, rdma_bytes, commit=False)
            if preview_s <= max(0.08, pred_gap * 1.25):
                dur_s = nic.transfer(launch_t, worker, rdma_bytes, commit=True)
                ready_t = launch_t + dur_s
                for pos, block, sc, ev_sc in chosen_r:
                    cache.upsert(
                        "hbm",
                        worker,
                        block.block_id,
                        block.num_tokens,
                        launch_t,
                        sc,
                        evictor,
                        index,
                        pin=False,
                        evict_score=ev_sc,
                        auto_trim=False,
                        session_id=sid,
                        block_pos=pos,
                        ready_at_s=ready_t,
                    )
                    prefetched_ids.add(block.block_id)
                applied = True

    if applied:
        evicted_hbm = cache.trim("hbm", worker, evictor, index)
        if evicted_hbm:
            _spill_hbm_victims(spec=spec, worker=worker, index=index, cache=cache, evictor=evictor, evicted_hbm=evicted_hbm)

    return applied, prefetched_ids


def _serve(
    *,
    spec: PolicySpec,
    request: Request,
    index: int,
    worker: int,
    cache: CacheStore,
    calendar: OpticalCalendar,
    nic: ElectricalNICCalendar,
    evictor,
    free: list[float],
    seen: set[str],
    prefill: PrefillModel,
    cluster: Cluster,
    kv_bytes: int,
    prefetched: bool,
    run_id: str,
    status: str,
    noise_std: float,
    rng: random.Random,
    admission_delay_s: float = 0.0,
) -> tuple[dict, bool]:
    alpha = cluster.alpha_overlap
    if spec.oracle:
        start = request.arrival_s
        queue_s = 0.0
    else:
        eff_arrival = request.arrival_s + admission_delay_s
        start = max(eff_arrival, free[worker])
        queue_s = start - request.arrival_s

    joined_inflight = False
    counts = {"hbm": 0, "dram": 0, "ssd": 0, "rdma": 0, "remote_hbm": 0, "optical_pool": 0}
    if spec.name == "no_cache":
        prefill_s, decode_s = prefill.request_times(0, request.input_tokens)
        transfer_s = 0.0
        reconfig_s = 0.0
        aborted = False
        accepted = 0
        transfer_bytes = 0
    elif spec.oracle:
        accepted = 0
        for block in request.blocks:
            if block.block_id not in seen:
                break
            accepted += block.num_tokens
        prefill_s, decode_s = prefill.request_times(accepted, request.input_tokens - accepted)
        transfer_s = 0.0
        reconfig_s = 0.0
        aborted = False
        counts["hbm"] = accepted
        transfer_bytes = 0
    else:
        local_tokens, offset, inflight_s = _leading_local(request.blocks, cache, worker, start)
        if inflight_s > 1e-9:
            joined_inflight = True
        active_peers = calendar.active_targets(src_worker=worker) if spec.optical != "none" else None
        eff_arrival = request.arrival_s + admission_delay_s
        prefer_opt = _prefer_optical_over_dram(
            spec, cluster, request.input_tokens - local_tokens, kv_bytes, calendar, eff_arrival, src_worker=worker
        )
        extension = _extension(
            request.blocks,
            offset,
            cache,
            worker,
            spec,
            preferred_peers=active_peers,
            prefer_optical_over_dram=prefer_opt,
        )
        dram_b, ssd_b, rdma_b, segments, ext_tokens, moved_tokens = _segment_bytes(
            extension, worker, kv_bytes, spec
        )
        local_tier_s = _tier_seconds(dram_b, cluster.dram_bandwidth_gb_s, 0.0) + _tier_seconds(
            ssd_b, cluster.ssd_bandwidth_gb_s, 0.0
        )
        rdma_preview_s = nic.transfer(start, worker, rdma_b, commit=False) if rdma_b > 0 else 0.0
        book_at = eff_arrival if spec.book_at_arrival else start
        plans = _transfer_plans(calendar, book_at, segments, spec, commit=False, src_worker=worker)
        reconfig_paid = sum(plan.reconfig_s for plan in plans)
        preview_residual = max(0.0, max(p.end_s for p in plans) - start) if plans else 0.0
        fetch_xfer = (
            max(inflight_s, preview_residual + local_tier_s + rdma_preview_s)
            if (inflight_s > 0.0 and (preview_residual + local_tier_s + rdma_preview_s) > 0.0)
            else (inflight_s + preview_residual + local_tier_s + rdma_preview_s)
        )

        # Law 1 + Law 3 closed-form Selective Recompute Abort: compare true overlapped TTFT
        prefill_ext, dec_ext = prefill.request_times(local_tokens + ext_tokens, request.input_tokens - local_tokens - ext_tokens)
        prefill_loc, dec_loc = prefill.request_times(local_tokens, request.input_tokens - local_tokens)
        ttft_fetch = layer_ttft(queue_s, fetch_xfer, prefill_ext, dec_ext, alpha)
        ttft_abort = layer_ttft(queue_s, inflight_s, prefill_loc, dec_loc, alpha)

        aborted = bool(spec.recompute_fallback and ext_tokens > 0 and ttft_fetch > ttft_abort + 1e-12)
        if aborted:
            transfer_s = inflight_s
            reconfig_s = 0.0
            transfer_bytes = 0
            accepted = local_tokens
            counts["hbm"] = local_tokens
            prefill_s, decode_s = prefill_loc, dec_loc
        else:
            rdma_s = nic.transfer(start, worker, rdma_b, commit=True) if rdma_b > 0 else 0.0
            if segments:
                plans = _transfer_plans(calendar, book_at, segments, spec, commit=True, src_worker=worker)
            residual = max(0.0, max(p.end_s for p in plans) - start) if plans else 0.0
            rem_xfer = residual + local_tier_s + rdma_s
            transfer_s = max(inflight_s, rem_xfer) if (inflight_s > 0.0 and rem_xfer > 0.0) else (inflight_s + rem_xfer)
            reconfig_s = reconfig_paid
            transfer_bytes = dram_b + ssd_b + rdma_b + sum(nbytes for _, nbytes in segments)
            accepted = local_tokens + ext_tokens
            counts["hbm"] = local_tokens + moved_tokens.get("hbm_tail", 0)
            counts["dram"] = moved_tokens["dram"]
            counts["ssd"] = moved_tokens["ssd"]
            counts["rdma"] = moved_tokens["rdma"]
            counts["remote_hbm"] = moved_tokens["remote_hbm"]
            counts["optical_pool"] = moved_tokens["optical_pool"]
            prefill_s, decode_s = prefill_ext, dec_ext

    if noise_std > 0.0:
        prefill_s *= max(0.05, 1.0 + rng.gauss(0.0, noise_std))
        transfer_s *= max(0.05, 1.0 + rng.gauss(0.0, noise_std))

    ttft_s = layer_ttft(queue_s, transfer_s, prefill_s, decode_s, alpha)
    if not spec.oracle:
        # Optional decode continuous-batching queue coupling (Physics Fidelity Mode):
        # models amortized multi-token decode execution under chunked continuous batching
        decode_busy_s = 0.0
        if cluster.couple_decode_queue and request.output_tokens > 1:
            decode_busy_s = (request.output_tokens - 1) * decode_s / float(max(1, cluster.decode_batch_slots))
        free[worker] = start + (ttft_s - queue_s) + decode_busy_s
    recomputed = request.input_tokens - accepted
    hit_rate = 0.0 if request.input_tokens == 0 else accepted / request.input_tokens
    return (
        {
            "run_id": run_id,
            "policy": spec.name,
            "request_id": request.request_id,
            "session_id": request.session_id,
            "turn_index": request.turn_index,
            "arrival_s": request.arrival_s,
            "worker": worker,
            "input_tokens": request.input_tokens,
            "output_tokens": request.output_tokens,
            "hbm_hit_tokens": counts["hbm"],
            "dram_hit_tokens": counts["dram"],
            "ssd_hit_tokens": counts["ssd"],
            "rdma_hit_tokens": counts["rdma"],
            "remote_hbm_hit_tokens": counts["remote_hbm"],
            "optical_pool_hit_tokens": counts["optical_pool"],
            "transferred_tokens": max(0, accepted - counts["hbm"]),
            "recomputed_tokens": recomputed,
            "hit_rate": hit_rate,
            "ttft_s": ttft_s,
            "queue_s": queue_s,
            "transfer_s": transfer_s,
            "reconfig_s": reconfig_s,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "aborted": aborted,
            "prefetch_applied": prefetched,
            "calibration_status": status,
            "transfer_bytes": transfer_bytes,
        },
        joined_inflight,
    )


def _transfer_plans(
    calendar: OpticalCalendar,
    book_at: float,
    segments,
    spec: PolicySpec,
    commit: bool,
    src_worker: int | None = None,
):
    """Execute or preview optical burst sequence on OCS Port Calendar."""
    if not segments:
        return []
    plans = calendar.transfer_sequence(book_at, segments, commit=commit, src_worker=src_worker)
    if commit and spec.optical == "naive":
        for port in calendar.ports:
            port.target = None
        if calendar.bipartite_port_contention:
            for wp in calendar.worker_ports:
                wp.target = None
    return plans


def _segment_bytes(extension, home: int, kv_bytes: int, spec: PolicySpec):
    """Aggregate extension blocks by tier and coalesce all remote optical blocks per peer."""
    dram = ssd = rdma = 0
    optical_by_target: dict[int, int] = {}
    target_order: list[int] = []
    ext_tokens = 0
    optical = spec.optical in {"naive", "reservation", "keep"}
    moved = {"dram": 0, "ssd": 0, "rdma": 0, "remote_hbm": 0, "optical_pool": 0, "hbm_tail": 0}
    for block, location in extension:
        ext_tokens += block.num_tokens
        nbytes = block.num_tokens * kv_bytes
        if location.tier == "hbm" and location.worker == home:
            moved["hbm_tail"] += block.num_tokens
            continue
        if location.tier == "dram":
            dram += nbytes
            moved["dram"] += block.num_tokens
        elif location.tier == "ssd":
            ssd += nbytes
            moved["ssd"] += block.num_tokens
        elif location.tier == "rdma":
            rdma += nbytes
            moved["rdma"] += block.num_tokens
        elif location.tier == "optical_pool":
            moved["optical_pool"] += block.num_tokens
            if OPTICAL_POOL_TARGET not in optical_by_target:
                optical_by_target[OPTICAL_POOL_TARGET] = 0
                target_order.append(OPTICAL_POOL_TARGET)
            optical_by_target[OPTICAL_POOL_TARGET] += nbytes
        elif location.tier == "remote_hbm" and location.worker is not None:
            if optical:
                moved["remote_hbm"] += block.num_tokens
                peer = location.worker
                if peer not in optical_by_target:
                    optical_by_target[peer] = 0
                    target_order.append(peer)
                optical_by_target[peer] += nbytes
            else:
                rdma += nbytes
                moved["rdma"] += block.num_tokens
    segments = [(target, optical_by_target[target]) for target in target_order]
    return dram, ssd, rdma, segments, ext_tokens, moved


def _tier_seconds(nbytes: int, bandwidth_gb_s: float, latency_us: float) -> float:
    if nbytes <= 0:
        return 0.0
    return latency_us * 1e-6 + nbytes / (bandwidth_gb_s * 1e9)


def _retain(
    *,
    spec: PolicySpec,
    request: Request,
    index: int,
    worker: int,
    cache: CacheStore,
    evictor,
    prefill: PrefillModel,
    kv_bytes: int,
    reuse_noise_std: float,
    rng: random.Random,
    obs_gap_s: float = 0.0,
) -> int:
    """Retain active request blocks in HBM (pinned during service), then spill HBM evictions.
    Returns count of HBM evicted blocks (used by CONCUR admission control signal)."""
    pinned = []
    seen_ids: set[str] = set()
    need_scores = bool(spec.placement or spec.eviction == "reuse")
    total_cold = prefill.cold_parts(request.input_tokens)[0] if need_scores else 0.0
    cum_tokens = 0
    use_dram = spec.use_dram
    use_ssd = spec.use_ssd
    dram_rows = cache.rows[("dram", worker)] if use_dram else None
    ssd_rows = cache.rows[("ssd", worker)] if use_ssd else None
    sid = request.session_id
    for pos, block in enumerate(request.blocks):
        if need_scores:
            prev_cold = prefill.cold_parts(cum_tokens)[0] if cum_tokens > 0 else 0.0
            protected_saved = max(1e-9, total_cold - prev_cold)
            cum_tokens += block.num_tokens

        bid = block.block_id
        if bid in seen_ids:
            continue
        seen_ids.add(bid)
        if dram_rows is not None and bid in dram_rows:
            cache.drop_if_unpinned("dram", worker, bid, count_eviction=False)
        if ssd_rows is not None and bid in ssd_rows:
            cache.drop_if_unpinned("ssd", worker, bid, count_eviction=False)

        if need_scores:
            noise = rng.gauss(0.0, reuse_noise_std) if (spec.eviction == "reuse" and reuse_noise_std > 0.0) else 0.0
            sc = _score(cache, bid, obs_gap_s, spec, noise=noise)
            nbytes = max(1, block.num_tokens * kv_bytes)
            ev_score = _prefix_gdsf_score(sc, protected_saved, nbytes, total_cold)
        else:
            sc = 0.0
            ev_score = 0.0
        entry = cache.upsert(
            "hbm",
            worker,
            bid,
            block.num_tokens,
            request.arrival_s,
            sc,
            evictor,
            index,
            pin=True,
            evict_score=ev_score,
            session_id=sid,
            block_pos=pos,
            ready_at_s=0.0,
        )
        pinned.append(entry)

    cache.unpin_many(pinned)
    evicted_hbm = cache.trim("hbm", worker, evictor, index)
    if not evicted_hbm:
        return 0
    _spill_hbm_victims(spec=spec, worker=worker, index=index, cache=cache, evictor=evictor, evicted_hbm=evicted_hbm)
    return len(evicted_hbm)


def _spill_hbm_victims(
    *,
    spec: PolicySpec,
    worker: int,
    index: int,
    cache: CacheStore,
    evictor,
    evicted_hbm: list,
) -> None:
    """Multi-tier spill on HBM capacity pressure preserving prefix-dependency scores across all tiers."""
    hbm_aging = cache._aging.get(("hbm", worker), 0.0)
    if spec.placement:
        dram_cap = cache.capacity[("dram", worker)]
        pool_cap = cache.capacity[("optical_pool", None)]
        for victim in evicted_hbm:
            is_anchor = victim.block_id in cache.anchors
            sc = victim.reuse_score
            dram_pressured = dram_cap > 0 and (cache.used[("dram", worker)] / dram_cap) >= 0.85
            pool_pressured = pool_cap > 0 and (cache.used[("optical_pool", None)] / pool_cap) >= 0.85
            admit_dram = (not dram_pressured) or is_anchor or (sc >= spec.tier_dram_min_p)
            admit_pool = (not pool_pressured) or is_anchor or (sc >= spec.tier_dram_min_p)
            # Preserve prefix-dependency ordering (root > leaf) when spilling to DRAM/Optical Pool
            deaged_ev = victim.evict_score - hbm_aging
            raw_ev = deaged_ev if deaged_ev > 1e-12 else ((sc * 1e-6) / max(1, victim.nbytes))
            if spec.use_dram and admit_dram:
                cache.upsert(
                    "dram",
                    worker,
                    victim.block_id,
                    victim.num_tokens,
                    victim.last_access_s,
                    sc,
                    evictor,
                    index,
                    pin=False,
                    evict_score=raw_ev,
                    auto_trim=False,
                    session_id=victim.session_id,
                    block_pos=victim.block_pos,
                )
            elif spec.use_ssd and (is_anchor or sc >= spec.tier_ssd_min_p):
                cache.upsert(
                    "ssd",
                    worker,
                    victim.block_id,
                    victim.num_tokens,
                    victim.last_access_s,
                    sc,
                    evictor,
                    index,
                    pin=False,
                    evict_score=raw_ev,
                    session_id=victim.session_id,
                    block_pos=victim.block_pos,
                )
            if spec.use_optical_pool and admit_pool:
                cache.upsert(
                    "optical_pool",
                    None,
                    victim.block_id,
                    victim.num_tokens,
                    victim.last_access_s,
                    sc,
                    evictor,
                    index,
                    pin=False,
                    evict_score=raw_ev,
                    session_id=victim.session_id,
                    block_pos=victim.block_pos,
                )
        if spec.use_dram:
            evicted_dram = cache.trim("dram", worker, evictor, index)
            if spec.use_ssd:
                dram_aging = cache._aging.get(("dram", worker), 0.0)
                for dv in evicted_dram:
                    if dv.reuse_score >= spec.tier_ssd_min_p or dv.block_id in cache.anchors:
                        deaged_dv = dv.evict_score - dram_aging
                        dv_ev = deaged_dv if deaged_dv > 1e-12 else ((dv.reuse_score * 1e-6) / max(1, dv.nbytes))
                        cache.upsert(
                            "ssd",
                            worker,
                            dv.block_id,
                            dv.num_tokens,
                            dv.last_access_s,
                            dv.reuse_score,
                            evictor,
                            index,
                            pin=False,
                            evict_score=dv_ev,
                            session_id=dv.session_id,
                            block_pos=dv.block_pos,
                        )
    else:
        if spec.use_dram:
            for victim in evicted_hbm:
                cache.upsert(
                    "dram",
                    worker,
                    victim.block_id,
                    victim.num_tokens,
                    victim.last_access_s,
                    victim.reuse_score,
                    evictor,
                    index,
                    pin=False,
                    evict_score=victim.evict_score,
                    auto_trim=False,
                    session_id=victim.session_id,
                    block_pos=victim.block_pos,
                )
            evicted_dram = cache.trim("dram", worker, evictor, index)
            if spec.use_ssd:
                for dv in evicted_dram:
                    cache.upsert(
                        "ssd",
                        worker,
                        dv.block_id,
                        dv.num_tokens,
                        dv.last_access_s,
                        dv.reuse_score,
                        evictor,
                        index,
                        pin=False,
                        evict_score=dv.evict_score,
                        session_id=dv.session_id,
                        block_pos=dv.block_pos,
                    )
        if spec.use_rdma:
            for victim in evicted_hbm:
                cache.upsert(
                    "rdma",
                    None,
                    victim.block_id,
                    victim.num_tokens,
                    victim.last_access_s,
                    victim.reuse_score,
                    evictor,
                    index,
                    pin=False,
                    evict_score=victim.evict_score,
                    session_id=victim.session_id,
                    block_pos=victim.block_pos,
                )


def _summarize(
    *,
    rows,
    spec: PolicySpec,
    trace: Trace,
    model_name: str,
    dtype_bytes: int,
    kv_bytes: int,
    cluster: Cluster,
    prefill: PrefillModel,
    seed: int,
    cache: CacheStore,
    calendar: OpticalCalendar,
    cost: dict,
    quadratic: bool,
    run_id: str,
    slo_ttft_s: float,
    reuse_noise_std: float,
    prefetch_hit_tokens: int,
    spurious_prefetch_bytes: int,
):
    def mean(key: str) -> float:
        if not rows:
            return 0.0
        return sum(float(row[key]) for row in rows) / len(rows)

    input_tokens = sum(int(row["input_tokens"]) for row in rows)
    recomputed = sum(int(row["recomputed_tokens"]) for row in rows)
    hbm_hits = sum(int(row["hbm_hit_tokens"]) for row in rows)
    dram_hits = sum(int(row["dram_hit_tokens"]) for row in rows)
    ssd_hits = sum(int(row["ssd_hit_tokens"]) for row in rows)
    rdma_hits = sum(int(row["rdma_hit_tokens"]) for row in rows)
    remote_hits = sum(int(row["remote_hbm_hit_tokens"]) for row in rows)
    pool_hits = sum(int(row["optical_pool_hit_tokens"]) for row in rows)

    def rate(numerator: int) -> float:
        return 0.0 if input_tokens == 0 else numerator / input_tokens

    ttfts = [float(row["ttft_s"]) for row in rows]
    slo_met = sum(1 for val in ttfts if val <= slo_ttft_s)
    slo_attainment = (slo_met / len(ttfts)) if ttfts else 0.0
    if rows:
        t_start = min(float(r["arrival_s"]) for r in rows)
        t_end = max(float(r["arrival_s"]) + float(r["ttft_s"]) for r in rows)
        makespan = max(1e-6, t_end - t_start)
    else:
        makespan = 1.0
    goodput_rps = slo_met / makespan

    is_optical = spec.optical != "none"
    if is_optical:
        fabric_watts = cluster.num_workers * float(cost.get("ocs_port_power_w", 0.39))
        fabric_dollars = cluster.num_workers * float(cost.get("ocs_port_cost_usd", 520.0))
    else:
        fabric_watts = cluster.num_workers * float(cost.get("electrical_400g_port_power_w", 39.4))
        low = float(cost.get("electrical_400g_port_cost_usd_low", 583.0))
        high = float(cost.get("electrical_400g_port_cost_usd_high", 1090.0))
        fabric_dollars = cluster.num_workers * 0.5 * (low + high)

    goodput_per_watt = goodput_rps / fabric_watts if fabric_watts > 0 else 0.0
    goodput_per_dollar = (goodput_rps / fabric_dollars) * 1000.0 if fabric_dollars > 0 else 0.0

    summary = {
        "run_id": run_id,
        "policy": spec.name,
        "model": model_name,
        "dtype_bytes": dtype_bytes,
        "kv_bytes_per_token": kv_bytes,
        "trace_name": trace.name,
        "trace_kind": trace.kind,
        "seed": seed,
        "num_requests": len(rows),
        "num_workers": cluster.num_workers,
        "hbm_bytes_per_worker": cluster.hbm_bytes_per_worker,
        "dram_bytes_per_worker": cluster.dram_bytes_per_worker,
        "ssd_bytes_per_worker": cluster.ssd_bytes_per_worker,
        "rdma_pool_bytes": cluster.rdma_pool_bytes,
        "ocs_concurrent_links": cluster.ocs_concurrent_links,
        "bandwidth_gb_s": cluster.ocs_bandwidth_gb_s,
        "latency_us": cluster.ocs_latency_us,
        "reconfig_us": cluster.ocs_reconfig_us,
        "rdma_bandwidth_gb_s": cluster.rdma_bandwidth_gb_s,
        "rdma_latency_us": cluster.rdma_latency_us,
        "dram_bandwidth_gb_s": cluster.dram_bandwidth_gb_s,
        "ssd_bandwidth_gb_s": cluster.ssd_bandwidth_gb_s,
        "alpha_overlap": cluster.alpha_overlap,
        "hardware_profile": prefill.profile,
        "scaling_label": prefill.scaling_label,
        "calibration_status": prefill.calibration_status,
        "quadratic_extrapolation": quadratic,
        "reuse_noise_std": reuse_noise_std,
        "enable_routing": spec.routing,
        "enable_recompute_fallback": spec.recompute_fallback,
        "enable_placement": spec.placement,
        "enable_prefetch": spec.prefetch,
        "enable_eviction": spec.eviction == "reuse",
        "prefetch_utilization_cap": spec.prefetch_utilization_cap,
        "input_tokens": input_tokens,
        "recomputed_tokens": recomputed,
        "hit_rate": rate(input_tokens - recomputed),
        "hbm_hit_rate": rate(hbm_hits),
        "dram_hit_rate": rate(dram_hits),
        "ssd_hit_rate": rate(ssd_hits),
        "rdma_hit_rate": rate(rdma_hits),
        "remote_hbm_hit_rate": rate(remote_hits),
        "optical_pool_hit_rate": rate(pool_hits),
        "mean_ttft_s": mean("ttft_s"),
        "p50_ttft_s": percentile(ttfts, 0.50),
        "p95_ttft_s": percentile(ttfts, 0.95),
        "p99_ttft_s": percentile(ttfts, 0.99),
        "mean_queue_s": mean("queue_s"),
        "mean_transfer_s": mean("transfer_s"),
        "mean_reconfig_s": mean("reconfig_s"),
        "mean_prefill_s": mean("prefill_s"),
        "mean_decode_s": mean("decode_s"),
        "mean_tpot_s": mean("decode_s"),
        "abort_count": sum(1 for row in rows if row["aborted"]),
        "prefetch_bursts": sum(1 for row in rows if row["prefetch_applied"]),
        "prefetch_hit_tokens": prefetch_hit_tokens,
        "spurious_prefetch_bytes": spurious_prefetch_bytes,
        "ocs_active_s": calendar.total_active_s,
        "ocs_reconfig_s": calendar.total_reconfig_s,
        "ocs_duty_cycle": calendar.duty_cycle,
        "standing_circuit_hits": calendar.standing_hits,
        "slo_ttft_s": slo_ttft_s,
        "slo_attainment": slo_attainment,
        "goodput_rps": goodput_rps,
        "goodput_per_watt": goodput_per_watt,
        "goodput_per_dollar": goodput_per_dollar,
        "evictions": cache.evictions,
        "pinned_evictions": cache.pinned_evictions,
        "transfer_bytes": sum(int(row["transfer_bytes"]) for row in rows),
        "git_commit": _git_commit(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    summary.update(cost)
    return summary
