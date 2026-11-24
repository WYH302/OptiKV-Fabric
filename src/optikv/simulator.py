"""Request-level multi-tier simulator for OptiKV-Fabric.

Models GPU HBM, local host DRAM, local NVMe SSD, contended electrical RDMA,
and an Optical Circuit Switch (OCS) port calendar with standing circuits,
reconfiguration dead time, windowed inter-turn burst prefetching, value-based
multi-tier spill placement, reuse-aware eviction with request-unit block pinning,
and 100% mathematically lossless selective recompute abort.
"""

from __future__ import annotations

import datetime
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
from optikv.trace import BlockRef, Request, Trace, stable_hash, trace_from_config
from policies import BeladyEvictor, PolicySpec, S3FIFOEvictor, build_policy, expand_policy_names
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
    if spec.eviction == "reuse":
        return ReuseEvictor()
    return LRUEvictor()


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

    for index, request in enumerate(requests):
        if request.turn_index == 0:
            for block in request.blocks[: spec.anchor_prefix_blocks]:
                cache.anchors.add(block.block_id)

        # Windowed asynchronous inter-turn burst prefetching (strictly causal: uses session_prev only)
        prefetched = False
        prefetched_ids: set[str] = set()
        prefetch_target_worker: int | None = None
        if spec.prefetch and request.pre_gap_s > 0.0 and request.session_id in session_prev:
            prefetch_target_worker = _choose_prefetch_worker(
                spec, request, index, cache, free, session_last_worker.get(request.session_id)
            )
            prefetched, prefetched_ids = _prefetch(
                spec=spec,
                request=request,
                worker=prefetch_target_worker,
                index=index,
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

        worker = _choose_worker(
            spec, request, index, cache, calendar, nic, free, prefill, cluster, kv_bytes
        )

        # Track prefetch precision vs spurious prefetching
        if prefetched and prefetched_ids:
            req_id_set = {b.block_id for b in request.blocks}
            for block in session_prev.get(request.session_id, ()):
                if block.block_id in prefetched_ids:
                    if worker == prefetch_target_worker and block.block_id in req_id_set:
                        total_prefetch_hit_tokens += block.num_tokens
                    else:
                        total_spurious_prefetch_bytes += block.num_tokens * kv_bytes

        row = _serve(
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
        )
        rows.append(row)

        if spec.store and not spec.oracle:
            _retain(
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
            )
        elif spec.oracle:
            for block in request.blocks:
                seen.add(block.block_id)

        if spec.store and not spec.oracle:
            for block_id in {block.block_id for block in request.blocks}:
                cache.obs[block_id] = cache.obs.get(block_id, 0) + 1
                cache.last_obs_s[block_id] = request.arrival_s
            session_prev[request.session_id] = request.blocks
            session_last_worker[request.session_id] = worker
            if record_hbm_snapshots:
                hbm_after[(spec.name, index)] = cache.ids("hbm", worker)

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
    return summary, rows, hbm_after


def _prefer_optical_over_dram(
    spec: PolicySpec,
    cluster: Cluster,
    missing_tokens: int,
    kv_bytes: int,
    calendar: OpticalCalendar,
) -> bool:
    """Analytical fabric crossover: prefer OCS over local PCIe DRAM when burst exceeds B_cross."""
    if spec.optical == "none" or cluster.ocs_bandwidth_gb_s <= cluster.dram_bandwidth_gb_s:
        return False
    missing_bytes = missing_tokens * kv_bytes
    if missing_bytes <= 0:
        return False
    # If a standing circuit is already open to the pool, effective switch cost is 0
    tau_eff = 0.0 if OPTICAL_POOL_TARGET in calendar.active_targets() else cluster.reconfig_s
    inv_diff = (1.0 / (cluster.dram_bandwidth_gb_s * 1e9)) - (1.0 / (cluster.ocs_bandwidth_gb_s * 1e9))
    b_cross = tau_eff / inv_diff if inv_diff > 0 else float("inf")
    return missing_bytes > b_cross


def _choose_prefetch_worker(
    spec: PolicySpec,
    request: Request,
    index: int,
    cache: CacheStore,
    free: list[float],
    prev_worker: int | None,
) -> int:
    """Pick the worker with lowest predicted queue and available HBM headroom for prefetch."""
    if request.assigned_worker is not None and not spec.routing:
        return int(request.assigned_worker) % cache.n_workers
    if not spec.routing:
        return min(
            range(cache.n_workers),
            key=lambda w: (max(0.0, free[w] - request.arrival_s), (w - index) % cache.n_workers),
        )
    best_w = 0
    best_key = None
    for w in range(cache.n_workers):
        cap = cache.capacity[("hbm", w)]
        util = cache.used[("hbm", w)] / cap if cap > 0 else 1.0
        over_cap = 1 if util >= spec.prefetch_utilization_cap else 0
        q = max(0.0, free[w] - request.arrival_s)
        not_prev = 0 if (prev_worker is not None and w == prev_worker) else 1
        key = (over_cap, round(q, 4), not_prev, util, w)
        if best_key is None or key < best_key:
            best_key = key
            best_w = w
    return best_w


def _choose_worker(
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
) -> int:
    workers = cache.n_workers
    if spec.oracle:
        return min(range(workers), key=lambda w: (free[w], w))
    if not spec.routing:
        if request.assigned_worker is not None:
            return int(request.assigned_worker) % workers
        # Realistic stateless least-loaded / round-robin cluster load balancer
        return min(
            range(workers),
            key=lambda w: (max(0.0, free[w] - request.arrival_s), (w - index) % workers),
        )

    # Candidate worker pruning for very long 16-token block requests (exact fast path)
    if len(request.blocks) > 512 and workers > 2:
        b0 = request.blocks[0].block_id
        cands = {
            w
            for w in range(workers)
            if (b0 in cache.rows[("hbm", w)]) or (spec.use_dram and b0 in cache.rows[("dram", w)])
        }
        by_q = sorted(range(workers), key=lambda w: (max(0.0, free[w] - request.arrival_s), w))
        cands.add(by_q[0])
        cands.add(by_q[1])
        candidate_workers = sorted(cands)
    else:
        candidate_workers = list(range(workers))

    alpha = cluster.alpha_overlap
    active_peers = calendar.active_targets() if spec.optical != "none" else None
    best_w = candidate_workers[0]
    best_key: tuple[float, float, int] | None = None

    for w in candidate_workers:
        start = max(request.arrival_s, free[w])
        queue_s = start - request.arrival_s
        local_tokens, offset = _leading_local(request.blocks, cache, w)
        prefer_opt = _prefer_optical_over_dram(
            spec, cluster, request.input_tokens - local_tokens, kv_bytes, calendar
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
        book_at = request.arrival_s if spec.book_at_arrival else start
        plans = _transfer_plans(calendar, book_at, segments, spec, commit=False)
        port_cost = (plans[-1].end_s - book_at) if plans else 0.0
        residual = max(0.0, plans[-1].end_s - start) if plans else 0.0
        move_cost = port_cost + lower_s
        delta = prefill.cold_parts(local_tokens + ext_tokens)[0] - prefill.cold_parts(local_tokens)[0]
        if spec.recompute_fallback and ext_tokens > 0 and move_cost > delta + 1e-15:
            transfer_s = 0.0
            accepted = local_tokens
        else:
            transfer_s = residual + lower_s
            accepted = local_tokens + ext_tokens
        prefill_s, decode_s = prefill.request_times(accepted, request.input_tokens - accepted)
        est_ttft = layer_ttft(queue_s, transfer_s, prefill_s, decode_s, alpha)
        cand = (est_ttft, queue_s, w)
        if best_key is None or cand < best_key:
            best_key = cand
            best_w = w
    return best_w


def _leading_local(blocks: tuple[BlockRef, ...], cache: CacheStore, worker: int) -> tuple[int, int]:
    hbm = cache.rows[("hbm", worker)]
    tokens = 0
    for index, block in enumerate(blocks):
        if block.block_id not in hbm:
            return tokens, index
        tokens += block.num_tokens
    return tokens, len(blocks)


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
    request: Request,
    spec: PolicySpec,
    noise: float = 0.0,
) -> float:
    return p_reuse(
        cache.obs.get(block_id, 0),
        block_id in cache.anchors,
        request.pre_gap_s,
        spec.recency_tau_s,
        noise=noise,
    )


def _prefetch(
    *,
    spec: PolicySpec,
    request: Request,
    worker: int,
    index: int,
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
    if not spec.prefetch or request.pre_gap_s <= 0.0:
        return False, set()
    previous = session_prev.get(request.session_id)
    if not previous:
        return False, set()
    capacity = cache.capacity[("hbm", worker)]
    if capacity <= 0:
        return False, set()
    pinned_bytes = sum(e.nbytes for e in cache.rows[("hbm", worker)].values() if e.pin_count > 0)
    if pinned_bytes / capacity >= spec.prefetch_utilization_cap:
        return False, set()
    budget = int(spec.prefetch_utilization_cap * capacity - pinned_bytes)

    window = min(request.pre_gap_s, max(0.25, request.pre_gap_s * 0.90))
    book_at = max(0.0, request.arrival_s - window)
    is_optical = spec.optical != "none"

    # Admission check 1: do not trigger optical prefetch if OCS ports are congested
    if is_optical and calendar.min_wait_s(book_at) > request.pre_gap_s * 0.5:
        return False, set()

    hbm_local = cache.rows[("hbm", worker)]
    dram_local = cache.rows[("dram", worker)]
    pool_rows = cache.rows[("optical_pool", None)]
    rdma_rows = cache.rows[("rdma", None)]
    active_peers = calendar.active_targets() if is_optical else set()

    # Precompute Prefix-GDSF physical s/byte unit scores matching _retain()
    prev_total_tokens = sum(b.num_tokens for b in previous)
    prev_total_cold = prefill.cold_parts(prev_total_tokens)[0] if prev_total_tokens > 0 else 0.0

    # Collect missing blocks from previous turn (strictly causal)
    dram_blocks: list[tuple[BlockRef, float, float]] = []
    remote_groups: dict[int, list[tuple[BlockRef, float, float]]] = {}
    rdma_blocks: list[tuple[BlockRef, float, float]] = []
    cum_tok = 0
    for block in previous:
        prev_cold = prefill.cold_parts(cum_tok)[0] if cum_tok > 0 else 0.0
        protected_saved = max(1e-9, prev_total_cold - prev_cold)
        cum_tok += block.num_tokens
        if block.block_id in hbm_local:
            continue
        noise = rng.gauss(0.0, reuse_noise_std) if reuse_noise_std > 0.0 else 0.0
        sc = _score(cache, block.block_id, request, spec, noise=noise)
        if sc < 0.35:
            continue
        ev_score = (sc * protected_saved) / float(max(1, block.num_tokens * kv_bytes))
        if spec.use_dram and block.block_id in dram_local:
            dram_blocks.append((block, sc, ev_score))
            continue
        if is_optical:
            # Standing-circuit & unified Optical Pool coalescing (consistent with best_location)
            target: int | None = None
            owners = cache._hbm_owners.get(block.block_id)
            if active_peers:
                if owners:
                    for peer in active_peers:
                        if peer != worker and peer in owners:
                            target = peer
                            break
                if target is None and OPTICAL_POOL_TARGET in active_peers and spec.use_optical_pool and block.block_id in pool_rows:
                    target = OPTICAL_POOL_TARGET
            if target is None and spec.use_optical_pool and block.block_id in pool_rows:
                target = OPTICAL_POOL_TARGET
            if target is None and owners:
                for other in sorted(owners):
                    if other != worker:
                        target = other
                        break
            if target is not None:
                remote_groups.setdefault(target, []).append((block, sc, ev_score))
        elif spec.use_rdma:
            owners = cache._hbm_owners.get(block.block_id)
            has_peer = bool(owners and any(other != worker for other in owners))
            if has_peer or block.block_id in rdma_rows:
                rdma_blocks.append((block, sc, ev_score))

    prefetched_ids: set[str] = set()
    applied = False
    remaining_budget = budget

    # Path A: Prefetch from local DRAM over PCIe during pre_gap
    if dram_blocks and remaining_budget > 0:
        chosen_dram: list[tuple[BlockRef, float, float]] = []
        dram_bytes = 0
        for block, sc, ev_sc in dram_blocks:
            nb = block.num_tokens * kv_bytes
            if dram_bytes + nb > remaining_budget:
                break
            chosen_dram.append((block, sc, ev_sc))
            dram_bytes += nb
        dram_s = _tier_seconds(dram_bytes, cluster.dram_bandwidth_gb_s, 0.0)
        if chosen_dram and dram_s <= request.pre_gap_s:
            for block, sc, ev_sc in chosen_dram:
                cache.upsert(
                    "hbm",
                    worker,
                    block.block_id,
                    block.num_tokens,
                    request.arrival_s,
                    sc,
                    evictor,
                    index,
                    pin=False,
                    evict_score=ev_sc,
                    auto_trim=False,
                )
                prefetched_ids.add(block.block_id)
            remaining_budget -= dram_bytes
            applied = True

    # Path B (Optical): Windowed Optical Burst Prefetch from dominant remote source
    if is_optical and remote_groups and remaining_budget > 0:
        target = max(
            remote_groups,
            key=lambda peer: (
                sum(b.num_tokens for b, _, _ in remote_groups[peer])
                * (1.15 if peer in active_peers else 1.0),
                -peer,
            ),
        )
        chosen: list[tuple[BlockRef, float, float]] = []
        total_bytes = 0
        weighted_tokens = 0.0
        for block, sc, ev_sc in remote_groups[target]:
            nb = block.num_tokens * kv_bytes
            if total_bytes + nb > remaining_budget:
                break
            chosen.append((block, sc, ev_sc))
            total_bytes += nb
            weighted_tokens += sc * block.num_tokens
        if total_bytes > 0:
            preview = calendar.transfer_sequence(book_at, [(target, total_bytes)], commit=False)
            if preview and preview[-1].end_s <= request.arrival_s + 1e-9:
                burst_cost = preview[-1].end_s - book_at
                expected_saved = prefill.cold_parts(int(round(weighted_tokens)))[0]
                if expected_saved > burst_cost:
                    calendar.transfer_sequence(book_at, [(target, total_bytes)], commit=True)
                    for block, sc, ev_sc in chosen:
                        cache.upsert(
                            "hbm",
                            worker,
                            block.block_id,
                            block.num_tokens,
                            request.arrival_s,
                            sc,
                            evictor,
                            index,
                            pin=False,
                            evict_score=ev_sc,
                            auto_trim=False,
                        )
                        prefetched_ids.add(block.block_id)
                    applied = True

    # Path B (Electrical RDMA SOTA): Contended-NIC inter-turn prefetch (KVFlow/SYMPHONY paradigm)
    if (not is_optical) and rdma_blocks and remaining_budget > 0:
        chosen_r: list[tuple[BlockRef, float, float]] = []
        rdma_bytes = 0
        for block, sc, ev_sc in rdma_blocks:
            nb = block.num_tokens * kv_bytes
            if rdma_bytes + nb > remaining_budget:
                break
            chosen_r.append((block, sc, ev_sc))
            rdma_bytes += nb
        if chosen_r and rdma_bytes > 0:
            preview_s = nic.transfer(book_at, worker, rdma_bytes, commit=False)
            if book_at + preview_s <= request.arrival_s + 1e-9:
                nic.transfer(book_at, worker, rdma_bytes, commit=True)
                for block, sc, ev_sc in chosen_r:
                    cache.upsert(
                        "hbm",
                        worker,
                        block.block_id,
                        block.num_tokens,
                        request.arrival_s,
                        sc,
                        evictor,
                        index,
                        pin=False,
                        evict_score=ev_sc,
                        auto_trim=False,
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
) -> dict:
    alpha = cluster.alpha_overlap
    if spec.oracle:
        start = request.arrival_s
        queue_s = 0.0
    else:
        start = max(request.arrival_s, free[worker])
        queue_s = start - request.arrival_s

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
        local_tokens, offset = _leading_local(request.blocks, cache, worker)
        active_peers = calendar.active_targets() if spec.optical != "none" else None
        prefer_opt = _prefer_optical_over_dram(
            spec, cluster, request.input_tokens - local_tokens, kv_bytes, calendar
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
        book_at = request.arrival_s if spec.book_at_arrival else start
        plans = _transfer_plans(calendar, book_at, segments, spec, commit=False)
        port_cost = (plans[-1].end_s - book_at) if plans else 0.0
        reconfig_paid = sum(plan.reconfig_s for plan in plans)
        move_cost = port_cost + local_tier_s + rdma_preview_s
        delta = prefill.cold_parts(local_tokens + ext_tokens)[0] - prefill.cold_parts(local_tokens)[0]
        aborted = bool(spec.recompute_fallback and ext_tokens > 0 and move_cost > delta + 1e-15)
        if aborted:
            transfer_s = 0.0
            reconfig_s = 0.0
            transfer_bytes = 0
            accepted = local_tokens
            counts["hbm"] = local_tokens
        else:
            rdma_s = nic.transfer(start, worker, rdma_b, commit=True) if rdma_b > 0 else 0.0
            if segments:
                plans = _transfer_plans(calendar, book_at, segments, spec, commit=True)
            residual = max(0.0, plans[-1].end_s - start) if plans else 0.0
            transfer_s = residual + local_tier_s + rdma_s
            reconfig_s = reconfig_paid
            transfer_bytes = dram_b + ssd_b + rdma_b + sum(nbytes for _, nbytes in segments)
            accepted = local_tokens + ext_tokens
            counts["hbm"] = local_tokens + moved_tokens.get("hbm_tail", 0)
            counts["dram"] = moved_tokens["dram"]
            counts["ssd"] = moved_tokens["ssd"]
            counts["rdma"] = moved_tokens["rdma"]
            counts["remote_hbm"] = moved_tokens["remote_hbm"]
            counts["optical_pool"] = moved_tokens["optical_pool"]
        prefill_s, decode_s = prefill.request_times(accepted, request.input_tokens - accepted)

    if noise_std > 0.0:
        prefill_s *= max(0.05, 1.0 + rng.gauss(0.0, noise_std))
        transfer_s *= max(0.05, 1.0 + rng.gauss(0.0, noise_std))

    ttft_s = layer_ttft(queue_s, transfer_s, prefill_s, decode_s, alpha)
    if not spec.oracle:
        free[worker] = start + (ttft_s - queue_s)
    recomputed = request.input_tokens - accepted
    hit_rate = 0.0 if request.input_tokens == 0 else accepted / request.input_tokens
    return {
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
    }


def _transfer_plans(calendar: OpticalCalendar, book_at: float, segments, spec: PolicySpec, commit: bool):
    """Execute or preview optical burst sequence on OCS Port Calendar.

    - `optical_naive`: tears down circuit after every committed transfer, so every
      subsequent transfer pays tau even to the same peer.
    - `optical_reservation`: keeps standing circuits latched (same peer pays tau = 0,
      new peer pays tau) and books at arrival so tau can overlap worker queueing.
    - `optikv` (`optical == "keep"`): keeps standing circuits latched (same peer pays
      tau = 0, new peer pays tau) and combines with optical-aware routing, windowed
      burst prefetch, and lossless recompute abort.
    """
    if not segments:
        return []
    plans = calendar.transfer_sequence(book_at, segments, commit=commit)
    if commit and spec.optical == "naive":
        for port in calendar.ports:
            port.target = None
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
) -> None:
    """Retain active request blocks in HBM (pinned during service), then spill HBM evictions."""
    pinned = []
    seen_ids: set[str] = set()
    need_scores = bool(spec.placement or spec.eviction == "reuse")
    total_cold = prefill.cold_parts(request.input_tokens)[0] if need_scores else 0.0
    cum_tokens = 0
    use_dram = spec.use_dram
    use_ssd = spec.use_ssd
    dram_rows = cache.rows[("dram", worker)] if use_dram else None
    ssd_rows = cache.rows[("ssd", worker)] if use_ssd else None
    for block in request.blocks:
        if need_scores:
            # Prefix-dependency-aware GDSF: evicting block i invalidates the entire
            # downstream prefix [cum_tokens .. input_tokens], so earlier prefix roots
            # protect strictly more recomputation than suffix leaves.
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
            sc = _score(cache, bid, request, spec, noise=noise)
            nbytes = max(1, block.num_tokens * kv_bytes)
            ev_score = (sc * protected_saved) / float(nbytes)
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
        )
        pinned.append(entry)

    cache.unpin_many(pinned)
    evicted_hbm = cache.trim("hbm", worker, evictor, index)
    if not evicted_hbm:
        return
    _spill_hbm_victims(spec=spec, worker=worker, index=index, cache=cache, evictor=evictor, evicted_hbm=evicted_hbm)


def _spill_hbm_victims(
    *,
    spec: PolicySpec,
    worker: int,
    index: int,
    cache: CacheStore,
    evictor,
    evicted_hbm: list,
) -> None:
    """Multi-tier spill on HBM capacity pressure (shared by _retain and _prefetch)."""
    if spec.placement:
        # Value-based placement (OptiKV): always admit evicted HBM blocks while DRAM/Optical
        # Pool has headroom; once a lower tier exceeds 85% utilization, apply second-observation
        # value filtering so one-shot cold tail blocks cannot evict proven reused/anchor blocks.
        dram_cap = cache.capacity[("dram", worker)]
        pool_cap = cache.capacity[("optical_pool", None)]
        for victim in evicted_hbm:
            is_anchor = victim.block_id in cache.anchors
            sc = victim.reuse_score
            dram_pressured = dram_cap > 0 and (cache.used[("dram", worker)] / dram_cap) >= 0.85
            pool_pressured = pool_cap > 0 and (cache.used[("optical_pool", None)] / pool_cap) >= 0.85
            admit_dram = (not dram_pressured) or is_anchor or (sc >= spec.tier_dram_min_p)
            admit_pool = (not pool_pressured) or is_anchor or (sc >= spec.tier_dram_min_p)
            raw_ev = (sc * 1e-6) / max(1, victim.nbytes)
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
                )
        if spec.use_dram:
            evicted_dram = cache.trim("dram", worker, evictor, index)
            if spec.use_ssd:
                for dv in evicted_dram:
                    if dv.reuse_score >= spec.tier_ssd_min_p or dv.block_id in cache.anchors:
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
                        )
    else:
        # Standard LRU / S3-FIFO / Belady spill for non-placement multi-tier baselines
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

    # Cited fabric port power & cost (num_workers ports for the fabric interconnect)
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
