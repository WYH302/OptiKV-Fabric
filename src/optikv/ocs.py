"""Optical Circuit Switch (OCS) Port Calendar with Strict Bipartite Matching Conservation,
and Contended Electrical RDMA NIC Calendar.

Physical OCS Conservation Laws:
- An optical circuit switch connecting W GPU worker transceivers to P Optical KV Pool
  ports (plus optional peer-to-peer GPU links) maintains a valid bipartite matching at
  every instant: each pool port p in {0..P-1} is physically latched to AT MOST ONE
  worker `connected_worker`, and each worker transceiver w in {0..W-1} is physically
  latched to AT MOST ONE optical port `matched_port`.
- Consequently, at any instant, at most min(W, P) workers can hold active standing
  circuits to the Optical KV Pool simultaneously.
- A standing circuit hit (tau = 0) occurs ONLY when the requested physical edge
  (src_worker, target) is already intact in the active matching.
- Switching port p from old_worker to a different src_worker (or to a different target)
  physically rotates the MEMS mirror, tearing down the old worker's latch and incurring
  reconfiguration dead time tau.
- Supports WDM multi-wavelength striping (W_wdm >= 1) and both non-oversubscribed
  (P >= W) and oversubscribed (P < W) port provisioning.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Port:
    free_at: float = 0.0
    target: int | None = None
    connected_worker: int | None = None
    matched_port: int | None = None


@dataclass(frozen=True)
class TransferPlan:
    port_index: int
    target: int
    start_s: float
    end_s: float
    wait_s: float
    reconfig_s: float
    transfer_s: float

    @property
    def critical_s(self) -> float:
        return self.end_s - self.start_s + self.wait_s


class OpticalCalendar:
    def __init__(
        self,
        num_ports: int,
        reconfig_s: float,
        bandwidth_B_s: float,
        latency_s: float,
        *,
        num_workers: int = 8,
        bipartite_port_contention: bool = True,
        wdm_wavelengths: int = 1,
    ) -> None:
        if num_ports not in (1, 2, 4, 8, 16, 32):
            raise ValueError(f"concurrent links must be in {{1, 2, 4, 8, 16, 32}}, got {num_ports}")
        if wdm_wavelengths < 1:
            raise ValueError(f"wdm_wavelengths must be >= 1, got {wdm_wavelengths}")
        self.reconfig_s = float(reconfig_s)
        self.bandwidth_B_s = float(bandwidth_B_s)
        self.latency_s = float(latency_s)
        self.ports = [Port() for _ in range(num_ports)]
        self.num_workers = int(max(1, num_workers))
        # Always enforce physical matching conservation when src_worker is provided;
        # bipartite_port_contention controls whether worker-side transceiver time serialization
        # is strictly enforced in addition to physical edge matching.
        self.bipartite_port_contention = bool(bipartite_port_contention)
        self.wdm_wavelengths = int(wdm_wavelengths)
        self.worker_ports = [Port() for _ in range(self.num_workers)]
        self.total_active_s: float = 0.0
        self.total_reconfig_s: float = 0.0
        self.total_transfers: int = 0
        self.standing_hits: int = 0

    def transfer_time(self, nbytes: int, wavelengths: int = 1) -> float:
        if nbytes <= 0 or self.bandwidth_B_s == float("inf"):
            return 0.0
        eff_bw = self.bandwidth_B_s * max(1, wavelengths)
        return self.latency_s + nbytes / eff_bw

    @property
    def duty_cycle(self) -> float:
        denom = self.total_active_s + self.total_reconfig_s
        if denom <= 0.0:
            return 0.0
        return self.total_active_s / denom

    def latched_pool_workers(self) -> set[int]:
        """Return the set of workers currently physically matched to the Optical Pool (-1).
        By bipartite matching conservation, len(latched_pool_workers()) <= min(W, P)."""
        return {
            p.connected_worker
            for p in self.ports
            if p.target == -1 and p.connected_worker is not None
        }

    def is_pool_latched(self, worker: int) -> bool:
        if not (0 <= worker < self.num_workers):
            return False
        mp = self.worker_ports[worker].matched_port
        if mp is None or not (0 <= mp < len(self.ports)):
            return False
        port = self.ports[mp]
        return port.connected_worker == worker and port.target == -1

    def min_wait_s(self, now: float, src_worker: int | None = None) -> float:
        pool_wait = min(max(0.0, port.free_at - now) for port in self.ports)
        if self.bipartite_port_contention and src_worker is not None and 0 <= src_worker < self.num_workers:
            w_wait = max(0.0, self.worker_ports[src_worker].free_at - now)
            return max(pool_wait, w_wait)
        return pool_wait

    def active_targets(self, src_worker: int | None = None) -> set[int]:
        """Return targets reachable from `src_worker` via an intact standing circuit (tau = 0)."""
        if src_worker is not None and 0 <= src_worker < self.num_workers:
            mp = self.worker_ports[src_worker].matched_port
            if mp is not None and 0 <= mp < len(self.ports):
                p = self.ports[mp]
                if p.connected_worker == src_worker and p.target is not None:
                    return {p.target}
            return set()
        return {port.target for port in self.ports if port.target is not None}

    def snapshot(self):
        return (
            [(p.free_at, p.target, p.connected_worker, p.matched_port) for p in self.ports],
            [(p.free_at, p.target, p.connected_worker, p.matched_port) for p in self.worker_ports],
            self.total_active_s,
            self.total_reconfig_s,
            self.total_transfers,
            self.standing_hits,
        )

    def restore(self, snap) -> None:
        port_state, worker_state, act, rec, xfers, hits = snap
        for port, (free_at, target, cw, mp) in zip(self.ports, port_state):
            port.free_at = free_at
            port.target = target
            port.connected_worker = cw
            port.matched_port = mp
        for port, (free_at, target, cw, mp) in zip(self.worker_ports, worker_state):
            port.free_at = free_at
            port.target = target
            port.connected_worker = cw
            port.matched_port = mp
        self.total_active_s = act
        self.total_reconfig_s = rec
        self.total_transfers = xfers
        self.standing_hits = hits

    def teardown_all(self) -> None:
        """Tear down all optical circuit latches (stateless Optical-Naive behavior)."""
        for port in self.ports:
            port.target = None
            port.connected_worker = None
            port.matched_port = None
        for wp in self.worker_ports:
            wp.target = None
            wp.connected_worker = None
            wp.matched_port = None

    def _one(self, now: float, target: int, nbytes: int, src_worker: int | None = None) -> TransferPlan:
        xfer = self.transfer_time(nbytes, wavelengths=self.wdm_wavelengths)
        valid_src = src_worker is not None and 0 <= src_worker < self.num_workers
        w_port = self.worker_ports[src_worker] if valid_src else None
        dst_w_port = (
            self.worker_ports[target]
            if (self.bipartite_port_contention and 0 <= target < self.num_workers)
            else None
        )

        best: tuple[float, int, float, int, float, float, bool] | None = None
        for index, port in enumerate(self.ports):
            start = max(now, port.free_at)
            if self.bipartite_port_contention and w_port is not None:
                start = max(start, w_port.free_at)
            if dst_w_port is not None:
                start = max(start, dst_w_port.free_at)

            # Strict bipartite matching check:
            # A standing circuit hit (tau = 0) requires the EXACT physical edge (src_worker, target)
            # to be currently latched on this optical port!
            if valid_src:
                is_standing = (
                    port.target == target
                    and port.connected_worker == src_worker
                    and w_port is not None
                    and w_port.matched_port == index
                    and w_port.target == target
                )
            else:
                is_standing = (port.target == target)

            tau = 0.0 if is_standing else self.reconfig_s
            end = start + tau + xfer
            # Tie-break preference when reconfiguring:
            # (0) standing circuit, (1) unlatched/empty port, (2) least-recently-used port (lowest free_at).
            steal_rank = 0 if is_standing else (1 if port.connected_worker is None else 2)
            cand = (end, steal_rank, port.free_at, index, start, tau, is_standing)
            if best is None or cand < best:
                best = cand

        assert best is not None
        end, _, _, index, start, tau, is_standing = best
        port = self.ports[index]

        if not is_standing:
            # 1. Tear down any previous worker latched to this optical port (strict port conservation!)
            old_w = port.connected_worker
            if old_w is not None and 0 <= old_w < self.num_workers:
                old_wp = self.worker_ports[old_w]
                if old_wp.matched_port == index:
                    old_wp.matched_port = None
                    old_wp.target = None
            old_dst = port.target
            if old_dst is not None and 0 <= old_dst < self.num_workers:
                old_dp = self.worker_ports[old_dst]
                if old_dp.matched_port == index:
                    old_dp.matched_port = None
                    old_dp.target = None

            # 2. If src_worker was previously latched to a DIFFERENT optical port, release that old port!
            if valid_src and w_port is not None:
                old_p_idx = w_port.matched_port
                if old_p_idx is not None and old_p_idx != index and 0 <= old_p_idx < len(self.ports):
                    old_p = self.ports[old_p_idx]
                    if old_p.connected_worker == src_worker:
                        old_p.connected_worker = None
                        old_p.target = None

            # 3. If destination peer worker was latched to a different port, release it too
            if dst_w_port is not None:
                old_dp_idx = dst_w_port.matched_port
                if old_dp_idx is not None and old_dp_idx != index and 0 <= old_dp_idx < len(self.ports):
                    old_dp_port = self.ports[old_dp_idx]
                    if old_dp_port.target == target or old_dp_port.connected_worker == target:
                        old_dp_port.connected_worker = None
                        old_dp_port.target = None

        # Commit new matching state
        port.free_at = end
        port.target = target
        port.connected_worker = src_worker if valid_src else None
        if valid_src and w_port is not None:
            w_port.free_at = end
            w_port.target = target
            w_port.matched_port = index
        if dst_w_port is not None:
            dst_w_port.free_at = end
            dst_w_port.target = src_worker if valid_src else target
            dst_w_port.matched_port = index

        self.total_active_s += xfer
        self.total_reconfig_s += tau
        self.total_transfers += 1
        if is_standing:
            self.standing_hits += 1
        return TransferPlan(
            port_index=index,
            target=target,
            start_s=start,
            end_s=end,
            wait_s=start - now,
            reconfig_s=tau,
            transfer_s=xfer,
        )

    def transfer_sequence(
        self,
        now: float,
        segments: list[tuple[int, int]],
        commit: bool,
        *,
        src_worker: int | None = None,
        parallel: bool = True,
    ) -> list[TransferPlan]:
        """Move each (target, nbytes) segment under strict bipartite matching conservation."""
        snap = self.snapshot()
        plans: list[TransferPlan] = []
        cursor = now
        use_parallel = parallel and len(self.ports) > 1 and (not self.bipartite_port_contention)
        for target, nbytes in segments:
            if nbytes <= 0:
                continue
            dispatch_t = now if use_parallel else cursor
            plan = self._one(dispatch_t, target, nbytes, src_worker=src_worker)
            plans.append(plan)
            cursor = max(cursor, plan.end_s)
        if not commit:
            self.restore(snap)
        return plans


class ElectricalNICCalendar:
    """Contended per-worker electrical RDMA NIC queue with tau = 0."""

    def __init__(self, num_workers: int, bandwidth_B_s: float, latency_s: float) -> None:
        self.num_workers = int(num_workers)
        self.bandwidth_B_s = float(bandwidth_B_s)
        self.latency_s = float(latency_s)
        self.free_at: list[float] = [0.0] * self.num_workers

    def transfer_time(self, nbytes: int) -> float:
        if nbytes <= 0 or self.bandwidth_B_s == float("inf"):
            return 0.0
        return self.latency_s + nbytes / self.bandwidth_B_s

    def min_wait_s(self, now: float, worker: int) -> float:
        return max(0.0, self.free_at[worker] - now)

    def transfer(self, now: float, worker: int, nbytes: int, commit: bool) -> float:
        """Return total elapsed time from `now` until transfer completes on worker's NIC."""
        if nbytes <= 0:
            return 0.0
        xfer = self.transfer_time(nbytes)
        start = max(now, self.free_at[worker])
        end = start + xfer
        if commit:
            self.free_at[worker] = end
        return end - now
