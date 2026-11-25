"""Optical circuit calendar (OCS) and contended electrical NIC calendar.

OCS physics:
- A standing circuit to the same target pays tau = 0.
- Switching a port to a different target pays the configured reconfiguration delay tau.
- Concurrent optical links per fabric pool must be in {1, 2, 4, 8}.
- Supports parallel multi-port execution across distinct targets, optional strict
  bipartite source-destination transceiver port contention, and WDM multi-wavelength
  striping (W in {1, 2, 4}) across optical links.
- Tracks active transfer time and reconfiguration dead time to compute OCS duty cycle.

Electrical NIC physics:
- Packet-switched RDMA has tau = 0 (no circuit switching) but finite per-worker NIC
  bandwidth with realistic serialization queuing under concurrent transfers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Port:
    free_at: float = 0.0
    target: int | None = None


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
        bipartite_port_contention: bool = False,
        wdm_wavelengths: int = 1,
    ) -> None:
        if num_ports not in (1, 2, 4, 8):
            raise ValueError(f"concurrent links must be in {{1, 2, 4, 8}}, got {num_ports}")
        if wdm_wavelengths < 1:
            raise ValueError(f"wdm_wavelengths must be >= 1, got {wdm_wavelengths}")
        self.reconfig_s = float(reconfig_s)
        self.bandwidth_B_s = float(bandwidth_B_s)
        self.latency_s = float(latency_s)
        self.ports = [Port() for _ in range(num_ports)]
        self.num_workers = int(max(1, num_workers))
        self.bipartite_port_contention = bool(bipartite_port_contention)
        self.wdm_wavelengths = int(wdm_wavelengths)
        # Per-worker optical transceiver state (used when bipartite_port_contention=True)
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

    def min_wait_s(self, now: float, src_worker: int | None = None) -> float:
        pool_wait = min(max(0.0, port.free_at - now) for port in self.ports)
        if self.bipartite_port_contention and src_worker is not None and 0 <= src_worker < self.num_workers:
            w_wait = max(0.0, self.worker_ports[src_worker].free_at - now)
            return max(pool_wait, w_wait)
        return pool_wait

    def active_targets(self, src_worker: int | None = None) -> set[int]:
        pool_targets = {port.target for port in self.ports if port.target is not None}
        if self.bipartite_port_contention and src_worker is not None and 0 <= src_worker < self.num_workers:
            wt = self.worker_ports[src_worker].target
            return {wt} if (wt is not None and wt in pool_targets) else set()
        return pool_targets

    def snapshot(self):
        return (
            [(port.free_at, port.target) for port in self.ports],
            [(port.free_at, port.target) for port in self.worker_ports],
            self.total_active_s,
            self.total_reconfig_s,
            self.total_transfers,
            self.standing_hits,
        )

    def restore(self, snap) -> None:
        port_state, worker_state, act, rec, xfers, hits = snap
        for port, (free_at, target) in zip(self.ports, port_state):
            port.free_at = free_at
            port.target = target
        for port, (free_at, target) in zip(self.worker_ports, worker_state):
            port.free_at = free_at
            port.target = target
        self.total_active_s = act
        self.total_reconfig_s = rec
        self.total_transfers = xfers
        self.standing_hits = hits

    def _one(self, now: float, target: int, nbytes: int, src_worker: int | None = None) -> TransferPlan:
        # WDM multi-wavelength striping scales effective burst bandwidth up to wdm_wavelengths
        # over a single fiber pair (Yoon & Yu 2026 WDM + MEMS model)
        xfer = self.transfer_time(nbytes, wavelengths=self.wdm_wavelengths)
        w_port = (
            self.worker_ports[src_worker]
            if (self.bipartite_port_contention and src_worker is not None and 0 <= src_worker < self.num_workers)
            else None
        )
        # In bipartite mode, if target is another worker's HBM (target >= 0), its transceiver also contends
        dst_w_port = (
            self.worker_ports[target]
            if (self.bipartite_port_contention and 0 <= target < self.num_workers)
            else None
        )
        best: tuple[float, int, float, float, bool] | None = None
        for index, port in enumerate(self.ports):
            start = max(now, port.free_at)
            if w_port is not None:
                start = max(start, w_port.free_at)
            if dst_w_port is not None:
                start = max(start, dst_w_port.free_at)
            pool_standing = port.target == target
            if w_port is not None:
                is_standing = pool_standing and (w_port.target == target)
            else:
                is_standing = pool_standing
            tau = 0.0 if is_standing else self.reconfig_s
            end = start + tau + xfer
            cand = (end, index, start, tau, is_standing)
            if best is None or cand < best:
                best = cand
        assert best is not None
        end, index, start, tau, is_standing = best
        port = self.ports[index]
        port.free_at = end
        port.target = target
        if w_port is not None:
            w_port.free_at = end
            w_port.target = target
        if dst_w_port is not None:
            dst_w_port.free_at = end
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
        """Move each (target, nbytes) segment. Uses parallel multi-port dispatch when multiple
        optical ports are available and bipartite single-port serialization does not bind."""
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
