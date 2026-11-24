"""Optical circuit calendar (OCS) and contended electrical NIC calendar.

OCS physics:
- A standing circuit to the same target pays tau = 0.
- Switching a port to a different target pays the configured reconfiguration delay tau.
- Concurrent optical links per fabric pool must be in {1, 2, 4, 8}.
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
    ) -> None:
        if num_ports not in (1, 2, 4, 8):
            raise ValueError(f"concurrent links must be in {{1, 2, 4, 8}}, got {num_ports}")
        self.reconfig_s = float(reconfig_s)
        self.bandwidth_B_s = float(bandwidth_B_s)
        self.latency_s = float(latency_s)
        self.ports = [Port() for _ in range(num_ports)]
        self.total_active_s: float = 0.0
        self.total_reconfig_s: float = 0.0
        self.total_transfers: int = 0
        self.standing_hits: int = 0

    def transfer_time(self, nbytes: int) -> float:
        if nbytes <= 0 or self.bandwidth_B_s == float("inf"):
            return 0.0
        return self.latency_s + nbytes / self.bandwidth_B_s

    @property
    def duty_cycle(self) -> float:
        denom = self.total_active_s + self.total_reconfig_s
        if denom <= 0.0:
            return 0.0
        return self.total_active_s / denom

    def min_wait_s(self, now: float) -> float:
        return min(max(0.0, port.free_at - now) for port in self.ports)

    def active_targets(self) -> set[int]:
        return {port.target for port in self.ports if port.target is not None}

    def snapshot(self):
        return (
            [(port.free_at, port.target) for port in self.ports],
            self.total_active_s,
            self.total_reconfig_s,
            self.total_transfers,
            self.standing_hits,
        )

    def restore(self, snap) -> None:
        port_state, act, rec, xfers, hits = snap
        for port, (free_at, target) in zip(self.ports, port_state):
            port.free_at = free_at
            port.target = target
        self.total_active_s = act
        self.total_reconfig_s = rec
        self.total_transfers = xfers
        self.standing_hits = hits

    def _one(self, now: float, target: int, nbytes: int) -> TransferPlan:
        xfer = self.transfer_time(nbytes)
        best: tuple[float, int, float, float, bool] | None = None
        for index, port in enumerate(self.ports):
            start = max(now, port.free_at)
            is_standing = port.target == target
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
    ) -> list[TransferPlan]:
        """Move each (target, nbytes) segment in order. Restore state if commit is false."""
        snap = self.snapshot()
        plans: list[TransferPlan] = []
        cursor = now
        for target, nbytes in segments:
            if nbytes <= 0:
                continue
            plan = self._one(cursor, target, nbytes)
            plans.append(plan)
            cursor = plan.end_s
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
