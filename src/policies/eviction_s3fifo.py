"""S3-FIFO eviction. Uses hit bits and queue position, never future requests."""

from __future__ import annotations


class S3FIFOEvictor:
    def __init__(self) -> None:
        self.ghost: set[str] = set()
        self.ghost_order: list[str] = []
        self._seq = 0

    def _bump(self, entries) -> int:
        hi = max((e.insert_seq for e in entries), default=0)
        self._seq = max(self._seq, hi) + 1
        return self._seq

    def on_admit(self, entry) -> None:
        if entry.block_id in self.ghost:
            entry.queue_name = "M"
            self.ghost.discard(entry.block_id)
        else:
            entry.queue_name = "S"
        entry.freq = 0

    def _remember(self, block_id: str) -> None:
        if block_id in self.ghost:
            return
        self.ghost.add(block_id)
        self.ghost_order.append(block_id)
        if len(self.ghost_order) > 16384:
            oldest = self.ghost_order.pop(0)
            self.ghost.discard(oldest)

    def choose(self, entries, request_index: int):
        unpinned = [entry for entry in entries if entry.pin_count == 0]
        if not unpinned:
            return None
        small = [entry for entry in unpinned if entry.queue_name == "S"]
        main = [entry for entry in unpinned if entry.queue_name != "S"]
        small_count = sum(1 for entry in entries if entry.queue_name == "S")
        quota = max(1, len(entries) // 10)
        if small_count > quota and small:
            victim = min(small, key=lambda entry: (entry.insert_seq, entry.block_id))
            if victim.freq > 0:
                victim.freq = 0
                victim.queue_name = "M"
                victim.insert_seq = self._bump(entries)
                return "promoted"
            self._remember(victim.block_id)
            return victim
        pool = main or unpinned
        victim = min(pool, key=lambda entry: (entry.insert_seq, entry.block_id))
        if victim.queue_name != "S" and victim.freq > 0:
            victim.freq -= 1
            victim.insert_seq = self._bump(entries)
            return "promoted"
        return victim
