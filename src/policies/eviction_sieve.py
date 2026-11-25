"""SIEVE eviction (NSDI '24; adopted in Mooncake KV store).

Simpler-than-LRU scan-resistant eviction using a FIFO queue with a moving hand
pointer and single visited bit per entry. Strictly causal (never peeks future).
Fast O(1)-amortized implementation exploiting CacheStore's FIFO OrderedDict.
"""

from __future__ import annotations


class SIEVEEvictor:
    """Moving-hand FIFO eviction with single-bit visited promotion."""

    def __init__(self) -> None:
        self._hand_seq: int = 0

    def on_admit(self, entry) -> None:
        entry.freq = 0

    def choose_from_row(self, row) -> object | None:
        """Fast path directly scanning OrderedDict (already ordered by insert_seq)."""
        if not row:
            return None
        # Pass 1: from _hand_seq to tail
        first_unpinned = None
        for entry in row.values():
            if entry.pin_count > 0:
                continue
            if first_unpinned is None:
                first_unpinned = entry
            if entry.insert_seq < self._hand_seq:
                continue
            if entry.freq > 0:
                entry.freq = 0
                continue
            self._hand_seq = entry.insert_seq + 1
            return entry
        # Pass 2: wrap around from head
        self._hand_seq = 0
        for entry in row.values():
            if entry.pin_count > 0:
                continue
            if entry.freq > 0:
                entry.freq = 0
                continue
            self._hand_seq = entry.insert_seq + 1
            return entry
        return first_unpinned

    def choose(self, entries, request_index: int):
        unpinned = [e for e in entries if e.pin_count == 0]
        if not unpinned:
            return None
        for entry in unpinned:
            if entry.insert_seq < self._hand_seq:
                continue
            if entry.freq > 0:
                entry.freq = 0
                continue
            self._hand_seq = entry.insert_seq + 1
            return entry
        self._hand_seq = 0
        for entry in unpinned:
            if entry.freq > 0:
                entry.freq = 0
                continue
            self._hand_seq = entry.insert_seq + 1
            return entry
        return unpinned[0]
