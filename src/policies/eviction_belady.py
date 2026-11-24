"""Belady eviction. This is the only policy component that may see the future."""

from __future__ import annotations

import bisect
import math


class BeladyEvictor:
    def __init__(self, requests) -> None:
        self.uses: dict[str, list[int]] = {}
        for index, request in enumerate(requests):
            for block in request.blocks:
                self.uses.setdefault(block.block_id, []).append(index)

    def distance(self, block_id: str, request_index: int) -> float:
        uses = self.uses.get(block_id, [])
        position = bisect.bisect_right(uses, request_index)
        if position >= len(uses):
            return math.inf
        return float(uses[position])

    def choose(self, entries, request_index: int):
        eligible = [entry for entry in entries if entry.pin_count == 0]
        if not eligible:
            return None
        return max(
            eligible,
            key=lambda entry: (self.distance(entry.block_id, request_index), entry.nbytes, entry.block_id),
        )
