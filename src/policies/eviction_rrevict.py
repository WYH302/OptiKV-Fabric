"""RR-Evict (Sept 2026): Fine-grained round-robin tail-chunk eviction for agentic LLM serving.

Mitigates LRU recency synchronization by cycling round-robin across the idle
agent trajectories (oldest-accessed sessions first) and evicting a small tail
chunk from each trajectory before advancing, preserving reusable prefix roots
across returning agents while still draining completed/long-idle sessions.
Strictly causal (uses only session_id, block_pos, and last_access_s).
"""

from __future__ import annotations


class RREvictEvictor:
    """Idle-aware round-robin per-session tail-chunk prefix evictor."""

    def __init__(self, chunk_blocks: int = 4) -> None:
        self.chunk_blocks = max(1, int(chunk_blocks))

    def on_admit(self, entry) -> None:
        return None

    def choose_batch(self, row, excess_bytes: int) -> list:
        """Select unpinned victims in one pass: cycle round-robin over idle sessions
        (ordered oldest-idle first) removing `chunk_blocks` deepest suffix blocks per turn."""
        by_sess: dict[str, list] = {}
        sess_last: dict[str, float] = {}
        for e in row.values():
            if e.pin_count > 0:
                continue
            sid = e.session_id or "__anon__"
            by_sess.setdefault(sid, []).append(e)
            if e.last_access_s > sess_last.get(sid, -1.0):
                sess_last[sid] = e.last_access_s
        if not by_sess:
            return []

        # Order sessions from oldest-idle to most-recently-active;
        # restrict RR pool to the older half of idle sessions (min 2) so active
        # sessions keep full context while idle trajectories shed tails gracefully.
        ordered_sids = sorted(by_sess.keys(), key=lambda s: (sess_last[s], s))
        for lst in by_sess.values():
            lst.sort(key=lambda e: (e.block_pos, -e.last_access_s))

        victims: list = []
        freed = 0
        # Phase 1: Round-robin tail chunk trimming across the older idle pool
        idle_pool_size = max(2, len(ordered_sids) // 2)
        idle_sids = list(ordered_sids[:idle_pool_size])
        remaining_sids = list(ordered_sids[idle_pool_size:])

        while freed < excess_bytes and (idle_sids or remaining_sids):
            if not idle_sids:
                idle_sids = remaining_sids
                remaining_sids = []
            next_round: list[str] = []
            for sid in idle_sids:
                if freed >= excess_bytes:
                    break
                lst = by_sess.get(sid)
                if not lst:
                    continue
                # Evict up to chunk_blocks deepest tail blocks from this idle trajectory
                for _ in range(self.chunk_blocks):
                    if not lst or freed >= excess_bytes:
                        break
                    v = lst.pop()
                    victims.append(v)
                    freed += v.nbytes
                if lst:
                    next_round.append(sid)
                else:
                    del by_sess[sid]
            idle_sids = next_round
        return victims

    def choose(self, entries, request_index: int):
        by_sess: dict[str, list] = {}
        sess_last: dict[str, float] = {}
        for e in entries:
            if e.pin_count > 0:
                continue
            sid = getattr(e, "session_id", "") or "__anon__"
            by_sess.setdefault(sid, []).append(e)
            if e.last_access_s > sess_last.get(sid, -1.0):
                sess_last[sid] = e.last_access_s
        if not by_sess:
            return None
        oldest_sid = min(by_sess.keys(), key=lambda s: (sess_last[s], s))
        return max(by_sess[oldest_sid], key=lambda e: (getattr(e, "block_pos", 0), -e.last_access_s, e.block_id))
