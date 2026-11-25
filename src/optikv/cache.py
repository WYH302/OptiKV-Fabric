"""Multi-tier block cache. Objects are native trace block IDs.

Tiers per worker:
  - ("hbm", worker): GPU HBM KV budget
  - ("dram", worker): Host CPU DRAM KV offload tier
  - ("ssd", worker): Local NVMe SSD KV tier
Cluster-wide shared pools:
  - ("rdma", None): Disaggregated electrical RDMA KV memory pool
  - ("optical_pool", None): Disaggregated optical-fabric KV memory pool

Eviction NEVER removes an entry with pin_count > 0. Active request blocks stay
pinned until the request has finished using them. `trim()` returns evicted entries
so multi-tier policies can spill victims to lower tiers rather than dropping them.

Performance:
- `LRUEvictor` uses Python's C-level `OrderedDict` (`move_to_end` / iteration from head),
  giving O(1) touch and O(1) eviction with zero heap bloat even on millions of 16-token blocks.
- `ReuseEvictor` uses a lazy-deletion min-heap over `(evict_score, reuse_score, last_access_s, block_id)`.
- `S3FIFOEvictor` and `BeladyEvictor` use the direct evictor path.
"""

from __future__ import annotations

import heapq
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(slots=True)
class Entry:
    block_id: str
    num_tokens: int
    nbytes: int
    tier: str
    worker: int | None
    pin_count: int = 0
    last_access_s: float = 0.0
    reuse_score: float = 0.0
    evict_score: float = 0.0
    freq: int = 0
    queue_name: str = "S"
    insert_seq: int = 0
    session_id: str = ""
    block_pos: int = 0
    ready_at_s: float = 0.0


@dataclass(frozen=True, slots=True)
class Location:
    tier: str
    worker: int | None


class LRUEvictor:
    """Recency eviction. Does not look at future requests."""

    def choose(self, entries: list[Entry], request_index: int) -> Entry | str | None:
        eligible = [entry for entry in entries if entry.pin_count == 0]
        if not eligible:
            return None
        return min(eligible, key=lambda entry: (entry.last_access_s, entry.insert_seq, entry.block_id))


def _is_lru(evictor) -> bool:
    return type(evictor).__name__ == "LRUEvictor"


def _order_key(evictor):
    kind = type(evictor).__name__
    if kind == "ReuseEvictor":
        return lambda entry: (entry.evict_score, entry.reuse_score, entry.last_access_s, entry.block_id)
    return None


class CacheStore:
    def __init__(self, cluster, kv_bytes_per_token: int) -> None:
        self.kv_bytes_per_token = int(kv_bytes_per_token)
        self.n_workers = cluster.num_workers
        self.capacity: dict[tuple[str, int | None], int] = {}
        self.used: dict[tuple[str, int | None], int] = {}
        self.rows: dict[tuple[str, int | None], OrderedDict[str, Entry]] = {}
        self.obs: dict[str, int] = {}
        self.last_obs_s: dict[str, float] = {}
        self.anchors: set[str] = set()
        self.evictions = 0
        self.pinned_evictions = 0
        self._seq = 0
        self._heaps: dict[tuple[str, int | None], list] = {}
        self._gen: dict[tuple[tuple[str, int | None], str], int] = {}
        self._aging: dict[tuple[str, int | None], float] = {}
        self._hbm_owners: dict[str, set[int]] = {}
        for worker in range(cluster.num_workers):
            for tier, cap in (
                ("hbm", cluster.hbm_bytes_per_worker),
                ("dram", cluster.dram_bytes_per_worker),
                ("ssd", cluster.ssd_bytes_per_worker),
            ):
                self._add_tier(tier, worker, cap)
        self._add_tier("rdma", None, cluster.rdma_pool_bytes)
        self._add_tier("optical_pool", None, cluster.rdma_pool_bytes)

    def _add_tier(self, tier: str, worker: int | None, cap: int) -> None:
        key = (tier, worker)
        self.capacity[key] = int(cap)
        self.used[key] = 0
        self.rows[key] = OrderedDict()

    def has(self, block_id: str, tier: str, worker: int | None) -> bool:
        return block_id in self.rows[(tier, worker)]

    def ids(self, tier: str, worker: int | None) -> tuple[str, ...]:
        return tuple(sorted(self.rows[(tier, worker)]))

    def best_location(
        self,
        block_id: str,
        home: int,
        spec,
        preferred_peers: set[int] | None = None,
        prefer_optical_over_dram: bool = False,
    ) -> Location | None:
        if spec.use_hbm and block_id in self.rows[("hbm", home)]:
            return Location("hbm", home)
        if not prefer_optical_over_dram and spec.use_dram and block_id in self.rows[("dram", home)]:
            return Location("dram", home)
        # Priority 1: reuse any LATCHED standing optical circuit (peer or Optical Pool) at tau_eff = 0
        if preferred_peers:
            if spec.use_remote_hbm:
                owners = self._hbm_owners.get(block_id)
                if owners:
                    for peer in preferred_peers:
                        if peer != home and peer in owners:
                            return Location("remote_hbm", peer)
            if -1 in preferred_peers and spec.use_optical_pool and block_id in self.rows[("optical_pool", None)]:
                return Location("optical_pool", None)
        # Priority 2: coalesce unlatched optical fetches through the unified Optical KV Pool
        # before fragmenting across multiple unlatched remote worker HBMs
        if spec.use_optical_pool and block_id in self.rows[("optical_pool", None)]:
            return Location("optical_pool", None)
        if spec.use_remote_hbm:
            owners = self._hbm_owners.get(block_id)
            if owners:
                for worker in sorted(owners):
                    if worker != home:
                        return Location("remote_hbm", worker)
        if prefer_optical_over_dram and spec.use_dram and block_id in self.rows[("dram", home)]:
            return Location("dram", home)
        if spec.use_rdma and block_id in self.rows[("rdma", None)]:
            return Location("rdma", None)
        if spec.use_ssd and block_id in self.rows[("ssd", home)]:
            return Location("ssd", home)
        return None

    def upsert(
        self,
        tier: str,
        worker: int | None,
        block_id: str,
        num_tokens: int,
        now: float,
        score: float,
        evictor,
        request_index: int,
        pin: bool,
        evict_score: float | None = None,
        auto_trim: bool = True,
        session_id: str = "",
        block_pos: int = 0,
        ready_at_s: float = 0.0,
    ) -> Entry:
        key = (tier, worker)
        row = self.rows[key]
        aging = self._aging.get(key, 0.0) if _order_key(evictor) is not None else 0.0
        eff_evict = aging + float(score if evict_score is None else evict_score)
        entry = row.get(block_id)
        if entry is not None:
            old_ev = entry.evict_score
            entry.last_access_s = now
            entry.reuse_score = score
            entry.evict_score = eff_evict
            entry.freq = min(3, entry.freq + 1)
            if session_id and not entry.session_id:
                entry.session_id = session_id
            if ready_at_s > entry.ready_at_s:
                entry.ready_at_s = ready_at_s
            elif ready_at_s == 0.0 and pin:
                # Newly computed/materialized block is immediately ready at `now`
                entry.ready_at_s = min(entry.ready_at_s, now)
            if pin:
                entry.pin_count += 1
            if _is_lru(evictor):
                row.move_to_end(block_id)
            elif abs(eff_evict - old_ev) > 1e-12:
                self._touch(key, entry, evictor)
            return entry
        nbytes = int(num_tokens) * self.kv_bytes_per_token
        entry = Entry(
            block_id=block_id,
            num_tokens=int(num_tokens),
            nbytes=nbytes,
            tier=tier,
            worker=worker,
            pin_count=1 if pin else 0,
            last_access_s=now,
            reuse_score=score,
            evict_score=eff_evict,
            session_id=session_id,
            block_pos=int(block_pos),
            ready_at_s=float(ready_at_s),
        )
        on_admit = getattr(evictor, "on_admit", None)
        if on_admit is not None:
            on_admit(entry)
        self._seq += 1
        entry.insert_seq = self._seq
        row[block_id] = entry
        if tier == "hbm" and worker is not None:
            owners = self._hbm_owners.get(block_id)
            if owners is None:
                self._hbm_owners[block_id] = {worker}
            else:
                owners.add(worker)
        self.used[key] += nbytes
        if not _is_lru(evictor):
            self._touch(key, entry, evictor)
        if auto_trim and not pin and self.used[key] > self.capacity[key]:
            self.trim(tier, worker, evictor, request_index)
        return entry

    def _touch(self, key: tuple[str, int | None], entry: Entry, evictor) -> None:
        keyfn = _order_key(evictor)
        if keyfn is None:
            return
        gkey = (key, entry.block_id)
        gen = self._gen.get(gkey, 0) + 1
        self._gen[gkey] = gen
        heap = self._heaps.setdefault(key, [])
        heapq.heappush(heap, (keyfn(entry), gen, entry.block_id))

    def _invalidate(self, key: tuple[str, int | None], block_id: str) -> None:
        gkey = (key, block_id)
        if gkey in self._gen:
            self._gen[gkey] += 1

    def _compact(self, key: tuple[str, int | None], evictor) -> None:
        keyfn = _order_key(evictor)
        heap = self._heaps.get(key)
        if keyfn is None or not heap:
            return
        rows = self.rows[key]
        if len(heap) < max(2048, 4 * max(1, len(rows))):
            return
        fresh = []
        for entry in rows.values():
            gkey = (key, entry.block_id)
            gen = self._gen.get(gkey, 0) + 1
            self._gen[gkey] = gen
            fresh.append((keyfn(entry), gen, entry.block_id))
        heapq.heapify(fresh)
        self._heaps[key] = fresh

    def _victim_reuse(self, key: tuple[str, int | None], evictor) -> Entry | None:
        keyfn = _order_key(evictor)
        heap = self._heaps.get(key)
        if keyfn is None or not heap:
            return None
        rows = self.rows[key]
        deferred = []
        found = None
        while heap:
            order, gen, block_id = heapq.heappop(heap)
            gkey = (key, block_id)
            if self._gen.get(gkey) != gen:
                continue
            entry = rows.get(block_id)
            if entry is None:
                continue
            if entry.pin_count > 0:
                deferred.append((order, gen, block_id))
                continue
            current = keyfn(entry)
            if current != order:
                gen2 = gen + 1
                self._gen[gkey] = gen2
                heapq.heappush(heap, (current, gen2, block_id))
                continue
            found = entry
            break
        for item in deferred:
            heapq.heappush(heap, item)
        return found

    def _victim_lru(self, key: tuple[str, int | None]) -> Entry | None:
        for entry in self.rows[key].values():
            if entry.pin_count == 0:
                return entry
        return None

    def trim(self, tier: str, worker: int | None, evictor, request_index: int) -> list[Entry]:
        """Evict unpinned entries until used <= capacity. Returns the list of evicted entries."""
        key = (tier, worker)
        cap = self.capacity[key]
        if self.used[key] <= cap:
            return []
        evicted: list[Entry] = []
        if _is_lru(evictor):
            row = self.rows[key]
            # Fast single-pass O(k) LRU eviction from OrderedDict head
            to_drop: list[Entry] = []
            excess = self.used[key] - cap
            for entry in row.values():
                if entry.pin_count == 0:
                    to_drop.append(entry)
                    excess -= entry.nbytes
                    if excess <= 0:
                        break
            for entry in to_drop:
                del row[entry.block_id]
                if tier == "hbm" and worker is not None:
                    owners = self._hbm_owners.get(entry.block_id)
                    if owners is not None:
                        owners.discard(worker)
                        if not owners:
                            del self._hbm_owners[entry.block_id]
                self.used[key] -= entry.nbytes
                self.evictions += 1
                evicted.append(entry)
            if self.used[key] < 0:
                self.used[key] = 0
            return evicted

        keyfn = _order_key(evictor)
        if keyfn is not None:
            self._compact(key, evictor)
        # Fast batch path for RR-Evict (single O(N) grouping per trim call)
        choose_batch = getattr(evictor, "choose_batch", None)
        if keyfn is None and choose_batch is not None:
            victims = choose_batch(self.rows[key], self.used[key] - cap)
            for v in victims:
                if v.pin_count > 0:
                    self.pinned_evictions += 1
                    break
                self._drop(v)
                evicted.append(v)
            return evicted
        # Fast direct-row scan for SIEVE (exploits FIFO OrderedDict without copying list)
        choose_row = getattr(evictor, "choose_from_row", None)
        if keyfn is None and choose_row is not None:
            row = self.rows[key]
            while self.used[key] > cap:
                v = choose_row(row)
                if v is None:
                    break
                if v.pin_count > 0:
                    self.pinned_evictions += 1
                    break
                self._drop(v)
                evicted.append(v)
            return evicted
        spins = 0
        promotes = 0
        while self.used[key] > cap:
            spins += 1
            if spins > 200000:
                break
            if keyfn is None:
                decision = evictor.choose(list(self.rows[key].values()), request_index)
            else:
                decision = self._victim_reuse(key, evictor)
            if decision is None:
                break
            if decision == "promoted":
                promotes += 1
                if promotes > len(self.rows[key]) + 8:
                    forced = self._oldest_unpinned(key)
                    if forced is None:
                        break
                    self._drop(forced)
                    evicted.append(forced)
                continue
            if decision.pin_count > 0:
                self.pinned_evictions += 1
                break
            if keyfn is not None and decision.evict_score > self._aging.get(key, 0.0):
                self._aging[key] = decision.evict_score
            self._drop(decision)
            evicted.append(decision)
        return evicted

    def _oldest_unpinned(self, key: tuple[str, int | None]) -> Entry | None:
        for entry in self.rows[key].values():
            if entry.pin_count == 0:
                return entry
        return None

    def _drop(self, entry: Entry, count_eviction: bool = True) -> None:
        key = (entry.tier, entry.worker)
        row = self.rows[key]
        current = row.get(entry.block_id)
        if current is None:
            return
        del row[entry.block_id]
        if entry.tier == "hbm" and entry.worker is not None:
            owners = self._hbm_owners.get(entry.block_id)
            if owners is not None:
                owners.discard(entry.worker)
                if not owners:
                    del self._hbm_owners[entry.block_id]
        self._invalidate(key, entry.block_id)
        self.used[key] -= current.nbytes
        if self.used[key] < 0:
            self.used[key] = 0
        if count_eviction:
            self.evictions += 1

    def drop_if_unpinned(
        self,
        tier: str,
        worker: int | None,
        block_id: str,
        count_eviction: bool = True,
    ) -> bool:
        entry = self.rows[(tier, worker)].get(block_id)
        if entry is None or entry.pin_count > 0:
            return False
        self._drop(entry, count_eviction=count_eviction)
        return True

    def unpin_many(self, entries: list[Entry]) -> None:
        for entry in entries:
            if entry.pin_count > 0:
                entry.pin_count -= 1
