from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from agent_memory.exceptions import RetrievalError
from agent_memory.logging_config import get_logger
from agent_memory.models import MemoryEntry, MemoryScope, RetrievalResult
from agent_memory.policy import DecisionPolicy, DefaultPolicy
from agent_memory.store import MemoryStore

log = get_logger(__name__)


# Module-level bucket class for RRF fusion — defined here so Python doesn't
# recreate the class on every fuse() call (each class creation costs ~350µs).
@dataclass
class _RRFBucket:
    entry: MemoryEntry
    semantic: float = 0.0
    keyword: float = 0.0
    rrf: float = 0.0


# ---------------------------------------------------------------------------
# Fusion strategy interface — Open/Closed principle
# ---------------------------------------------------------------------------

_FusedTriple = tuple[MemoryEntry, float, float]   # entry, semantic, keyword


class FusionStrategy(ABC):
    """Abstract strategy for fusing vector and keyword retrieval lists.

    Implement this to swap in a different fusion algorithm (CombSUM,
    weighted linear, learned reranker, …) without touching MemoryRetriever.
    """

    @abstractmethod
    def fuse(
        self,
        vector_hits: list[tuple[MemoryEntry, float]],
        keyword_hits: list[tuple[MemoryEntry, float]],
    ) -> list[_FusedTriple]:
        """Return entries sorted by combined score descending."""


class RRFFusionStrategy(FusionStrategy):
    """Reciprocal Rank Fusion — the default.

    RRF score = Σ 1/(k + rank_i) for each list i.
    Outperforms simple score combination on most retrieval benchmarks.
    k=60 is the empirically strong default from the original RRF paper.
    """

    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(
        self,
        vector_hits: list[tuple[MemoryEntry, float]],
        keyword_hits: list[tuple[MemoryEntry, float]],
    ) -> list[_FusedTriple]:
        buckets: dict[str, _RRFBucket] = {}
        k = self.k

        for rank, (entry, score) in enumerate(vector_hits, start=1):
            b = buckets.setdefault(entry.id, _RRFBucket(entry=entry))
            b.semantic = max(b.semantic, score)
            b.rrf += 1.0 / (k + rank)

        for rank, (entry, score) in enumerate(keyword_hits, start=1):
            b = buckets.setdefault(entry.id, _RRFBucket(entry=entry))
            b.keyword = max(b.keyword, score)
            b.rrf += 1.0 / (k + rank)

        fused = sorted(buckets.values(), key=lambda b: b.rrf, reverse=True)
        return [(b.entry, b.semantic, b.keyword) for b in fused]


class LinearFusionStrategy(FusionStrategy):
    """Weighted linear combination of normalised scores.

    Simpler than RRF but easier to tune when semantic and keyword
    channels have very different score distributions.
    """

    def __init__(
        self, semantic_weight: float = 0.6, keyword_weight: float = 0.4
    ) -> None:
        self._sw = semantic_weight
        self._kw = keyword_weight

    def fuse(
        self,
        vector_hits: list[tuple[MemoryEntry, float]],
        keyword_hits: list[tuple[MemoryEntry, float]],
    ) -> list[_FusedTriple]:
        scores: dict[str, list[float]] = {}   # id → [semantic, keyword]
        entries: dict[str, MemoryEntry] = {}

        for entry, score in vector_hits:
            scores.setdefault(entry.id, [0.0, 0.0])[0] = max(scores.get(entry.id, [0.0, 0.0])[0], score)
            entries[entry.id] = entry

        for entry, score in keyword_hits:
            scores.setdefault(entry.id, [0.0, 0.0])[1] = max(scores.get(entry.id, [0.0, 0.0])[1], score)
            entries[entry.id] = entry

        combined = [
            (entries[eid], s[0], s[1], self._sw * s[0] + self._kw * s[1])
            for eid, s in scores.items()
        ]
        combined.sort(key=lambda x: x[3], reverse=True)
        return [(e, sem, kw) for e, sem, kw, _ in combined]


class _LRUCache:
    """Simple fixed-size LRU cache with TTL for retrieval results.

    Used to avoid redundant SQLite queries when the same query string is
    resolved multiple times within a short window (e.g. concurrent agents,
    test loops).  Uses an ``OrderedDict`` for O(1) move-to-end.
    """

    def __init__(self, maxsize: int = 256, ttl_seconds: float = 5.0) -> None:
        from collections import OrderedDict
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl_seconds

    def get(self, key: str) -> Any | None:
        if key not in self._cache:
            return None
        ts, value = self._cache[key]
        if time.monotonic() - ts > self._ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (time.monotonic(), value)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def invalidate(self) -> None:
        self._cache.clear()


class MemoryRetriever:
    """Hybrid retrieval: BM25 keyword search + vector search + policy rerank.

    Results for identical queries within a short window (default 5 s) are
    served from an in-process LRU cache to avoid redundant SQLite round-trips.
    Call ``invalidate_cache()`` whenever memories are written to ensure
    subsequent resolves see fresh results.
    """

    def __init__(
        self,
        store: MemoryStore,
        policy: DecisionPolicy | None = None,
        *,
        fusion: FusionStrategy | None = None,
        cache_size: int = 256,
        cache_ttl: float = 5.0,
    ) -> None:
        self._store = store
        self._policy = policy or DefaultPolicy()
        self._fusion: FusionStrategy = fusion or RRFFusionStrategy()
        self._cache = _LRUCache(maxsize=cache_size, ttl_seconds=cache_ttl)

    @property
    def policy(self) -> DecisionPolicy:
        return self._policy

    @property
    def _has_semantic_search(self) -> bool:
        """True only when the store can return genuinely different semantic hits."""
        return bool(getattr(self._store, "semantic_search_enabled", False))

    def invalidate_cache(self) -> None:
        """Clear the query result cache (call after any write operation)."""
        self._cache.invalidate()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        scopes: list[MemoryScope] | None = None,
        diversify_key: str | None = None,
        max_per_group: int = 2,
    ) -> list[RetrievalResult]:
        """Retrieve top-k memories via hybrid search.

        When *diversify_key* names a metadata field, the top-k is spread
        across groups: at most *max_per_group* entries per distinct value of
        that field (e.g. ``diversify_key="session_id"``), so one dominant
        group cannot crowd out evidence living elsewhere.  Remaining slots
        are back-filled by score if there are too few groups.
        """
        cache_key = (
            f"{query}|{top_k}|{sorted(s.value for s in scopes) if scopes else ''}"
            f"|{diversify_key}|{max_per_group if diversify_key else ''}"
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            log.debug("retrieve cache_hit query=%r top_k=%d", query, top_k)
            return cached  # type: ignore[no-any-return]

        # Diversification needs a deeper candidate pool to pick groups from.
        fetch_k = top_k * 3 if diversify_key else top_k * 2

        t0 = time.perf_counter()
        try:
            # Always run keyword search — fast, reliable baseline.
            keyword_hits = self._store.keyword_search(query, top_k=fetch_k, scopes=scopes)

            if self._has_semantic_search:
                vector_hits = self._store.search(query, top_k=fetch_k, scopes=scopes)
            else:
                # Without embeddings, search() == keyword_search(); skip the duplicate.
                vector_hits = keyword_hits

        except Exception as exc:
            raise RetrievalError(
                f"Retrieval failed for query {query!r}: {exc}", query=query
            ) from exc

        # Share a single timestamp across all policy score calls — saves
        # N datetime.now() syscalls and makes scoring consistent within a batch.
        now = datetime.now(timezone.utc)
        fused = self._fusion.fuse(vector_hits, keyword_hits)

        pool_n = top_k * 3 if diversify_key else top_k
        results: list[RetrievalResult] = []
        for rank, (entry, semantic, keyword) in enumerate(fused[:pool_n], start=1):
            final_score = self._policy.score(entry, semantic, keyword, now=now)  # type: ignore[call-arg]
            results.append(
                RetrievalResult(
                    entry=entry,
                    semantic_score=semantic,
                    keyword_score=keyword,
                    final_score=final_score,
                    rank=rank,
                )
            )

        results.sort(key=lambda r: r.final_score, reverse=True)

        if diversify_key:
            picked: list[RetrievalResult] = []
            skipped: list[RetrievalResult] = []
            group_counts: dict[object, int] = {}
            for result in results:
                group = (result.entry.metadata or {}).get(diversify_key)
                if group is None or group_counts.get(group, 0) < max_per_group:
                    picked.append(result)
                    if group is not None:
                        group_counts[group] = group_counts.get(group, 0) + 1
                else:
                    skipped.append(result)
                if len(picked) == top_k:
                    break
            # Back-fill by score when there are fewer groups than slots.
            picked.extend(skipped[: top_k - len(picked)])
            results = picked
        else:
            results = results[:top_k]

        for index, result in enumerate(results, start=1):
            result.rank = index

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.debug(
            "retrieve  %.2fms  query=%r  hits=%d  semantic=%s",
            elapsed_ms, query, len(results), self._has_semantic_search,
        )
        self._cache.set(cache_key, results)
        return results

    def retrieve_best(
        self,
        query: str,
        *,
        scopes: list[MemoryScope] | None = None,
    ) -> RetrievalResult | None:
        results = self.retrieve(query, top_k=1, scopes=scopes)
        return results[0] if results else None

    def record_access(self, entry: MemoryEntry) -> MemoryEntry:
        """Increment access count without triggering a full content reindex."""
        entry.touch()                   # update in-memory object
        self._store.touch(entry.id)     # fast SQL UPDATE, no FTS5 reindex
        return entry

