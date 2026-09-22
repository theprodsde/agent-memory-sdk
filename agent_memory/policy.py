from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from agent_memory.logging_config import get_logger
from agent_memory.models import (
    VERIFY_TYPES,
    MemoryAction,
    MemoryEntry,
    RetrievalResult,
)

log = get_logger(__name__)


class DecisionPolicy(ABC):
    """Scoring and action-selection policy for memory resolution."""

    @abstractmethod
    def score(self, entry: MemoryEntry, semantic: float, keyword: float) -> float:
        ...

    @abstractmethod
    def select_action(
        self,
        results: list[RetrievalResult],
        *,
        replay_threshold: float,
        restore_threshold: float,
        verify_threshold: float,
    ) -> tuple[MemoryAction, float, str]:
        ...


class DefaultPolicy(DecisionPolicy):
    """
    Default policy combining:
      semantic score + recency + confidence + usage
    """

    def __init__(
        self,
        semantic_weight: float = 0.55,
        recency_weight: float = 0.15,
        confidence_weight: float = 0.20,
        usage_weight: float = 0.10,
        recency_half_life_days: float = 30.0,
        usage_cap: int = 20,
    ) -> None:
        self.semantic_weight = semantic_weight
        self.recency_weight = recency_weight
        self.confidence_weight = confidence_weight
        self.usage_weight = usage_weight
        self.recency_half_life_days = recency_half_life_days
        self.usage_cap = usage_cap

    def score(
        self,
        entry: MemoryEntry,
        semantic: float,
        keyword: float,
        *,
        now: datetime | None = None,
    ) -> float:
        return self.score_breakdown(entry, semantic, keyword, now=now)["policy_score"]

    def score_breakdown(
        self,
        entry: MemoryEntry,
        semantic: float,
        keyword: float,
        *,
        now: datetime | None = None,
    ) -> dict[str, float]:
        """Compute per-component scores.

        *now* is shared across all entries in a single retrieve() call so
        recency is consistent and we avoid N ``datetime.now()`` syscalls — a
        simple but correct form of memoisation.
        """
        # Let the stronger retrieval channel dominate: embeddings score
        # paraphrases conservatively, keyword coverage scores them lexically —
        # solid evidence from either channel should carry the match.
        hybrid_semantic = max(
            0.7 * semantic + 0.3 * keyword,
            0.7 * keyword + 0.3 * semantic,
        )
        recency = self._recency_score(entry.updated_at, now=now)
        usage = min(entry.access_count, self.usage_cap) / self.usage_cap
        confidence = entry.confidence
        policy_score = (
            self.semantic_weight * hybrid_semantic
            + self.recency_weight * recency
            + self.confidence_weight * confidence
            + self.usage_weight * usage
        )
        return {
            "semantic_score": hybrid_semantic,
            "keyword_score": keyword,
            "recency_score": recency,
            "confidence_score": confidence,
            "usage_score": usage,
            "policy_score": policy_score,
            "final_score": policy_score,
        }

    def select_action(
        self,
        results: list[RetrievalResult],
        *,
        replay_threshold: float,
        restore_threshold: float,
        verify_threshold: float,
    ) -> tuple[MemoryAction, float, str]:
        best = results[0]
        score = best.decision_score
        entry = best.entry

        # An explicit requires_verification flag outranks replay: the caller
        # marked this memory as needing validation before any reuse.
        if entry.requires_verification and score >= restore_threshold:
            return (
                MemoryAction.VERIFY,
                score,
                "Memory is flagged requires_verification — validate before reuse.",
            )

        if score >= replay_threshold:
            return (
                MemoryAction.REPLAY,
                score,
                "High composite score — replaying stored response.",
            )

        if score >= restore_threshold:
            if self._should_verify(entry, score, verify_threshold):
                return (
                    MemoryAction.VERIFY,
                    score,
                    "Moderate score on freshness-sensitive memory — verify before reuse.",
                )
            return (
                MemoryAction.RESTORE,
                score,
                "Moderate score — restoring memory as context for synthesis.",
            )

        return (
            MemoryAction.NONE,
            score,
            "Low composite score — no memory applied.",
        )

    def _should_verify(self, entry: MemoryEntry, score: float, verify_threshold: float) -> bool:
        if entry.requires_verification:
            return True
        if entry.type.value in VERIFY_TYPES and score < verify_threshold:
            return True
        age_days = (datetime.now(timezone.utc) - entry.updated_at).total_seconds() / 86400
        if entry.type.value in VERIFY_TYPES and age_days > self.recency_half_life_days:
            return True
        return False

    def _recency_score(
        self, updated_at: datetime, *, now: datetime | None = None
    ) -> float:
        _now = now or datetime.now(timezone.utc)
        ts = updated_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (_now - ts).total_seconds() / 86400)
        return math.exp(-0.693 * age_days / self.recency_half_life_days)
