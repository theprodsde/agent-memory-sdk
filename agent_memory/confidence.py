from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from agent_memory.models import MemoryEntry


class ConfidenceEvent(str, Enum):
    """Events that update a memory's confidence score."""

    ACCESSED = "accessed"
    VERIFIED_CORRECT = "verified_correct"
    VERIFIED_INCORRECT = "verified_incorrect"
    USER_CONFIRMED = "user_confirmed"
    USER_REJECTED = "user_rejected"
    STALE = "stale"


@dataclass
class ConfidenceUpdate:
    """Record of a single confidence change."""

    memory_id: str
    event: ConfidenceEvent
    old_confidence: float
    new_confidence: float
    delta: float


class ConfidenceLearner:
    """Adaptive confidence scoring based on feedback events.

    Confidence is updated as a clamped additive delta so that a single wrong
    answer does not completely invalidate a well-established memory and a
    single correct answer does not over-inflate a new one.

    Temporal decay uses a half-life model: confidence halves every
    ``half_life_days`` days of inactivity.

    Usage::

        learner = ConfidenceLearner()

        # After a correct verification
        update = learner.record_event(entry, ConfidenceEvent.VERIFIED_CORRECT)

        # Decay all memories nightly
        updates = [learner.decay(e) for e in memory.list()]
    """

    #: Deltas applied for each event type (positive = boost, negative = penalty).
    ACCESS_BOOST: float = 0.02
    CORRECT_BOOST: float = 0.10
    INCORRECT_PENALTY: float = 0.20
    CONFIRM_BOOST: float = 0.15
    REJECT_PENALTY: float = 0.25
    STALE_PENALTY: float = 0.05

    #: Days until confidence halves under temporal decay.
    HALF_LIFE_DAYS: float = 90.0

    def record_event(
        self,
        entry: MemoryEntry,
        event: ConfidenceEvent,
        *,
        persist_timestamp: bool = False,
    ) -> ConfidenceUpdate:
        """Apply *event* to *entry*.confidence and return the change record.

        Set ``persist_timestamp=True`` to also bump ``entry.updated_at`` when
        confidence changes (useful when you will persist the entry afterwards).
        """
        old = entry.confidence
        delta = self._delta_for(event)
        new = max(0.0, min(1.0, old + delta))
        entry.confidence = new
        if persist_timestamp and delta != 0:
            from datetime import datetime, timezone

            entry.updated_at = datetime.now(timezone.utc)
        return ConfidenceUpdate(
            memory_id=entry.id,
            event=event,
            old_confidence=old,
            new_confidence=new,
            delta=delta,
        )

    def decay(self, entry: MemoryEntry) -> ConfidenceUpdate:
        """Apply temporal decay proportional to time since last update.

        The half-life model ensures that a memory with confidence 1.0
        decays to ~0.5 after ``HALF_LIFE_DAYS`` days of no updates.
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        updated = entry.updated_at
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - updated).total_seconds() / 86_400.0)
        # exp(-ln2 * age / half_life)  →  halves every half_life days
        decay_factor = math.exp(-0.693 * age_days / self.HALF_LIFE_DAYS)
        old = entry.confidence
        new = max(0.0, min(1.0, old * decay_factor))
        entry.confidence = new
        return ConfidenceUpdate(
            memory_id=entry.id,
            event=ConfidenceEvent.STALE,
            old_confidence=old,
            new_confidence=new,
            delta=new - old,
        )

    def batch_update(
        self,
        entries: list[MemoryEntry],
        events: list[ConfidenceEvent],
    ) -> list[ConfidenceUpdate]:
        """Apply one event per entry, returning all change records."""
        if len(entries) != len(events):
            raise ValueError("entries and events must have the same length")
        return [self.record_event(e, ev) for e, ev in zip(entries, events)]

    def _delta_for(self, event: ConfidenceEvent) -> float:
        if event == ConfidenceEvent.ACCESSED:
            return self.ACCESS_BOOST
        if event == ConfidenceEvent.VERIFIED_CORRECT:
            return self.CORRECT_BOOST
        if event == ConfidenceEvent.VERIFIED_INCORRECT:
            return -self.INCORRECT_PENALTY
        if event == ConfidenceEvent.USER_CONFIRMED:
            return self.CONFIRM_BOOST
        if event == ConfidenceEvent.USER_REJECTED:
            return -self.REJECT_PENALTY
        if event == ConfidenceEvent.STALE:
            return -self.STALE_PENALTY
        return 0.0
