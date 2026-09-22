"""Tests for the ConfidenceLearner — event deltas, decay, batch updates."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent_memory.confidence import ConfidenceEvent, ConfidenceLearner, ConfidenceUpdate
from agent_memory.models import MemoryEntry


@pytest.fixture()
def learner():
    return ConfidenceLearner()


@pytest.fixture()
def entry():
    return MemoryEntry(query="test query", response="test response", confidence=0.5)


# ---------------------------------------------------------------------------
# Individual event deltas
# ---------------------------------------------------------------------------


def test_access_boosts_confidence(learner, entry):
    update = learner.record_event(entry, ConfidenceEvent.ACCESSED)
    assert update.new_confidence > update.old_confidence
    assert update.delta == pytest.approx(ConfidenceLearner.ACCESS_BOOST)


def test_verified_correct_boosts(learner, entry):
    update = learner.record_event(entry, ConfidenceEvent.VERIFIED_CORRECT)
    assert update.delta == pytest.approx(ConfidenceLearner.CORRECT_BOOST)


def test_verified_incorrect_penalizes(learner, entry):
    update = learner.record_event(entry, ConfidenceEvent.VERIFIED_INCORRECT)
    assert update.delta == pytest.approx(-ConfidenceLearner.INCORRECT_PENALTY)
    assert update.new_confidence < update.old_confidence


def test_user_confirmed_boosts(learner, entry):
    update = learner.record_event(entry, ConfidenceEvent.USER_CONFIRMED)
    assert update.delta == pytest.approx(ConfidenceLearner.CONFIRM_BOOST)


def test_user_rejected_penalizes(learner, entry):
    update = learner.record_event(entry, ConfidenceEvent.USER_REJECTED)
    assert update.delta == pytest.approx(-ConfidenceLearner.REJECT_PENALTY)


def test_stale_penalizes(learner, entry):
    update = learner.record_event(entry, ConfidenceEvent.STALE)
    assert update.delta == pytest.approx(-ConfidenceLearner.STALE_PENALTY)


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------


def test_confidence_does_not_exceed_one(learner):
    e = MemoryEntry(query="q", response="r", confidence=0.99)
    for _ in range(10):
        learner.record_event(e, ConfidenceEvent.VERIFIED_CORRECT)
    assert e.confidence <= 1.0


def test_confidence_does_not_go_below_zero(learner):
    e = MemoryEntry(query="q", response="r", confidence=0.01)
    for _ in range(10):
        learner.record_event(e, ConfidenceEvent.USER_REJECTED)
    assert e.confidence >= 0.0


# ---------------------------------------------------------------------------
# Decay
# ---------------------------------------------------------------------------


def test_decay_reduces_confidence(learner, entry):
    # Simulate an entry last updated 180 days ago
    old_time = datetime.now(timezone.utc) - timedelta(days=180)
    entry.updated_at = old_time
    update = learner.decay(entry)
    assert update.new_confidence < update.old_confidence


def test_decay_half_life(learner):
    e = MemoryEntry(query="q", response="r", confidence=1.0)
    e.updated_at = datetime.now(timezone.utc) - timedelta(days=ConfidenceLearner.HALF_LIFE_DAYS)
    learner.decay(e)
    assert e.confidence == pytest.approx(0.5, abs=0.02)


def test_decay_fresh_entry_minimal_change(learner):
    e = MemoryEntry(query="q", response="r", confidence=1.0)
    # Updated now — essentially no decay
    e.updated_at = datetime.now(timezone.utc)
    learner.decay(e)
    assert e.confidence > 0.99


# ---------------------------------------------------------------------------
# Batch update
# ---------------------------------------------------------------------------


def test_batch_update(learner):
    entries = [
        MemoryEntry(query=f"q{i}", response=f"r{i}", confidence=0.5) for i in range(3)
    ]
    events = [
        ConfidenceEvent.VERIFIED_CORRECT,
        ConfidenceEvent.VERIFIED_INCORRECT,
        ConfidenceEvent.ACCESSED,
    ]
    updates = learner.batch_update(entries, events)
    assert len(updates) == 3
    assert updates[0].delta > 0
    assert updates[1].delta < 0
    assert updates[2].delta > 0


def test_batch_update_length_mismatch(learner):
    entries = [MemoryEntry(query="q", response="r")]
    events = [ConfidenceEvent.ACCESSED, ConfidenceEvent.STALE]
    with pytest.raises(ValueError):
        learner.batch_update(entries, events)


# ---------------------------------------------------------------------------
# ConfidenceUpdate dataclass
# ---------------------------------------------------------------------------


def test_update_dataclass(learner, entry):
    update = learner.record_event(entry, ConfidenceEvent.ACCESSED)
    assert isinstance(update, ConfidenceUpdate)
    assert update.memory_id == entry.id
    assert update.event == ConfidenceEvent.ACCESSED
    assert update.delta == pytest.approx(update.new_confidence - update.old_confidence)
