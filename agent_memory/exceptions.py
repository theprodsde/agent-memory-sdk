"""Custom exception hierarchy for agent-memory-sdk.

All exceptions inherit from ``AgentMemoryError`` so callers can catch the
entire family with a single except clause, or target a specific sub-type.

Hierarchy::

    AgentMemoryError
    ├── ConfigurationError          invalid constructor args / env vars
    ├── BackendConnectionError      cannot reach the storage backend
    ├── MemoryStoreError            backend operation failed
    │   ├── MemoryNotFoundError     requested entry does not exist
    │   └── MemoryWriteError        write / upsert failed
    ├── RetrievalError              search / retrieval failed
    ├── DecisionError               decision engine error
    └── ExtractionError             entity / memory extraction failed
"""
from __future__ import annotations


class AgentMemoryError(Exception):
    """Base class for all agent-memory-sdk exceptions."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigurationError(AgentMemoryError):
    """Invalid SDK configuration (bad backend name, missing secrets, …)."""


# ---------------------------------------------------------------------------
# Backend / storage
# ---------------------------------------------------------------------------


class BackendConnectionError(AgentMemoryError):
    """Cannot connect to the storage backend (Redis down, wrong DSN, …).

    Raised during ``__init__`` when the backend is unreachable.
    """

    def __init__(self, backend: str, reason: str) -> None:
        self.backend = backend
        self.reason = reason
        super().__init__(f"Cannot connect to {backend} backend: {reason}")


class MemoryStoreError(AgentMemoryError):
    """A backend storage operation failed."""

    def __init__(self, message: str, *, backend: str = "", memory_id: str = "") -> None:
        self.backend = backend
        self.memory_id = memory_id
        super().__init__(message)


class MemoryNotFoundError(MemoryStoreError):
    """The requested memory entry does not exist."""

    def __init__(self, memory_id: str, *, backend: str = "") -> None:
        super().__init__(
            f"Memory '{memory_id}' not found",
            backend=backend,
            memory_id=memory_id,
        )


class MemoryWriteError(MemoryStoreError):
    """A write or upsert operation failed."""


# ---------------------------------------------------------------------------
# Retrieval / decision
# ---------------------------------------------------------------------------


class RetrievalError(AgentMemoryError):
    """Search or retrieval operation failed."""

    def __init__(self, message: str, *, query: str = "") -> None:
        self.query = query
        super().__init__(message)


class DecisionError(AgentMemoryError):
    """The decision engine encountered an unexpected error."""


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class ExtractionError(AgentMemoryError):
    """Entity or memory extraction failed."""
