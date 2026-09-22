"""Package-wide logging for agent-memory-sdk.

Every module obtains its logger via::

    from agent_memory.logging_config import get_logger
    log = get_logger(__name__)

The root logger is ``agent_memory``.  Consumers can configure it however
they like using standard Python logging::

    import logging
    logging.getLogger("agent_memory").setLevel(logging.DEBUG)
    logging.getLogger("agent_memory").addHandler(logging.StreamHandler())

Log levels used by this package:
  DEBUG   — per-query details (query text, scores, latency)
  INFO    — lifecycle events (store initialised, memory stored/forgotten)
  WARNING — degraded-mode notices (no embeddings, fallback BM25)
  ERROR   — operation failures (caught exceptions with full context)
"""
from __future__ import annotations

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

_ROOT = "agent_memory"


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the ``agent_memory`` root logger.

    Typically called as ``get_logger(__name__)`` at module level.
    """
    # Strip the package prefix so child names are shorter, e.g.
    # "agent_memory.sqlite_store" → logger name "agent_memory.sqlite_store"
    if not name.startswith(_ROOT):
        name = f"{_ROOT}.{name}"
    return logging.getLogger(name)


@contextmanager
def timed(
    log: logging.Logger,
    operation: str,
    level: int = logging.DEBUG,
    **context: Any,
) -> Generator[None, None, None]:
    """Context manager that logs the wall-clock duration of a block.

    Usage::

        with timed(log, "keyword_search", query=query, top_k=top_k):
            results = store.keyword_search(query, top_k=top_k)
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        ctx_str = "  ".join(f"{k}={v!r}" for k, v in context.items())
        log.log(level, "%s  %.2fms  %s", operation, elapsed_ms, ctx_str)


def configure_debug_logging() -> None:
    """One-call helper to emit DEBUG logs to stderr.

    Intended for development use only — production code should configure
    logging through the standard ``logging`` module.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s")
    )
    root = logging.getLogger(_ROOT)
    root.setLevel(logging.DEBUG)
    if not root.handlers:
        root.addHandler(handler)
