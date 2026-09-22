from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_memory import Memory
from agent_memory.benchmark import format_benchmark_report, run_benchmark
from agent_memory.eval import EvalDataset, format_eval_report, run_eval_suite, seed_dataset


def _default_datasets_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "benchmarks" / "datasets"


def _create_memory(args: argparse.Namespace) -> Memory:
    return Memory(persist_dir=args.data_dir, backend=args.backend)


def cmd_remember(args: argparse.Namespace) -> int:
    memory = _create_memory(args)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
    entry = memory.remember(
        args.query,
        args.response,
        type=args.type,
        scope=args.scope,
        ttl=args.ttl,
        tags=tags,
        confidence=args.confidence,
        requires_verification=args.requires_verification,
    )
    print(f"✓ Stored  id={entry.id[:16]}…  [{entry.type.value}]  scope={entry.scope.value}")
    if entry.tags:
        print(f"  tags: {', '.join(entry.tags)}")
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    memory = _create_memory(args)
    decision = memory.resolve(args.query)

    ACTION_LABEL = {
        "replay":  "✅ REPLAY",
        "restore": "📋 RESTORE",
        "verify":  "⚠️  VERIFY",
        "none":    "❌ NONE",
    }
    print(f"action:     {ACTION_LABEL.get(decision.action.value, decision.action.value)}")
    print(f"confidence: {decision.confidence:.2f}")

    if decision.reasons:
        print(f"reasons:    {' · '.join(decision.reasons)}")

    if decision.action.value == "replay" and decision.memory:
        e = decision.memory
        print(f'matched:    "{e.query}"')
        print(f'stored:     {e.created_at:%Y-%m-%d}  reused {e.access_count}×  confidence {e.confidence:.0%}')
        print(f"response:   {decision.response}")

    elif decision.action.value in ("restore", "verify") and decision.context:
        print(f"matches:    {len(decision.context)} context entr{'y' if len(decision.context) == 1 else 'ies'}")
        for i, ctx in enumerate(decision.context, 1):
            e = ctx.entry
            print(f"  [{i}] score={ctx.final_score:.2f}  [{e.type.value}]  {e.query[:70]}")
            print(f"       → {e.response[:100]}")
        if decision.action.value == "verify":
            print("  ⚠  This memory requires verification before reuse.")

    if args.explain:
        print()
        print(decision.explain())
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    memory = _create_memory(args)
    stats = memory.stats()
    print("Agent Memory Stats")
    print("==================")
    print(f"total:        {stats.get('total', 0)}")
    print(f"access count: {stats.get('total_access_count', 0)}")
    by_state = stats.get("by_state", {})
    if by_state:
        print("by state:     " + "  ".join(f"{k}={v}" for k, v in sorted(by_state.items())))
    by_type = stats.get("by_type", {})
    if by_type:
        print("by type:      " + "  ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
    return 0


def cmd_cleanup(args: argparse.Namespace) -> int:
    memory = _create_memory(args)
    result = memory.cleanup(delete=args.delete)
    print(result)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    memory = _create_memory(args)

    if args.seed:
        for dataset_path in _default_datasets_dir().glob("*.json"):
            dataset = EvalDataset.load(dataset_path)
            seed_dataset(memory, dataset)
            for case in dataset.cases:
                args.queries.extend([case.query] * max(1, args.repeat))

    queries = args.queries or [
        "How do I reset my password?",
        "I forgot my password",
        "What is the API rate limit?",
        "What's the weather today?",
    ]

    if args.repeat > 1 and not args.seed:
        queries = [q for q in queries for _ in range(args.repeat)]

    result, comparison = run_benchmark(memory, queries, baseline_no_memory_ms=args.baseline_ms)
    print(format_benchmark_report(result, comparison))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    memory = Memory(persist_dir=args.data_dir)
    datasets_dir = Path(args.datasets) if args.datasets else _default_datasets_dir()

    if not datasets_dir.exists():
        print(f"Datasets directory not found: {datasets_dir}", file=sys.stderr)
        return 1

    paths = sorted(datasets_dir.glob("*.json"))
    if not paths:
        print(f"No datasets found in {datasets_dir}", file=sys.stderr)
        return 1

    results = run_eval_suite(memory, paths)
    print(format_eval_report(results))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-memory", description="Agent Memory CLI")
    parser.add_argument(
        "--data-dir",
        default=".agent_memory",
        help="Persistence directory (default: .agent_memory)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    def add_common_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--backend",
            choices=["sqlite", "chromadb", "redis", "postgres"],
            default="sqlite",
            help="Storage backend (default: sqlite)",
        )

    remember = sub.add_parser("remember", help="Store a memory")
    remember.add_argument("query")
    remember.add_argument("response")
    remember.add_argument("--type",  default="conversation",
                          help="Memory type: conversation|fact|workflow|tool_output|code|preference|document|summary")
    remember.add_argument("--scope", default="user",
                          help="Memory scope: user|project|team|global|session|workspace")
    remember.add_argument("--ttl",   default=None,
                          help="Expiry: 30d, 2h, 3600 (seconds)")
    remember.add_argument("--tags",  default="",
                          help="Comma-separated tags, e.g. auth,faq")
    remember.add_argument("--confidence", type=float, default=1.0,
                          help="Confidence score 0.0–1.0 (default 1.0)")
    remember.add_argument("--requires-verification", action="store_true",
                          help="Always return VERIFY instead of REPLAY for this memory")
    add_common_args(remember)
    remember.set_defaults(func=cmd_remember)

    resolve = sub.add_parser("resolve", help="Resolve a query against memory")
    resolve.add_argument("query")
    resolve.add_argument("--explain", action="store_true", help="Print score breakdown")
    add_common_args(resolve)
    resolve.set_defaults(func=cmd_resolve)

    stats = sub.add_parser("stats", help="Show memory statistics")
    add_common_args(stats)
    stats.set_defaults(func=cmd_stats)

    cleanup = sub.add_parser("cleanup", help="Expire or delete stale memories")
    cleanup.add_argument("--delete", action="store_true", help="Delete expired memories")
    add_common_args(cleanup)
    cleanup.set_defaults(func=cmd_cleanup)

    benchmark = sub.add_parser("benchmark", help="Run latency and hit-rate benchmark")
    benchmark.add_argument("queries", nargs="*", help="Queries to benchmark")
    benchmark.add_argument("--seed", action="store_true", help="Seed from eval datasets first")
    benchmark.add_argument("--repeat", type=int, default=1, help="Repeat each query N times")
    benchmark.add_argument(
        "--baseline-ms",
        type=float,
        default=None,
        help="Your measured no-memory latency in ms (enables the comparison section)",
    )
    add_common_args(benchmark)
    benchmark.set_defaults(func=cmd_benchmark)

    eval_cmd = sub.add_parser("eval", help="Run evaluation datasets")
    eval_cmd.add_argument("--datasets", default=None, help="Path to datasets directory")
    add_common_args(eval_cmd)
    eval_cmd.set_defaults(func=cmd_eval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
