from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_memory import Memory
from agent_memory.benchmark import format_benchmark_report, run_benchmark
from agent_memory.eval import EvalDataset, format_eval_report, run_eval_suite, seed_dataset
from agent_memory.hooks import DEFAULT_GATE_MIN_RELEVANCE, HookConfig


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


def build_hook_config(argv: list[str] | None = None) -> HookConfig:
    """Parse ``agent-memory hook`` flags into a :class:`HookConfig`.

    Lives here rather than in :mod:`agent_memory.hooks` so the hook module
    stays importable without argparse-time side effects, and so the flag names
    are defined next to every other subcommand's.
    """
    parser = argparse.ArgumentParser(
        prog="agent-memory hook",
        description="Resolve a UserPromptSubmit hook payload against memory",
    )
    parser.add_argument(
        "event",
        nargs="?",
        default="user-prompt-submit",
        choices=["user-prompt-submit"],
        help="Hook event to handle (default: user-prompt-submit)",
    )
    parser.add_argument(
        "--mode",
        choices=["context", "gate"],
        default=None,
        help="context: inject memory alongside the prompt (default). "
             "gate: let a high-confidence REPLAY answer without a model call",
    )
    parser.add_argument("--data-dir", default=None, help="Persistence directory")
    parser.add_argument("--backend", default=None, help="Storage backend")
    parser.add_argument("--collection", default=None, help="Collection name")
    parser.add_argument("--top-k", type=int, default=None, help="Candidates to retrieve")
    parser.add_argument(
        "--scope",
        default=None,
        help="Comma-separated scopes to restrict retrieval to",
    )
    parser.add_argument(
        "--actions",
        default=None,
        help="Comma-separated actions that inject context "
             "(default: replay,restore,verify)",
    )
    parser.add_argument(
        "--gate-min-relevance",
        type=float,
        default=None,
        help="Relevance floor for gate mode "
             f"(default: {DEFAULT_GATE_MIN_RELEVANCE})",
    )
    parser.add_argument(
        "--max-prompt-chars", type=int, default=None,
        help="Skip prompts longer than this",
    )
    parser.add_argument(
        "--min-prompt-chars", type=int, default=None,
        help="Skip prompts shorter than this",
    )
    parser.add_argument(
        "--bypass-token", default=None,
        help="Prompt substring that disables the hook for one turn",
    )
    parser.add_argument(
        "--lexical",
        action="store_true",
        help="Disable embeddings — halves cold start, lowers paraphrase recall",
    )
    args = parser.parse_args(argv)

    scope = None
    if args.scope:
        scope = [s.strip() for s in args.scope.split(",") if s.strip()] or None
    actions = None
    if args.actions:
        actions = [a.strip().lower() for a in args.actions.split(",") if a.strip()]

    return HookConfig.from_env(
        mode=args.mode,
        data_dir=args.data_dir,
        backend=args.backend,
        collection=args.collection,
        top_k=args.top_k,
        scope=scope,
        actions=actions,
        gate_min_relevance=args.gate_min_relevance,
        max_prompt_chars=args.max_prompt_chars,
        min_prompt_chars=args.min_prompt_chars,
        bypass_token=args.bypass_token,
        semantic=False if args.lexical else None,
    )


def cmd_hook(args: argparse.Namespace) -> int:
    from agent_memory.hooks import main as hook_main

    return hook_main(args.hook_args)


def cmd_hook_install(args: argparse.Namespace) -> int:
    """Write the hook entry into a settings file, preserving existing hooks."""
    from agent_memory.hooks import HOOK_EVENT

    target = Path(args.settings) if args.settings else Path(".claude/settings.json")

    settings: dict = {}
    if target.exists():
        try:
            settings = json.loads(target.read_text() or "{}")
        except ValueError:
            print(f"✗ {target} is not valid JSON — fix or move it first.", file=sys.stderr)
            return 1
        if not isinstance(settings, dict):
            print(f"✗ {target} does not contain a JSON object.", file=sys.stderr)
            return 1

    command = f"agent-memory hook user-prompt-submit --data-dir {args.data_dir}"
    if args.mode == "gate":
        command += " --mode gate"
    if args.lexical:
        command += " --lexical"

    entry = {
        "type": "command",
        "command": command,
        "timeout": args.timeout,
        "statusMessage": "Checking agent memory…",
    }

    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        print(f"✗ 'hooks' in {target} is not an object.", file=sys.stderr)
        return 1
    matchers = hooks.setdefault(HOOK_EVENT, [])
    if not isinstance(matchers, list):
        print(f"✗ 'hooks.{HOOK_EVENT}' in {target} is not a list.", file=sys.stderr)
        return 1

    # Replace any previous agent-memory entry rather than stacking duplicates,
    # which would resolve the same prompt several times per turn.
    replaced = False
    for matcher in matchers:
        if not isinstance(matcher, dict):
            continue
        group = matcher.get("hooks")
        if not isinstance(group, list):
            continue
        for i, existing in enumerate(group):
            if (
                isinstance(existing, dict)
                and "agent-memory hook" in str(existing.get("command", ""))
            ):
                group[i] = entry
                replaced = True
    if not replaced:
        matchers.append({"hooks": [entry]})

    if args.dry_run:
        print(json.dumps(settings, indent=2))
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(settings, indent=2) + "\n")
    verb = "Updated" if replaced else "Installed"
    print(f"✓ {verb} {HOOK_EVENT} hook in {target}")
    print(f"  command: {command}")
    print(f"  mode:    {args.mode}")
    if args.mode == "gate":
        print("  ⚠  gate mode lets a high-confidence REPLAY answer without a model")
        print("     call. A wrong match takes the turn; resend with 'no-memory' to")
        print("     reach the model.")
    print("  Restart the session (or run /hooks) to pick it up.")
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

    # Flags are parsed by build_hook_config() rather than here: the hook is
    # invoked by the host, not typed by a user, and REMAINDER keeps the two
    # flag namespaces from colliding (both define --data-dir and --backend).
    hook = sub.add_parser(
        "hook",
        help="Handle a coding-agent lifecycle hook (reads JSON on stdin)",
    )
    hook.add_argument("hook_args", nargs=argparse.REMAINDER)
    hook.set_defaults(func=cmd_hook)

    hook_install = sub.add_parser(
        "hook-install",
        help="Write the UserPromptSubmit hook entry into a settings file",
    )
    hook_install.add_argument(
        "--settings",
        default=None,
        help="Settings file to edit (default: .claude/settings.json)",
    )
    hook_install.add_argument(
        "--mode",
        choices=["context", "gate"],
        default="context",
        help="context: inject memory (default). gate: REPLAY answers without a model call",
    )
    hook_install.add_argument(
        "--data-dir",
        dest="data_dir",
        default=".agent_memory",
        help="Persistence directory the hook should read (default: .agent_memory)",
    )
    hook_install.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Hook timeout in seconds (default: 10; the host default is 30)",
    )
    hook_install.add_argument(
        "--lexical",
        action="store_true",
        help="Run the hook without embeddings for a faster cold start",
    )
    hook_install.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resulting settings JSON instead of writing it",
    )
    hook_install.set_defaults(func=cmd_hook_install)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
