"""graspgraph-bench: run the task suite through a planner backend.

    uv run graspgraph-bench --backend ollama --model gemma4:12b --mode open --families all \\
        --out results/gemma4-12b_open.jsonl
    uv run graspgraph-bench --backend anthropic --model claude-haiku-5-5 --max-usd 0.50
    uv run graspgraph-bench --backend fake --model oracle          # no network: scripted plans
    uv run graspgraph-bench --backend ollama --plan-protocol multi_call   # the original protocol, for comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from graspgraph.bench.pricing import load_prices, price_for
from graspgraph.bench.runner import run_benchmark, tracing_enabled
from graspgraph.planner.backends import DEFAULT_MAX_TOKENS, DEFAULT_MODELS, FAKE_POLICIES, make_backend
from graspgraph.planner.plans import PLAN_PROTOCOLS
from graspgraph.tasks import FAMILY_NAMES, build_suite, get_task


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="graspgraph-bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backend", choices=list(DEFAULT_MODELS), required=True)
    parser.add_argument(
        "--model",
        help=f"model name; defaults: {DEFAULT_MODELS}. Fake policies: {', '.join(FAKE_POLICIES)}",
    )
    parser.add_argument("--mode", choices=["open"], default="open", help="planner mode (closed loop comes in week 3)")
    parser.add_argument(
        "--plan-protocol",
        choices=PLAN_PROTOCOLS,
        default="submit_plan",
        help="submit_plan: one call holding every step (default); multi_call: one tool call per step (original)",
    )
    parser.add_argument("--families", nargs="+", default=["all"], help=f"'all' or any of: {', '.join(FAMILY_NAMES)}")
    parser.add_argument("--task-ids", nargs="+", help="run only these task ids (overrides --families)")
    parser.add_argument("--limit", type=int, help="run only the first N selected tasks")
    parser.add_argument("--repeats", type=int, default=1, help="run every task N times")
    parser.add_argument(
        "--think", action=argparse.BooleanOptionalAction, default=False, help="Ollama thinking mode (default: off)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help=f"max output tokens (Ollama: num_predict, thinking included); defaults: {DEFAULT_MAX_TOKENS}",
    )
    parser.add_argument("--temperature", type=float, help="sampling temperature (default: the provider's own)")
    parser.add_argument("--max-usd", type=float, default=1.00, help="stop before spend would exceed this (default 1.00)")
    parser.add_argument("--ollama-url", help="Ollama server URL (default: http://localhost:11434)")
    parser.add_argument("--out", type=Path, help="JSONL output; default results/<backend>-<model>_<mode>_<protocol>.jsonl")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the graspgraph-bench command."""
    args = parse_args(argv)
    model = args.model or DEFAULT_MODELS[args.backend]
    try:
        if args.task_ids:
            tasks = [get_task(t) for t in args.task_ids]
        else:
            tasks = build_suite(None if args.families == ["all"] else args.families)
        price = price_for(load_prices(), args.backend, model)
        backend = make_backend(
            args.backend, model, think=args.think, max_tokens=args.max_tokens, ollama_url=args.ollama_url,
            temperature=args.temperature,
        )
    except ValueError as exc:
        sys.exit(f"graspgraph-bench: {exc}")
    if args.limit is not None:
        tasks = tasks[: args.limit]
    slug = model.replace(":", "-").replace("/", "-")
    out = args.out or Path("results") / f"{args.backend}-{slug}_{args.mode}_{args.plan_protocol}.jsonl"

    print(
        f"{len(tasks)} tasks x {args.repeats} repeats with {args.backend}/{model} (think={backend.think}, "
        f"temperature={backend.temperature if backend.temperature is not None else 'default'}, protocol={args.plan_protocol}), "
        f"cap ${args.max_usd:.2f}, LangSmith tracing {'on' if tracing_enabled() else 'off'} -> {out}"
    )
    report = run_benchmark(
        tasks, backend, out, price, repeats=args.repeats, max_usd=args.max_usd, mode=args.mode, protocol=args.plan_protocol
    )
    print(
        f"done: {report.ran} run, {report.skipped} already recorded, ${report.spent_usd:.4f} spent in total"
        + (" (stopped by the spend cap)" if report.stopped_by_cap else "")
    )


if __name__ == "__main__":
    main()
