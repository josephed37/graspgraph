"""graspgraph-matrix: plan, calibrate and run the whole benchmark matrix.

    uv run graspgraph-matrix plan experiments/main.toml
    uv run graspgraph-matrix calibrate --model claude-haiku-5-5 --tasks 10
    uv run graspgraph-matrix run experiments/main.toml --yes
    uv run graspgraph-matrix run experiments/main.toml --only gemma-main --skip sonnet-subset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from graspgraph.bench.matrix import (
    ask,
    calibrate,
    check_anthropic,
    check_ollama,
    load_matrix,
    plan,
    preflight,
    run_matrix,
)
from graspgraph.bench.runner import tracing_enabled
from graspgraph.planner.backends import make_backend


def _backend_for(model: str, backend: str | None) -> str:
    return backend or ("anthropic" if model.startswith("claude-") else "ollama")


def main(argv: list[str] | None = None) -> None:
    """Entry point for the graspgraph-matrix command."""
    parser = argparse.ArgumentParser(
        prog="graspgraph-matrix", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="print every run with estimated calls, cost and time")
    p_plan.add_argument("matrix", type=Path)
    p_plan.add_argument("--max-usd", type=float, help="global cap to compare against (default: the matrix's)")

    p_run = sub.add_parser("run", help="run the matrix, cheapest configs first, under a global spend cap")
    p_run.add_argument("matrix", type=Path)
    p_run.add_argument("--only", nargs="+", default=[], metavar="CONFIG", help="run only these configs")
    p_run.add_argument("--skip", nargs="+", default=[], metavar="CONFIG", help="leave these configs out")
    p_run.add_argument("--yes", action="store_true", help="run paid configs without asking")
    p_run.add_argument("--max-usd", type=float, help="global spend cap across every config (default: the matrix's, 8.00)")
    p_run.add_argument("--no-smoke", action="store_true", help="skip the 1-task smoke run per backend")
    p_run.add_argument("--ollama-url", help="Ollama server URL (default: http://localhost:11434)")

    p_cal = sub.add_parser("calibrate", help="measure a model's real token use on a few tasks")
    p_cal.add_argument("--model", required=True)
    p_cal.add_argument("--backend", choices=["ollama", "anthropic", "fake"], help="default: from the model name")
    p_cal.add_argument("--tasks", type=int, default=10, help="tasks per mode, spread over the families (default 10)")
    p_cal.add_argument("--max-usd", type=float, default=0.30, help="spend cap for the calibration (default 0.30)")
    p_cal.add_argument("--faults", type=float, default=0.3, help="fault rate, as in the matrix (default 0.3)")
    p_cal.add_argument("--results-dir", type=Path, default=Path("results"))
    p_cal.add_argument("--ollama-url")
    args = parser.parse_args(argv)

    try:
        if args.command == "plan":
            print(plan(load_matrix(args.matrix), max_usd=args.max_usd)[1])
        elif args.command == "calibrate":
            backend_name = _backend_for(args.model, args.backend)
            problem = {"anthropic": check_anthropic, "ollama": lambda: check_ollama(args.model, args.ollama_url)}.get(
                backend_name, lambda: None
            )()
            if problem:
                sys.exit(f"graspgraph-matrix: {problem}")
            backend = make_backend(backend_name, args.model, ollama_url=args.ollama_url)
            entry = calibrate(backend, args.results_dir, args.tasks, args.max_usd, faults=args.faults)
            for mode, m in entry["modes"].items():
                print(
                    f"{args.model} [{mode}]: {m['runs']} runs, {m['input_tokens_per_run']:.0f} in / "
                    f"{m['output_tokens_per_run']:.0f} out tokens, {m['llm_calls_per_run']:.2f} calls, "
                    f"${m['cost_usd_per_run']:.4f} per run"
                )
        else:
            matrix = load_matrix(args.matrix)
            print(plan(matrix, max_usd=args.max_usd)[1] + "\n")
            print(f"LangSmith tracing {'on' if tracing_enabled() else 'off'}")
            report = run_matrix(
                matrix,
                only=args.only,
                skip=args.skip,
                yes=args.yes,
                max_usd=args.max_usd,
                confirm=ask,
                smoke=not args.no_smoke,
                backend_factory=lambda c: c.make_backend(args.ollama_url),
                checks=lambda c: preflight(c, matrix.results_dir, args.ollama_url),
            )
            print(f"\ndone: {sum(report.ran.values())} task runs; ${report.spent_usd:.4f} recorded in total")
            for name, why in report.skipped.items():
                print(f"  skipped {name}: {why}")
            if report.stopped_by_cap:
                print("  stopped at the global spend cap")
    except ValueError as exc:
        sys.exit(f"graspgraph-matrix: {exc}")


if __name__ == "__main__":
    main()
