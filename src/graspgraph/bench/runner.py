"""Run tasks through a planner backend and write one JSON line per task run.

Resumable (runs already in the output file are skipped) and capped (the run stops cleanly
before the spend would exceed --max-usd). LangSmith tracing is on only when both
LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set; otherwise nothing leaves the machine.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import langsmith

from graspgraph.bench.pricing import Price
from graspgraph.bench.scoring import SUCCESSES, classify
from graspgraph.planner.backends import PlannerBackend
from graspgraph.planner.graph import run_open_loop
from graspgraph.planner.plans import PlanProtocol
from graspgraph.planner.prompts import PROMPT_VERSIONS
from graspgraph.tasks import Task, start_episode

KEY_FIELDS = ("task_id", "repeat", "backend", "model", "think", "mode", "plan_protocol")
# Records written before plan_protocol existed all used the original one-call-per-step protocol.
LEGACY_DEFAULTS = {"plan_protocol": "multi_call"}
MAX_RESPONSE_CHARS = 2000  # keep a model's prose (useful for no_tool_calls) without bloating logs


def tracing_enabled() -> bool:
    """LangSmith tracing needs both LANGSMITH_TRACING=true and an API key."""
    return os.environ.get("LANGSMITH_TRACING", "").lower() == "true" and bool(os.environ.get("LANGSMITH_API_KEY"))


def run_key(record: dict[str, Any]) -> tuple:
    """Identity of one task run, used for resuming and de-duplicating."""
    return tuple(record.get(k, LEGACY_DEFAULTS.get(k)) for k in KEY_FIELDS)


def run_task(
    task: Task,
    backend: PlannerBackend,
    price: Price,
    repeat: int = 0,
    mode: str = "open",
    protocol: PlanProtocol = "submit_plan",
) -> dict[str, Any]:
    """Plan and execute one task, score it from the final simulator state, return its record."""
    episode = start_episode(task)
    run = run_open_loop(backend, episode, protocol)
    final = episode.robot.get_scene()
    verdict = episode.verdict()
    outcome = classify(task, run, verdict, episode.start_state, final)
    notes = [n for n in (run.error, price.note(run.input_tokens)) if n]
    return {
        "task_id": task.task_id,
        "family": task.family,
        "seed": task.seed,
        "backend": backend.backend,
        "model": backend.model,
        "think": backend.think,
        "mode": mode,
        "repeat": repeat,
        "plan_protocol": protocol,
        "prompt_version": PROMPT_VERSIONS[protocol],
        "success": outcome in SUCCESSES,
        "expected_outcome": task.expected_outcome,
        "outcome_type": outcome,
        "detail": verdict.detail,
        "tool_calls": [c.to_dict() for c in run.tool_calls],
        "planned_steps": run.plan.planned_steps if run.plan else 0,
        "raw_plan": run.plan.raw if run.plan else None,
        "plan_error": run.plan.error if run.plan else None,
        "primitive_results": run.results,
        "steps": sum(r["steps"] for r in run.results),
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
        "cost_usd": price.cost(run.input_tokens, run.output_tokens),
        "latency_s": round(run.latency_s, 3),
        "error": "; ".join(notes) or None,
        "response_text": run.response_text[:MAX_RESPONSE_CHARS],
    }


def read_records(path: Path) -> list[dict[str, Any]]:
    """Records in a results file; a line cut short by a crash is ignored."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


@dataclass
class BenchReport:
    ran: int
    skipped: int  # already in the output file
    spent_usd: float  # total in the file, including earlier sessions
    stopped_by_cap: bool


def run_benchmark(
    tasks: list[Task],
    backend: PlannerBackend,
    out: Path,
    price: Price,
    repeats: int = 1,
    max_usd: float = 1.00,
    mode: str = "open",
    protocol: PlanProtocol = "submit_plan",
    log: Callable[[str], None] = print,
) -> BenchReport:
    """Run every (task, repeat) not yet recorded in `out`, appending one JSON line each.

    Runs that ended in an llm_error are retried on resume. Before each run the cap is
    checked against the money already spent plus the most expensive run so far, so the
    run stops before it would cross `max_usd`.
    """
    existing = read_records(out)
    done = {run_key(r) for r in existing if r.get("outcome_type") != "llm_error"}
    spent = sum(r.get("cost_usd") or 0.0 for r in existing)
    worst = max((r.get("cost_usd") or 0.0 for r in existing), default=0.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.read_bytes()[-1:] not in (b"", b"\n"):
        with out.open("a") as f:
            f.write("\n")  # finish a line a crash cut short, so new records start cleanly

    ran = skipped = 0
    stopped = False
    with langsmith.tracing_context(enabled=tracing_enabled()), out.open("a") as f:
        for repeat in range(repeats):
            for task in tasks:
                key = (task.task_id, repeat, backend.backend, backend.model, backend.think, mode, protocol)
                if key in done:
                    skipped += 1
                    continue
                if spent + worst > max_usd:
                    log(f"spend cap reached: ${spent:.4f} spent, next run could cost up to ${worst:.4f}, cap ${max_usd:.2f}. Stopping.")
                    stopped = True
                    break
                record = run_task(task, backend, price, repeat, mode, protocol)
                f.write(json.dumps(record) + "\n")
                f.flush()
                spent += record["cost_usd"]
                worst = max(worst, record["cost_usd"])
                ran += 1
                log(
                    f"{task.task_id:16s} r{repeat} {'ok  ' if record['success'] else 'FAIL'} "
                    f"{record['outcome_type']:22s} ${record['cost_usd']:.4f} {record['latency_s']:.1f}s"
                )
            if stopped:
                break
    return BenchReport(ran=ran, skipped=skipped, spent_usd=spent, stopped_by_cap=stopped)
