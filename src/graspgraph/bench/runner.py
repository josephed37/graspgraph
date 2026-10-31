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

from graspgraph.bench.faults import FaultInjector, choose_fault
from graspgraph.bench.pricing import Price
from graspgraph.bench.scoring import SUCCESSES, classify
from graspgraph.planner.backends import PlannerBackend, is_model_failure
from graspgraph.planner.closed_loop import DEFAULT_MAX_REPLANS, DEFAULT_MAX_STEPS, run_closed_loop
from graspgraph.planner.graph import run_open_loop
from graspgraph.planner.plans import PlanProtocol
from graspgraph.planner.prompts import CLOSED_LOOP_PROMPT_VERSION, PROMPT_VERSIONS
from graspgraph.tasks import Task, start_episode

CONFIG_FIELDS = (
    "backend", "model", "think", "temperature", "mode", "plan_protocol", "prompt_version", "fault_rate", "max_replans"
)
KEY_FIELDS = ("task_id", "repeat", *CONFIG_FIELDS)
# Records written before these fields existed all used the original one-call-per-step protocol,
# the provider's default sampling, open loop and no faults.
LEGACY_DEFAULTS = {"plan_protocol": "multi_call", "temperature": None, "fault_rate": 0.0, "max_replans": None}
MODES = ("open", "closed")
MAX_RESPONSE_CHARS = 2000  # keep a model's prose (useful for no_tool_calls) without bloating logs


def tracing_enabled() -> bool:
    """LangSmith tracing needs both LANGSMITH_TRACING=true and an API key."""
    return os.environ.get("LANGSMITH_TRACING", "").lower() == "true" and bool(os.environ.get("LANGSMITH_API_KEY"))


def run_key(record: dict[str, Any]) -> tuple:
    """Identity of one task run, used for resuming and de-duplicating."""
    return tuple(record.get(k, LEGACY_DEFAULTS.get(k)) for k in KEY_FIELDS)


@dataclass(frozen=True)
class RunSettings:
    """How tasks are run, beyond the model: planner mode, protocol, faults and loop budgets."""

    mode: str = "open"  # "open" or "closed"
    protocol: PlanProtocol = "submit_plan"
    fault_rate: float = 0.0  # share of act tasks that get an injected fault
    max_replans: int = DEFAULT_MAX_REPLANS  # closed loop only
    max_steps: int = DEFAULT_MAX_STEPS  # closed loop only

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown mode '{self.mode}'; choose from {MODES}")
        if self.mode == "closed" and self.protocol != "submit_plan":
            raise ValueError("the closed loop uses the submit_plan protocol only")

    @property
    def prompt_version(self) -> str:
        return CLOSED_LOOP_PROMPT_VERSION if self.mode == "closed" else PROMPT_VERSIONS[self.protocol]

    def config(self, backend: PlannerBackend) -> dict[str, Any]:
        """The CONFIG_FIELDS values for runs with this backend."""
        return {
            "backend": backend.backend,
            "model": backend.model,
            "think": backend.think,
            "temperature": backend.temperature,
            "mode": self.mode,
            "plan_protocol": self.protocol,
            "prompt_version": self.prompt_version,
            "fault_rate": self.fault_rate,
            "max_replans": self.max_replans if self.mode == "closed" else None,
        }


def run_task(
    task: Task, backend: PlannerBackend, price: Price, repeat: int = 0, settings: RunSettings = RunSettings()
) -> dict[str, Any]:
    """Plan and execute one task, score it from the final simulator state, return its record."""
    episode = start_episode(task)
    kind = choose_fault(task, repeat, settings.fault_rate)
    injector = FaultInjector.for_run(task, repeat, kind) if kind else None
    if settings.mode == "closed":
        run = run_closed_loop(backend, episode, injector, settings.max_replans, settings.max_steps)
    else:
        run = run_open_loop(backend, episode, settings.protocol, injector)
    final = episode.robot.get_scene()
    verdict = episode.verdict()
    outcome = classify(task, run, verdict, episode.start_state, final)
    notes = [n for n in (run.error, price.note(run.input_tokens)) if n]
    fault = injector.record() if injector else None
    return {
        "task_id": task.task_id,
        "family": task.family,
        "seed": task.seed,
        "repeat": repeat,
        **settings.config(backend),
        "success": outcome in SUCCESSES,
        "expected_outcome": task.expected_outcome,
        "outcome_type": outcome,
        "detail": verdict.detail,
        "tool_calls": [c.to_dict() for c in run.tool_calls],
        "planned_steps": run.plan.planned_steps if run.plan else 0,
        "raw_plan": run.plan.raw if run.plan else None,
        "plan_error": run.plan.error if run.plan else None,
        "llm_calls": run.llm_calls,
        "replans": run.replans,
        "plans": run.plans,
        "fault": fault,
        "recovered": bool(fault and fault["triggered"] and outcome in SUCCESSES),
        "primitive_results": run.results,
        "steps": sum(r["steps"] for r in run.results),
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
        "cost_usd": price.cost(run.input_tokens, run.output_tokens),
        "latency_s": round(run.latency_s, 3),
        "finish_reason": run.finish_reason,
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


def _retryable(record: dict[str, Any]) -> bool:
    """Only transient failures are re-run. Older files logged repetition loops as llm_error,
    so those are recognised by their error text."""
    return record.get("outcome_type") == "llm_error" and not is_model_failure(record.get("error") or "")


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
    settings: RunSettings = RunSettings(),
    log: Callable[[str], None] = print,
) -> BenchReport:
    """Run every (task, repeat) not yet recorded in `out`, appending one JSON line each.

    Runs that ended in an llm_error (network or API trouble) are retried on resume. Model
    failures such as degenerate_output are not, so a rerun gives no task extra attempts.
    Before each run the cap is checked against the money already spent plus the most
    expensive run so far, so the run stops before it would cross `max_usd`.
    """
    existing = read_records(out)
    done = {run_key(r) for r in existing if not _retryable(r)}
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
                config = settings.config(backend)
                key = (task.task_id, repeat, *(config[k] for k in CONFIG_FIELDS))
                if key in done:
                    skipped += 1
                    continue
                if spent + worst > max_usd:
                    log(f"spend cap reached: ${spent:.4f} spent, next run could cost up to ${worst:.4f}, cap ${max_usd:.2f}. Stopping.")
                    stopped = True
                    break
                record = run_task(task, backend, price, repeat, settings)
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
