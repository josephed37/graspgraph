"""The experiment matrix: several benchmark configs, planned, costed and run under one budget.

A matrix file (TOML, see experiments/main.toml) lists configs: backend, model, modes,
faults, repeats, task set and output file. This module
- estimates every config's model calls, tokens, cost and time, from results/calibration.json
  when it exists and from conservative defaults otherwise;
- runs the configs cheapest first through the ordinary benchmark runner (one JSONL per
  config, resumable), under a GLOBAL spend cap computed from the money actually recorded in
  every result file, so a resumed run never double-counts or overspends;
- checks the environment first (Ollama, API key, Menagerie, results directory) and makes a
  1-task smoke run per backend before a config's full run;
- calibrates a model's real token use on a small stratified task set.

Open and closed runs of a config share task, repeat and fault seed, so they get the same
faults and can be compared pair by pair.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graspgraph.bench.pricing import Price, PriceTable, load_prices, price_for
from graspgraph.bench.runner import RunSettings, read_records, run_benchmark
from graspgraph.planner.backends import PlannerBackend, make_backend
from graspgraph.sim.scene import menagerie_dir
from graspgraph.tasks import FAMILY_NAMES, build_suite, get_task

SMOKE_TASK = "pick-place-000"
CALIBRATION_FILE = "calibration.json"

# Used when a model has no calibration yet: deliberately on the high side.
DEFAULT_INPUT_TOKENS_PER_CALL = 2500  # Gemma's real runs used about 1,600
DEFAULT_OUTPUT_TOKENS_PER_CALL = {"plain": 500, "thinking": 1500}
DEFAULT_CALLS_PER_RUN = {"open": 1.0, "closed": 1.5}  # closed: replans after mismatches
DEFAULT_SECONDS_PER_CALL = {"ollama": 12.0, "anthropic": 5.0, "fake": 0.0}  # Gemma no-think measured ~10 s
THINKING_SLOWDOWN = 3.0  # a thinking model writes far more tokens per call
SIM_SECONDS_PER_RUN = 1.5  # physics, scene building and scoring


@functools.cache
def all_task_ids() -> tuple[str, ...]:
    """Every task id in the suite (building the suite takes seconds, so this is cached)."""
    return tuple(t.task_id for t in build_suite())


@dataclass(frozen=True)
class MatrixConfig:
    """One row of the experiment table."""

    name: str
    backend: str
    model: str
    modes: tuple[str, ...]
    tasks: str  # "all", or a file of task ids (one per line), relative to the matrix file
    out: Path
    faults: float = 0.0
    fault_seed: int = 0
    repeats: int = 1
    think: bool = False  # Ollama thinking
    effort: str | None = None  # Anthropic reasoning effort (default: per model)
    thinking: str | None = None  # Anthropic thinking: "disabled", "adaptive" or "omit" (default: per model)
    max_tokens: int | None = None
    base_dir: Path = Path(".")

    @property
    def thinks(self) -> bool:
        """Whether this config's model reasons before answering (more output tokens)."""
        if self.backend == "anthropic":
            return (self.thinking or "") == "adaptive" or self.model.startswith("claude-sonnet-5-5")
        return self.think

    def task_ids(self) -> list[str]:
        if self.tasks == "all":
            return list(all_task_ids())
        return (self.base_dir / self.tasks).read_text().split()

    def settings(self, mode: str) -> RunSettings:
        return RunSettings(mode=mode, fault_rate=self.faults, fault_seed=self.fault_seed)

    def make_backend(self, ollama_url: str | None = None) -> PlannerBackend:
        return make_backend(
            self.backend, self.model, think=self.think, max_tokens=self.max_tokens, ollama_url=ollama_url,
            effort=self.effort, thinking=self.thinking,
        )


@dataclass(frozen=True)
class Matrix:
    path: Path
    configs: tuple[MatrixConfig, ...]
    results_dir: Path
    max_usd: float


def load_matrix(path: Path) -> Matrix:
    """Read a matrix TOML file; raises ValueError with the offending config on a bad entry."""
    raw = tomllib.loads(path.read_text())
    results_dir = Path(raw.get("results_dir", "results"))
    configs = []
    for entry in raw.get("config", []):
        name = entry.get("name", "?")
        try:
            configs.append(
                MatrixConfig(
                    name=name,
                    backend=entry["backend"],
                    model=entry["model"],
                    modes=tuple(entry.get("modes", ["open"])),
                    tasks=entry.get("tasks", "all"),
                    out=Path(entry.get("out", results_dir / f"{name}.jsonl")),
                    faults=float(entry.get("faults", 0.0)),
                    fault_seed=int(entry.get("fault_seed", 0)),
                    repeats=int(entry.get("repeats", 1)),
                    think=bool(entry.get("think", False)),
                    effort=entry.get("effort"),
                    thinking=entry.get("thinking"),
                    max_tokens=entry.get("max_tokens"),
                    base_dir=path.parent,
                )
            )
        except KeyError as exc:
            raise ValueError(f"{path}: config '{name}' is missing {exc}") from None
    names = [c.name for c in configs]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: config names must be unique")
    return Matrix(path, tuple(configs), results_dir, float(raw.get("max_usd", 8.0)))


# ------------------------------------------------------------------ estimates


@dataclass(frozen=True)
class Estimate:
    config: str
    mode: str
    runs: int
    llm_calls: float
    input_tokens: float
    output_tokens: float
    cost_usd: float
    seconds: float
    basis: str  # "calibrated" or "default estimate"


def load_calibration(results_dir: Path) -> dict[str, Any]:
    path = results_dir / CALIBRATION_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("models", {})


def estimate(config: MatrixConfig, mode: str, price: Price, calibration: dict[str, Any]) -> Estimate:
    """Expected calls, tokens, cost and time for one config in one mode."""
    runs = len(config.task_ids()) * config.repeats
    measured = calibration.get(config.model, {}).get("modes", {}).get(mode)
    if measured:
        calls = runs * measured["llm_calls_per_run"]
        tokens_in = runs * measured["input_tokens_per_run"]
        tokens_out = runs * measured["output_tokens_per_run"]
        seconds = runs * (measured["latency_s_per_run"] + SIM_SECONDS_PER_RUN)
        basis = "calibrated"
    else:
        calls = runs * DEFAULT_CALLS_PER_RUN[mode]
        tokens_in = calls * DEFAULT_INPUT_TOKENS_PER_CALL
        tokens_out = calls * DEFAULT_OUTPUT_TOKENS_PER_CALL["thinking" if config.thinks else "plain"]
        per_call = DEFAULT_SECONDS_PER_CALL.get(config.backend, 10.0) * (THINKING_SLOWDOWN if config.thinks else 1.0)
        seconds = calls * per_call + runs * SIM_SECONDS_PER_RUN
        basis = "default estimate"
    cost = (tokens_in * price.input + tokens_out * price.output) / 1_000_000
    return Estimate(config.name, mode, runs, calls, tokens_in, tokens_out, cost, seconds, basis)


def _duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    return f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def spent_usd(results_dir: Path) -> float:
    """Money already recorded in every result file under `results_dir` (configs, smoke runs,
    calibration). Each line is one real run, so this is what was actually spent."""
    if not results_dir.exists():
        return 0.0
    return sum(r.get("cost_usd") or 0.0 for path in results_dir.rglob("*.jsonl") for r in read_records(path))


def plan(matrix: Matrix, prices: PriceTable | None = None, max_usd: float | None = None) -> tuple[list[Estimate], str]:
    """Every config and mode with its estimate, as data and as a printable table."""
    prices = prices or load_prices()
    calibration = load_calibration(matrix.results_dir)
    estimates = [
        estimate(c, mode, price_for(prices, c.backend, c.model), calibration)
        for c in ordered(matrix.configs, prices, calibration)
        for mode in c.modes
    ]
    cap = matrix.max_usd if max_usd is None else max_usd
    by_name = {c.name: c for c in matrix.configs}
    lines = [
        f"Plan for {matrix.path} (global cap ${cap:.2f}, results in {matrix.results_dir}/)",
        "All costs, token counts and times below are ESTIMATES.",
        "",
        f"{'config':16s} {'backend/model':34s} {'mode':6s} {'tasks':>5s} {'reps':>4s} {'runs':>5s} "
        f"{'calls':>6s} {'tokens in':>10s} {'tokens out':>10s} {'cost':>8s} {'time':>7s}  basis",
    ]
    for e in estimates:
        c = by_name[e.config]
        lines.append(
            f"{c.name:16s} {c.backend + '/' + c.model + (' +think' if c.thinks else ''):34s} {e.mode:6s} "
            f"{len(c.task_ids()):5d} {c.repeats:4d} {e.runs:5d} {e.llm_calls:6.0f} {e.input_tokens:10,.0f} "
            f"{e.output_tokens:10,.0f} ${e.cost_usd:7.2f} {_duration(e.seconds):>7s}  {e.basis}"
        )
    total_cost = sum(e.cost_usd for e in estimates)
    lines += [
        "",
        f"total: {sum(e.runs for e in estimates)} runs, {sum(e.llm_calls for e in estimates):,.0f} model calls, "
        f"estimated ${total_cost:.2f}, estimated {_duration(sum(e.seconds for e in estimates))}",
        f"already spent (recorded in {matrix.results_dir}/): ${spent_usd(matrix.results_dir):.4f}",
    ]
    if total_cost > cap:
        lines.append(f"WARNING: the estimate exceeds the ${cap:.2f} cap; the run will stop when the cap is reached.")
    if any(e.basis != "calibrated" for e in estimates if e.cost_usd > 0):
        lines.append("Paid configs without calibration use conservative defaults: run `graspgraph-matrix calibrate` first.")
    return estimates, "\n".join(lines)


def ordered(
    configs: tuple[MatrixConfig, ...], prices: PriceTable, calibration: dict[str, Any] | None = None
) -> list[MatrixConfig]:
    """Configs in run order: cheapest first (free ones first), keeping file order on ties."""
    calibration = calibration or {}

    def cost(c: MatrixConfig) -> float:
        price = price_for(prices, c.backend, c.model)
        return sum(estimate(c, m, price, calibration).cost_usd for m in c.modes)

    return sorted(configs, key=cost)


# ------------------------------------------------------------------ preflight


def check_ollama(model: str, url: str | None = None) -> str | None:
    """None if Ollama answers at `url` and has `model` pulled, else what to do about it."""
    import ollama

    try:
        listed = ollama.Client(host=url).list()
    except Exception as exc:  # connection refused, timeouts, bad URL
        return f"Ollama is not reachable at {url or 'http://localhost:11434'} ({type(exc).__name__}); start it with `ollama serve`"
    names = {m.model for m in listed.models}
    if model not in names and f"{model}:latest" not in names:
        return f"Ollama does not have {model}; run `ollama pull {model}`"
    return None


def check_anthropic() -> str | None:
    """None if ANTHROPIC_API_KEY is set (its value is never shown)."""
    return None if os.environ.get("ANTHROPIC_API_KEY") else "ANTHROPIC_API_KEY is not set; `export ANTHROPIC_API_KEY=...` first"


def check_menagerie() -> str | None:
    scene = menagerie_dir() / "franka_emika_panda" / "scene.xml"
    return None if scene.exists() else f"Menagerie not found at {scene.parent}; clone it into third_party/ (see README)"


def check_writable(directory: Path) -> str | None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".write-test"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        return f"cannot write to {directory}: {exc}"
    return None


def preflight(config: MatrixConfig, results_dir: Path, ollama_url: str | None = None) -> list[str]:
    """Every problem that would stop this config from running."""
    checks = [check_menagerie(), check_writable(results_dir), check_writable(config.out.parent)]
    if config.backend == "ollama":
        checks.append(check_ollama(config.model, ollama_url))
    elif config.backend == "anthropic":
        checks.append(check_anthropic())
    return [c for c in checks if c]


# ------------------------------------------------------------------------ run


@dataclass
class MatrixReport:
    ran: dict[str, int] = field(default_factory=dict)  # config -> task runs made now
    skipped: dict[str, str] = field(default_factory=dict)  # config -> why it did not run
    spent_usd: float = 0.0
    stopped_by_cap: bool = False


BackendFactory = Callable[[MatrixConfig], PlannerBackend]


def run_matrix(
    matrix: Matrix,
    only: list[str] | None = None,
    skip: list[str] | None = None,
    yes: bool = False,
    max_usd: float | None = None,
    prices: PriceTable | None = None,
    confirm: Callable[[str], bool] | None = None,
    log: Callable[[str], None] = print,
    backend_factory: BackendFactory | None = None,
    checks: Callable[[MatrixConfig], list[str]] | None = None,
    smoke: bool = True,
) -> MatrixReport:
    """Run the selected configs, cheapest first, under one global spend cap.

    Paid configs need `yes` or a True answer from `confirm` (asked with their estimated cost
    and the remaining budget). A config whose preflight checks or smoke run fail is skipped,
    and the run moves on. Hitting the global cap stops the whole matrix.
    """
    prices = prices or load_prices()
    cap = matrix.max_usd if max_usd is None else max_usd
    backend_factory = backend_factory or (lambda c: c.make_backend())
    checks = checks or (lambda c: preflight(c, matrix.results_dir))
    calibration = load_calibration(matrix.results_dir)
    names = {c.name for c in matrix.configs}
    unknown = sorted((set(only or []) | set(skip or [])) - names)
    if unknown:
        raise ValueError(f"unknown configs {unknown}; the matrix has {sorted(names)}")
    selected = [c for c in ordered(matrix.configs, prices, calibration) if (not only or c.name in only) and c.name not in (skip or [])]
    report = MatrixReport()
    smoked: set[tuple[str, str]] = set()

    for config in selected:
        price = price_for(prices, config.backend, config.model)
        spent = spent_usd(matrix.results_dir)
        if spent >= cap:
            log(f"global spend cap ${cap:.2f} reached (${spent:.4f} recorded); stopping before {config.name}")
            report.stopped_by_cap = True
            break
        paid = price.input > 0 or price.output > 0
        if paid:
            est = sum(estimate(config, m, price, calibration).cost_usd for m in config.modes)
            question = (
                f"{config.name} ({config.backend}/{config.model}) is estimated at ${est:.2f}; "
                f"${cap - spent:.2f} of the ${cap:.2f} budget remains."
            )
            log(question)
            if not (yes or (confirm and confirm(question + " Continue? [y/N] "))):
                report.skipped[config.name] = "paid config not confirmed (pass --yes)"
                log(f"skipping {config.name}: not confirmed (pass --yes to run paid configs)")
                continue
        problems = checks(config)
        if problems:
            report.skipped[config.name] = "; ".join(problems)
            log(f"skipping {config.name}: " + "; ".join(problems))
            continue
        backend = backend_factory(config)
        if smoke and (config.backend, config.model) not in smoked:
            problem = _smoke(config, backend, price, matrix.results_dir, cap, log)
            if problem:
                report.skipped[config.name] = problem
                continue
            smoked.add((config.backend, config.model))

        tasks = [get_task(t) for t in config.task_ids()]
        report.ran[config.name] = 0
        for mode in config.modes:
            spent_here = sum(r.get("cost_usd") or 0.0 for r in read_records(config.out))
            file_cap = cap - (spent_usd(matrix.results_dir) - spent_here)  # this file may use what is left
            log(f"== {config.name} [{mode}]: {len(tasks)} tasks x {config.repeats} repeats -> {config.out}")
            result = run_benchmark(
                tasks, backend, config.out, price, config.repeats, file_cap, config.settings(mode), log=log
            )
            report.ran[config.name] += result.ran
            if result.stopped_by_cap:
                report.stopped_by_cap = True
                break
        if report.stopped_by_cap:
            log(f"global spend cap ${cap:.2f} reached during {config.name}; stopping the matrix")
            break
    report.spent_usd = spent_usd(matrix.results_dir)
    return report


def _smoke(
    config: MatrixConfig, backend: PlannerBackend, price: Price, results_dir: Path, cap: float, log: Callable[[str], None]
) -> str | None:
    """One task through the backend before the full config; None if the backend answered."""
    out = results_dir / "smoke" / f"{config.name}.jsonl"
    log(f"smoke run for {config.backend}/{config.model}: {SMOKE_TASK}")
    file_cap = cap - (spent_usd(results_dir) - sum(r.get("cost_usd") or 0.0 for r in read_records(out)))
    run_benchmark([get_task(SMOKE_TASK)], backend, out, price, 1, file_cap, RunSettings(), log=log)
    record = (read_records(out) or [{}])[-1]
    if record.get("outcome_type") == "llm_error":
        problem = f"smoke run failed: {record.get('error')}"
        log(f"skipping {config.name}: {problem}")
        return problem
    return None


# ---------------------------------------------------------------- calibration


def calibration_tasks(n: int) -> list[str]:
    """`n` task ids spread over the families: seed 0 of every family, then seed 1, and so on."""
    def rank(task_id: str) -> tuple[int, int]:
        family, _, seed = task_id.rpartition("-")
        return int(seed), FAMILY_NAMES.index(family.replace("-", "_"))

    return sorted(all_task_ids(), key=rank)[:n]


def calibrate(
    backend: PlannerBackend,
    results_dir: Path,
    n_tasks: int = 10,
    max_usd: float = 0.30,
    modes: tuple[str, ...] = ("open", "closed"),
    faults: float = 0.3,
    fault_seed: int = 0,
    prices: PriceTable | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Measure a model's real token use per run and mode, and save it to results/calibration.json.

    Runs go to results/calibration/<model>.jsonl (resumable, and counted by the matrix's global
    cap). Returns this model's calibration entry.
    """
    prices = prices or load_prices()
    price = price_for(prices, backend.backend, backend.model)
    tasks = [get_task(t) for t in calibration_tasks(n_tasks)]
    slug = backend.model.replace(":", "-").replace("/", "-")
    out = results_dir / "calibration" / f"{slug}.jsonl"
    for mode in modes:
        log(f"calibrating {backend.backend}/{backend.model} [{mode}] on {len(tasks)} tasks (cap ${max_usd:.2f})")
        run_benchmark(tasks, backend, out, price, 1, max_usd, RunSettings(mode=mode, fault_rate=faults, fault_seed=fault_seed), log=log)

    ids = {t.task_id for t in tasks}
    records = [r for r in read_records(out) if r["model"] == backend.model and r["task_id"] in ids]
    entry: dict[str, Any] = {
        "backend": backend.backend,
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modes": {},
        "tasks": [
            {k: r[k] for k in ("task_id", "mode", "input_tokens", "output_tokens", "llm_calls", "latency_s", "cost_usd")}
            for r in records
        ],
    }
    errors = [r for r in records if r["outcome_type"] == "llm_error"]
    if errors:  # failed calls used no tokens: averaging them in would understate the cost
        log(f"WARNING: {len(errors)} calibration runs failed ({errors[-1].get('error')}); they are left out")
    for mode in modes:
        rs = [r for r in records if r["mode"] == mode and r["outcome_type"] != "llm_error"]
        if not rs:
            continue
        entry["modes"][mode] = {
            "runs": len(rs),
            "input_tokens_per_run": sum(r["input_tokens"] for r in rs) / len(rs),
            "output_tokens_per_run": sum(r["output_tokens"] for r in rs) / len(rs),
            "llm_calls_per_run": sum(r.get("llm_calls", 1) for r in rs) / len(rs),
            "latency_s_per_run": sum(r["latency_s"] for r in rs) / len(rs),
            "cost_usd_per_run": sum(r["cost_usd"] for r in rs) / len(rs),
        }
    path = results_dir / CALIBRATION_FILE
    existing = json.loads(path.read_text()) if path.exists() else {"models": {}}
    existing["models"][backend.model] = entry
    results_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=1) + "\n")
    log(f"saved {backend.model} calibration to {path}")
    return entry


def ask(question: str) -> bool:
    """Interactive yes/no on a terminal; anything but y/yes is no, and so is a non-interactive stdin."""
    if not sys.stdin.isatty():
        return False
    return input(question).strip().lower() in ("y", "yes")
