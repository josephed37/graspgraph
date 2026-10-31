"""Turn benchmark result files into a markdown report, a failure catalogue and charts.

Reads every config file (`*.jsonl`) directly in a results directory (smoke runs and
calibration live in subdirectories and are left out) and writes:
- results.md: headline table with 95% Wilson CIs, per-family success, paired comparisons
  (bootstrap CI and McNemar), outcome types, and how to reproduce;
- failures.md: up to three real examples per failure outcome type;
- success_by_family.png, open_vs_closed.png, cost_vs_success.png.

Works with partial results and open-only results. Every file names the backends its data
came from, and anything built from the fake backend says "FAKE DATA - not real results".
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from graspgraph.bench.scoring import SUCCESSES
from graspgraph.bench.stats import paired_comparison, pairs, wilson
from graspgraph.bench.summary import latest_records
from graspgraph.tasks import FAMILY_NAMES, get_task

FAKE_BANNER = "FAKE DATA - not real results"
MAX_FAILURE_EXAMPLES = 3
# Okabe-Ito, colour-blind friendly. A model keeps its colour in every chart.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
KNOWN_MODEL_COLOURS = {"gemma4:12b": PALETTE[0], "claude-haiku-5-5": PALETTE[1], "claude-sonnet-5-5": PALETTE[2]}

Record = dict[str, Any]


@dataclass
class Group:
    """The runs of one config in one mode."""

    config: str  # the result file's stem, which the matrix names after the config
    mode: str
    records: list[Record]

    @property
    def label(self) -> str:
        return f"{self.config} [{self.mode}]"

    @property
    def backend(self) -> str:
        return self.records[0]["backend"]

    @property
    def model(self) -> str:
        return self.records[0]["model"]

    @property
    def think(self) -> bool:
        return bool(self.records[0].get("think"))

    @property
    def successes(self) -> int:
        return sum(bool(r["success"]) for r in self.records)

    @property
    def faulted(self) -> list[Record]:
        return [r for r in self.records if r.get("fault_injected")]


def load_groups(results_dir: Path) -> list[Group]:
    """One group per (config file, mode), in file then mode order."""
    groups = []
    for path in sorted(results_dir.glob("*.jsonl")):
        by_mode: dict[str, list[Record]] = defaultdict(list)
        for r in latest_records(path):
            by_mode[r.get("mode", "open")].append(r)
        groups += [Group(path.stem, mode, rs) for mode, rs in sorted(by_mode.items(), key=lambda kv: kv[0] != "open")]
    return groups


def is_fake(groups: list[Group]) -> bool:
    return any(r["backend"] == "fake" for g in groups for r in g.records)


def model_colours(groups: list[Group]) -> dict[str, str]:
    colours = dict(KNOWN_MODEL_COLOURS)
    spare = [c for c in PALETTE if c not in colours.values()]
    for g in groups:
        if g.model not in colours:
            colours[g.model] = spare.pop(0) if spare else "#777777"
    return colours


def _pct(k: int, n: int) -> str:
    if not n:
        return "-"
    lo, hi = wilson(k, n)
    return f"{k / n:.1%} [{lo:.1%}, {hi:.1%}]"


def _sources(groups: list[Group]) -> str:
    models = sorted({f"{g.backend}/{g.model}" for g in groups})
    return "Data from: " + (", ".join(models) if models else "no results")


# ------------------------------------------------------------------ results.md


def headline_table(groups: list[Group]) -> list[str]:
    rows = [
        "| config | mode | backend/model | runs | success [95% CI] | faulted | recovery [95% CI] | replans | "
        "LLM calls | latency (s) | cost | cost per success |",
        "|---|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for g in groups:
        n, k = len(g.records), g.successes
        faulted = g.faulted
        cost = sum(r.get("cost_usd") or 0.0 for r in g.records)
        per_success = f"${cost / k:.4f}" if k else "-"
        rows.append(
            f"| {g.config} | {g.mode} | {g.backend}/{g.model}{' +think' if g.think else ''} | {n} | {_pct(k, n)} | "
            f"{len(faulted)} | {_pct(sum(bool(r['success']) for r in faulted), len(faulted))} | "
            f"{sum(r.get('replans', 0) for r in g.records) / n:.2f} | "
            f"{sum(r.get('llm_calls', 1) for r in g.records) / n:.2f} | "
            f"{sum(r.get('latency_s', 0.0) for r in g.records) / n:.2f} | ${cost:.4f} | {per_success} |"
        )
    return rows


def family_table(groups: list[Group]) -> list[str]:
    rows = ["| family | " + " | ".join(g.label for g in groups) + " |", "|---|" + "---:|" * len(groups)]
    for family in FAMILY_NAMES:
        cells = []
        for g in groups:
            rs = [r for r in g.records if r["family"] == family]
            k = sum(bool(r["success"]) for r in rs)
            cells.append(f"{k}/{len(rs)} ({k / len(rs):.0%})" if rs else "-")
        rows.append(f"| {family} | " + " | ".join(cells) + " |")
    return rows


def comparisons(groups: list[Group]) -> list[tuple[str, str, Group, Group]]:
    """(kind, description, A, B) for every paired comparison the data supports."""
    found = []
    by_config: dict[str, dict[str, Group]] = defaultdict(dict)
    for g in groups:
        by_config[g.config][g.mode] = g
    for config, modes in by_config.items():
        if "open" in modes and "closed" in modes:
            found.append(("open vs closed", f"{config}: closed - open", modes["closed"], modes["open"]))
    for a, b in combinations(groups, 2):
        if a.mode != b.mode or a.config == b.config:
            continue
        if (a.backend, a.model) == (b.backend, b.model) and a.think != b.think:
            on, off = (a, b) if a.think else (b, a)
            found.append(("think on vs off", f"{on.config} - {off.config} [{a.mode}]", on, off))
        elif a.model != b.model:
            found.append(("model vs model", f"{a.config} - {b.config} [{a.mode}]", a, b))
    return found


def comparison_table(groups: list[Group]) -> list[str]:
    rows = [
        "| comparison | A - B | paired runs | A | B | difference [95% bootstrap CI] | A only / B only | McNemar p |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    notes = []
    for kind, description, a, b in comparisons(groups):
        result = paired_comparison(pairs(a.records, b.records))
        if result is None:
            notes.append(f"- {description}: no shared (task, repeat) runs to pair.")
            continue
        rows.append(
            f"| {kind} | {description} | {result.n} | {result.rate_a:.1%} | {result.rate_b:.1%} | "
            f"{result.diff:+.1%} [{result.ci[0]:+.1%}, {result.ci[1]:+.1%}] | {result.a_only} / {result.b_only} | "
            f"{result.p_value:.3g} |"
        )
    if len(rows) == 2:
        rows = ["No paired comparisons yet: they need two configs or both modes on the same tasks."]
    return rows + notes


def outcome_table(groups: list[Group]) -> list[str]:
    counts = [Counter(r["outcome_type"] for r in g.records) for g in groups]
    outcomes = sorted({o for c in counts for o in c}, key=lambda o: (o not in SUCCESSES, o))
    rows = ["| outcome | " + " | ".join(g.label for g in groups) + " |", "|---|" + "---:|" * len(groups)]
    rows += [f"| {o} | " + " | ".join(str(c.get(o, 0)) for c in counts) + " |" for o in outcomes]
    return rows


def reproduce_block(groups: list[Group], matrix_path: str) -> list[str]:
    commits = sorted({str(r.get("git_commit")) for g in groups for r in g.records if r.get("git_commit")})
    versions: dict[str, set[str]] = defaultdict(set)
    for g in groups:
        for r in g.records:
            for name, version in (r.get("versions") or {}).items():
                versions[name].add(version)
    lines = [
        "```bash",
        "uv sync",
        f"uv run graspgraph-matrix plan {matrix_path}",
        f"uv run graspgraph-matrix run {matrix_path} --yes",
        "uv run python scripts/report.py results/ --out reports/",
        "```",
        "",
        f"- git commit(s): {', '.join(f'`{c}`' for c in commits) or 'not recorded (older result files)'}",
    ]
    if versions:
        lines.append("- packages: " + ", ".join(f"{n} {'/'.join(sorted(v))}" for n, v in sorted(versions.items())))
    return lines


def results_markdown(groups: list[Group], missing: list[str], matrix_path: str) -> str:
    fake = is_fake(groups)
    lines = []
    if fake:
        lines += [f"# {FAKE_BANNER}", "", "Generated from the scripted fake backend to show the format. These are not model results.", ""]
    lines += ["# Benchmark results", "", _sources(groups), ""]
    if missing:
        lines += ["Not run yet (skipped in this report): " + ", ".join(missing), ""]
    lines += [
        "## Headline",
        "",
        "Success and recovery rates with 95% Wilson intervals. Recovery is success on runs that had a fault "
        "injected. Latency is model time per run; cost per success divides total cost by successful runs.",
        "",
        *headline_table(groups),
        "",
        "## Success by family",
        "",
        *family_table(groups),
        "",
        "## Paired comparisons",
        "",
        "Runs pair up on task and repeat, so both sides faced the same scene and the same fault. The "
        "difference is A - B in success rate with a paired bootstrap 95% CI (10,000 resamples, fixed seed); "
        "the p-value is McNemar's exact test on the pairs where only one side succeeded.",
        "",
        *comparison_table(groups),
        "",
        "## Outcome types",
        "",
        *outcome_table(groups),
        "",
        "## How to reproduce",
        "",
        *reproduce_block(groups, matrix_path),
        "",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------- failures.md


def failures_markdown(groups: list[Group]) -> str:
    """Up to MAX_FAILURE_EXAMPLES real runs per failure outcome type, spread over families."""
    by_outcome: dict[str, list[tuple[Group, Record]]] = defaultdict(list)
    for g in groups:
        for r in g.records:
            if r["outcome_type"] not in SUCCESSES:
                by_outcome[r["outcome_type"]].append((g, r))
    lines = [f"# {FAKE_BANNER}", ""] if is_fake(groups) else []
    lines += ["# Failure examples", "", _sources(groups), ""]
    if not by_outcome:
        lines.append("No failures in these results.")
    for outcome, examples in sorted(by_outcome.items(), key=lambda kv: -len(kv[1])):
        lines += [f"## {outcome} ({len(examples)} runs)", ""]
        chosen, families = [], set()
        for g, r in examples:  # prefer one example per family
            if r["family"] not in families:
                chosen.append((g, r))
                families.add(r["family"])
            if len(chosen) == MAX_FAILURE_EXAMPLES:
                break
        for g, r in chosen:
            failed = [f"{p['name']}: {p['message']}" for p in r.get("primitive_results", []) if not p.get("success")]
            lines += [
                f"### {r['task_id']} ({g.label}, repeat {r.get('repeat', 0)})",
                "",
                f"- instruction: {get_task(r['task_id']).instruction}",
                f"- plan(s): `{r.get('plans') or r.get('raw_plan') or r.get('tool_calls')}`",
                f"- outcome: {r['outcome_type']}: {r.get('detail', '')}",
            ]
            if r.get("fault_injected"):
                lines.append(f"- fault: {r.get('fault_type')} at step {r.get('fault_step')}: {r.get('fault_detail', '')}")
            lines += [f"- mismatch: {m}" for m in r.get("mismatches") or []]
            lines += [f"- failed step: {f}" for f in failed]
            if r.get("plan_error"):
                lines.append(f"- plan error: {r['plan_error']}")
            if r.get("error"):
                lines.append(f"- error: {r['error']}")
            lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------- charts


def _title(text: str, groups: list[Group]) -> str:
    sources = sorted({g.backend for g in groups})
    title = f"{text}\n({', '.join(sources)})"
    return f"{FAKE_BANNER}\n{title}" if is_fake(groups) else title


def write_charts(groups: list[Group], out: Path) -> list[Path]:
    """The three PNG charts; returns their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colours = model_colours(groups)
    hatches = {"open": "//", "closed": ""}
    written = []

    def finish(fig, name: str) -> None:
        path = out / name
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    # 1. Success per family, grouped by config, with Wilson whiskers.
    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(FAMILY_NAMES)), 5))
    width = 0.8 / max(len(groups), 1)
    for i, g in enumerate(groups):
        rates, low, high = [], [], []
        for family in FAMILY_NAMES:
            rs = [r for r in g.records if r["family"] == family]
            k, n = sum(bool(r["success"]) for r in rs), len(rs)
            lo, hi = wilson(k, n)
            rate = k / n if n else 0.0
            rates.append(rate)
            low.append(max(rate - lo, 0.0))  # rounding can put a bound a hair past the rate
            high.append(max(hi - rate, 0.0))
        x = np.arange(len(FAMILY_NAMES)) + (i - (len(groups) - 1) / 2) * width
        ax.bar(x, rates, width, yerr=[low, high], capsize=2, label=g.label, color=colours[g.model],
               hatch=hatches.get(g.mode, ""), edgecolor="white", alpha=0.6 if g.think else 1.0)
    ax.set_xticks(np.arange(len(FAMILY_NAMES)), FAMILY_NAMES, rotation=30, ha="right")
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(_title("Success rate per task family (95% Wilson CI; hatched = open loop)", groups))
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    finish(fig, "success_by_family.png")

    # 2. Open vs closed, with and without faults, per config.
    configs = list(dict.fromkeys(g.config for g in groups))
    cells = [(mode, faulted) for mode in ("open", "closed") for faulted in (False, True)]
    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(configs)), 5))
    width = 0.8 / len(cells)
    for j, (mode, faulted) in enumerate(cells):
        rates, errs, colour_list = [], [[], []], []
        for config in configs:
            g = next((g for g in groups if g.config == config and g.mode == mode), None)
            rs = [r for r in (g.records if g else []) if bool(r.get("fault_injected")) == faulted]
            k, n = sum(bool(r["success"]) for r in rs), len(rs)
            lo, hi = wilson(k, n)
            rate = k / n if n else 0.0
            rates.append(rate)
            errs[0].append(max(rate - lo, 0.0))
            errs[1].append(max(hi - rate, 0.0))
            colour_list.append(colours[g.model] if g else "#cccccc")
        x = np.arange(len(configs)) + (j - (len(cells) - 1) / 2) * width
        ax.bar(x, rates, width, yerr=errs, capsize=2, color=colour_list, hatch=hatches[mode],
               alpha=0.55 if faulted else 1.0, edgecolor="white")
    ax.set_xticks(np.arange(len(configs)), configs)
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(_title("Open vs closed loop, with and without faults (colour = model)", groups))
    from matplotlib.patches import Patch

    styles = [  # the legend shows the bar styles in grey: colour belongs to the model
        Patch(facecolor="#888888", hatch=hatches[mode], alpha=0.55 if faulted else 1.0, edgecolor="white",
              label=f"{mode} loop, {'with' if faulted else 'no'} fault")
        for mode, faulted in cells
    ]
    ax.legend(handles=styles, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    finish(fig, "open_vs_closed.png")

    # 3. Cost per successful task vs success rate, one point per config and mode.
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"open": "o", "closed": "s"}
    for i, g in enumerate(groups):
        cost = sum(r.get("cost_usd") or 0.0 for r in g.records)
        rate = g.successes / len(g.records)
        per_success = cost / g.successes if g.successes else 0.0
        ax.scatter(rate, per_success, s=80, color=colours[g.model], marker=markers.get(g.mode, "o"), label=g.label)
        ax.annotate(g.label, (rate, per_success), textcoords="offset points", xytext=(6, 4 + 10 * (i % 3)), fontsize=8)
    ax.set_xlabel("success rate")
    ax.set_ylabel("cost per successful task (USD; free models sit at 0)")
    ax.set_xlim(0, 1.1)
    ax.set_ylim(bottom=0)
    ax.set_title(_title("Cost per successful task vs success rate (circle = open, square = closed)", groups))
    finish(fig, "cost_vs_success.png")
    return written


def write_report(results_dir: Path, out: Path, expected_configs: list[str] | None = None, matrix_path: str = "experiments/main.toml") -> list[Path]:
    """Write results.md, failures.md and the charts for every config file in `results_dir`."""
    groups = load_groups(results_dir)
    present = {g.config for g in groups}
    missing = [c for c in (expected_configs or []) if c not in present]
    out.mkdir(parents=True, exist_ok=True)
    results_md, failures_md = out / "results.md", out / "failures.md"
    if not groups:
        results_md.write_text(f"# Benchmark results\n\nNo result files in {results_dir}/ yet.\n")
        return [results_md]
    results_md.write_text(results_markdown(groups, missing, matrix_path))
    failures_md.write_text(failures_markdown(groups))
    return [results_md, failures_md, *write_charts(groups, out)]
