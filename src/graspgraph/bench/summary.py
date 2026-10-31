"""Summarise benchmark JSONL files as plain-text tables."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from graspgraph.bench.runner import CONFIG_FIELDS, LEGACY_DEFAULTS, read_records, run_key


def latest_records(path: Path) -> list[dict[str, Any]]:
    """Records in a file, keeping only the last one per task run (a resumed retry replaces an error)."""
    latest: dict[tuple, dict[str, Any]] = {}
    for record in read_records(path):
        latest[run_key(record)] = record
    return list(latest.values())


def _rate(ok: int, n: int) -> str:
    return f"{ok}/{n} ({ok / n:.1%})" if n else "0/0"


def _config(record: dict[str, Any]) -> str:
    return " ".join(f"{k}={record.get(k, LEGACY_DEFAULTS.get(k))}" for k in CONFIG_FIELDS)


def _groups(path: Path) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in latest_records(path):
        groups[_config(r)].append(r)
    return groups


def summarize(path: Path) -> str:
    """Tables for one results file, one per configuration in it (model, mode, prompt version...)."""
    groups = _groups(path)
    if not groups:
        return f"{path}: no records"
    return "\n\n".join(f"== {path}  ({config})\n{_table(rs)}" for config, rs in groups.items())


def _successes(records: list[dict[str, Any]]) -> int:
    return sum(r["success"] for r in records)


def _faulted(record: dict[str, Any]) -> bool:
    return bool(record.get("fault_injected", False))


def _table(records: list[dict[str, Any]]) -> str:
    """Success overall and per family, outcome counts, faults and recovery, latency, tokens and cost."""
    lines = [f"overall success  {_rate(_successes(records), len(records))}", ""]

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_family[r["family"]].append(r)
    lines.append(f"{'family':12s} {'runs':>5s} {'success':>16s}")
    for family, rs in by_family.items():
        lines.append(f"{family:12s} {len(rs):5d} {_rate(_successes(rs), len(rs)):>16s}")

    lines += ["", "outcome types"]
    for outcome, n in Counter(r["outcome_type"] for r in records).most_common():
        lines.append(f"  {outcome:24s} {n:5d}")

    faulted = [r for r in records if _faulted(r)]
    clean = [r for r in records if not _faulted(r)]
    lines += [
        "",
        f"mean replans per task {sum(r.get('replans', 0) for r in records) / len(records):.2f}",
        f"success without faults {_rate(_successes(clean), len(clean))} | with faults {_rate(_successes(faulted), len(faulted))}",
    ]
    if faulted:
        kinds = Counter(r["fault_type"] for r in faulted)
        lines.append(f"recovery rate {_rate(_successes(faulted), len(faulted))} ({', '.join(f'{k} {n}' for k, n in kinds.items())})")

    latency = sum(r["latency_s"] for r in records) / len(records)
    tokens_in = sum(r["input_tokens"] for r in records)
    tokens_out = sum(r["output_tokens"] for r in records)
    cost = sum(r["cost_usd"] for r in records)
    planned = [r["planned_steps"] for r in records if "planned_steps" in r]
    steps = f"mean planned steps {sum(planned) / len(planned):.1f} | " if planned else ""
    lines += ["", f"{steps}mean latency {latency:.2f}s | tokens in {tokens_in:,} out {tokens_out:,} | cost ${cost:.4f}"]
    return "\n".join(lines)


def compare(paths: list[Path]) -> str:
    """Success per family side by side, one column per (file, configuration), e.g. open vs closed."""
    columns: list[tuple[str, list[dict[str, Any]]]] = []
    for path in paths:
        groups = _groups(path)
        for records in groups.values():
            r = records[0]
            label = f"{path.stem}" if len(groups) == 1 else f"{path.stem}:{r.get('mode', 'open')}/{r.get('prompt_version')}"
            columns.append((label, records))
    if len(columns) < 2:
        return ""
    width = max(16, *(len(label) for label, _ in columns))
    families = list(dict.fromkeys(r["family"] for _, records in columns for r in records))
    header = f"{'family':12s} " + " ".join(f"{label:>{width}s}" for label, _ in columns)
    lines = ["== comparison", header]

    def row(name: str, pick) -> str:
        cells = []
        for _, records in columns:
            subset = [r for r in records if pick(r)]
            cells.append(f"{_rate(_successes(subset), len(subset)) if subset else '-':>{width}s}")
        return f"{name:12s} " + " ".join(cells)

    lines += [row(f, lambda r, f=f: r["family"] == f) for f in families]
    lines += [row("overall", lambda r: True), row("with faults", _faulted), row("no faults", lambda r: not _faulted(r))]
    replans = " ".join(f"{sum(r.get('replans', 0) for r in rs) / len(rs):>{width}.2f}" for _, rs in columns)
    lines.append(f"{'replans/task':12s} {replans}")
    return "\n".join(lines)
