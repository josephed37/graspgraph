"""Summarise benchmark JSONL files as plain-text tables."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from graspgraph.bench.runner import read_records, run_key


def latest_records(path: Path) -> list[dict[str, Any]]:
    """Records in a file, keeping only the last one per task run (a resumed retry replaces an error)."""
    latest: dict[tuple, dict[str, Any]] = {}
    for record in read_records(path):
        latest[run_key(record)] = record
    return list(latest.values())


def _rate(ok: int, n: int) -> str:
    return f"{ok}/{n} ({ok / n:.1%})" if n else "0/0"


def summarize(path: Path) -> str:
    """A table for one results file: success overall and per family, outcomes, latency, tokens, cost."""
    records = latest_records(path)
    if not records:
        return f"{path}: no records"
    configs = sorted({f"{r['backend']}/{r['model']} think={r['think']} mode={r['mode']}" for r in records})
    lines = [f"== {path}  ({'; '.join(configs)})", f"overall success  {_rate(sum(r['success'] for r in records), len(records))}", ""]

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_family[r["family"]].append(r)
    lines.append(f"{'family':12s} {'runs':>5s} {'success':>16s}")
    for family, rs in by_family.items():
        lines.append(f"{family:12s} {len(rs):5d} {_rate(sum(r['success'] for r in rs), len(rs)):>16s}")

    lines += ["", "outcome types"]
    for outcome, n in Counter(r["outcome_type"] for r in records).most_common():
        lines.append(f"  {outcome:24s} {n:5d}")

    latency = sum(r["latency_s"] for r in records) / len(records)
    tokens_in = sum(r["input_tokens"] for r in records)
    tokens_out = sum(r["output_tokens"] for r in records)
    cost = sum(r["cost_usd"] for r in records)
    lines += ["", f"mean latency {latency:.2f}s | tokens in {tokens_in:,} out {tokens_out:,} | cost ${cost:.4f}"]
    return "\n".join(lines)
