import json

from graspgraph.bench.report import FAKE_BANNER, comparisons, load_groups, write_report

FAMILIES = {"pick-place-000": "pick_place", "stack-000": "stack", "tower-000": "tower", "ambiguous-000": "ambiguous"}


def _record(task_id, mode, success, backend="fake", model="oracle", repeat=0, think=None, fault=False, cost=0.0):
    outcome = ("recovered_success" if fault and mode == "closed" else "success") if success else "primitive_failed"
    return {
        "task_id": task_id, "family": FAMILIES[task_id], "seed": 0, "repeat": repeat, "mode": mode,
        "backend": backend, "model": model, "think": think, "success": success, "outcome_type": outcome,
        "detail": "the detail", "fault_injected": fault, "fault_type": "drop" if fault else None,
        "fault_step": 2 if fault else None, "fault_detail": "it fell" if fault else None,
        "replans": int(fault and mode == "closed"), "llm_calls": 1 + int(fault and mode == "closed"),
        "latency_s": 1.5, "cost_usd": cost, "input_tokens": 100, "output_tokens": 10,
        "mismatches": ["Expected block_red to be held"] if fault else [],
        "primitive_results": [] if success else [{"name": "place_on", "success": False, "message": "dropped"}],
        "plans": [[{"tool": "pick", "args": {"obj_name": "block_red"}}]],
        "git_commit": "abc123", "versions": {"graspgraph": "0.1.0"},
    }


def _write(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _suite(mode, backend="fake", model="oracle", **kwargs):
    return [
        _record("pick-place-000", mode, True, backend, model, **kwargs),
        _record("stack-000", mode, mode == "closed", backend, model, fault=True, **kwargs),
        _record("tower-000", mode, False, backend, model, **kwargs),
        _record("ambiguous-000", mode, True, backend, model, **kwargs),
    ]


def test_report_from_open_only_results_is_labelled_fake(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    _write(results / "gemma.jsonl", _suite("open"))
    written = write_report(results, tmp_path / "reports")
    names = {p.name for p in written}
    assert names == {"results.md", "failures.md", "success_by_family.png", "open_vs_closed.png", "cost_vs_success.png"}
    md = (tmp_path / "reports" / "results.md").read_text()
    assert md.startswith(f"# {FAKE_BANNER}") and "Data from: fake/oracle" in md
    assert "| gemma | open | fake/oracle | 4 | 50.0% [" in md  # success with its Wilson interval
    assert "No paired comparisons yet" in md
    assert all((tmp_path / "reports" / n).stat().st_size > 10_000 for n in names if n.endswith(".png"))
    failures = (tmp_path / "reports" / "failures.md").read_text()
    assert failures.startswith(f"# {FAKE_BANNER}") and "## primitive_failed (2 runs)" in failures
    assert "instruction: " in failures and "failed step: place_on: dropped" in failures


def test_open_and_closed_and_model_comparisons(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    _write(results / "gemma.jsonl", _suite("open") + _suite("closed"))
    # every Haiku run fails: 0% cells, whose Wilson bounds once gave the charts negative whiskers
    haiku = [{**r, "success": False, "outcome_type": "bad_placement"} for r in _suite("open", "anthropic", "claude-haiku-5-5", cost=0.01)]
    _write(results / "haiku.jsonl", haiku)
    groups = load_groups(results)
    kinds = [(kind, description) for kind, description, _, _ in comparisons(groups)]
    assert ("open vs closed", "gemma: closed - open") in kinds
    assert ("model vs model", "gemma - haiku [open]") in kinds
    write_report(results, tmp_path / "reports")
    md = (tmp_path / "reports" / "results.md").read_text()
    assert "| open vs closed | gemma: closed - open | 4 | 75.0% | 50.0% | +25.0% [" in md
    assert "| recovered_success |" in md and "$0.0400" in md  # outcome table and Haiku's total cost
    assert (tmp_path / "reports" / "open_vs_closed.png").exists()
    assert md.startswith(f"# {FAKE_BANNER}")  # one fake input is enough to label the whole report


def test_think_comparison_and_a_partial_matrix_without_fake_data(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    _write(results / "gemma-main.jsonl", _suite("open", "ollama", "gemma4:12b", think=False))
    _write(results / "gemma-think.jsonl", _suite("open", "ollama", "gemma4:12b", think=True))
    (results / "smoke").mkdir()
    _write(results / "smoke" / "gemma-main.jsonl", _suite("open", "fake"))  # subdirectories are ignored
    write_report(results, tmp_path / "reports", expected_configs=["gemma-main", "gemma-think", "sonnet-subset"])
    md = (tmp_path / "reports" / "results.md").read_text()
    assert FAKE_BANNER not in md and md.startswith("# Benchmark results")
    assert "Not run yet (skipped in this report): sonnet-subset" in md
    assert "| think on vs off | gemma-think - gemma-main [open] |" in md
    assert "git commit(s): `abc123`" in md


def test_an_empty_results_directory(tmp_path):
    (tmp_path / "results").mkdir()
    [md] = write_report(tmp_path / "results", tmp_path / "reports")
    assert "No result files" in md.read_text()
