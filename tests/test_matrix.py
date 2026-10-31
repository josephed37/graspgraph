import json

import pytest

from graspgraph.bench import matrix as m
from graspgraph.bench.matrix_cli import main as matrix_main
from graspgraph.bench.pricing import Price
from graspgraph.bench.runner import read_records
from graspgraph.planner import FakeBackend
from graspgraph.planner.backends import PlannerBackend, oracle_policy, oracle_repair

PRICES = {"fake": {"*": Price(0.0, 0.0), "paid-a": Price(100.0, 100.0), "paid-b": Price(100.0, 100.0)}}
CHEAP = ["ambiguous-000", "impossible-000"]  # nothing moves: fast


def _matrix(tmp_path, configs, max_usd=8.0, tasks=CHEAP):
    (tmp_path / "tasks.txt").write_text("\n".join(tasks) + "\n")
    lines = [f'results_dir = "{tmp_path / "results"}"', f"max_usd = {max_usd}"]
    for name, model, modes in configs:
        lines += [
            "[[config]]",
            f'name = "{name}"',
            'backend = "fake"',
            f'model = "{model}"',
            f"modes = {json.dumps(modes)}",
            'tasks = "tasks.txt"',
            "faults = 0.3",
            f'out = "{tmp_path / "results" / (name + ".jsonl")}"',
        ]
    path = tmp_path / "matrix.toml"
    path.write_text("\n".join(lines) + "\n")
    return m.load_matrix(path)


def _factory(config):  # each run costs $0.20 at the paid test price (1000 + 1000 tokens at $100/M)
    return FakeBackend(oracle_policy, model=config.model, input_tokens=1000, output_tokens=1000, repair=oracle_repair)


def _ran(report):
    """Configs that made task runs this time (a finished config reports 0)."""
    return {k: v for k, v in report.ran.items() if v}


def _run(matrix, **kwargs):
    log = []
    kwargs.setdefault("yes", True)
    report = m.run_matrix(
        matrix, prices=PRICES, backend_factory=_factory, checks=lambda c: [], smoke=False, log=log.append, **kwargs
    )
    return report, log


def test_plan_lists_every_run_and_the_total(tmp_path):
    matrix = _matrix(tmp_path, [("paid", "paid-a", ["open", "closed"]), ("free", "oracle", ["open"])])
    estimates, text = m.plan(matrix, prices=PRICES)
    assert [(e.config, e.mode) for e in estimates] == [("free", "open"), ("paid", "open"), ("paid", "closed")]
    assert all(e.runs == 2 for e in estimates)
    assert "ESTIMATES" in text and "total: 6 runs" in text and "default estimate" in text
    for name in ("paid", "free"):
        assert name in text


def test_free_configs_run_first(tmp_path):
    matrix = _matrix(tmp_path, [("paid", "paid-a", ["open"]), ("free", "oracle", ["open"])])
    report, log = _run(matrix)
    starts = [line for line in log if line.startswith("== ")]
    assert starts[0].startswith("== free") and starts[1].startswith("== paid")
    assert report.ran == {"free": 2, "paid": 2}


def test_the_global_cap_holds_across_configs_and_across_a_resume(tmp_path):
    matrix = _matrix(tmp_path, [("a", "paid-a", ["open", "closed"]), ("b", "paid-b", ["open", "closed"])], max_usd=1.0)
    report, log = _run(matrix)
    assert report.stopped_by_cap and report.spent_usd == pytest.approx(1.0)  # 4 runs of a, then 1 of b
    assert report.ran == {"a": 4, "b": 1}
    again, _ = _run(matrix)  # resuming: nothing more may be spent
    assert again.stopped_by_cap and _ran(again) == {} and again.spent_usd == pytest.approx(1.0)
    raised, _ = _run(matrix, max_usd=1.5)  # a higher cap picks up where it stopped
    assert raised.spent_usd == pytest.approx(1.4) and _ran(raised) == {"b": 2}


def test_paid_configs_need_yes_or_a_confirmation(tmp_path):
    matrix = _matrix(tmp_path, [("paid", "paid-a", ["open"]), ("free", "oracle", ["open"])])
    report, log = _run(matrix, yes=False)
    assert report.ran == {"free": 2} and "not confirmed" in report.skipped["paid"]
    asked = []
    report, _ = _run(matrix, yes=False, confirm=lambda q: asked.append(q) or True)
    assert _ran(report) == {"paid": 2} and "estimated at $" in asked[0] and "remains" in asked[0]


def test_only_and_skip(tmp_path):
    matrix = _matrix(tmp_path, [("one", "oracle", ["open"]), ("two", "oracle", ["open"]), ("three", "oracle", ["open"])])
    assert _run(matrix, only=["two"])[0].ran == {"two": 2}
    assert set(_run(matrix, skip=["two"])[0].ran) == {"one", "three"}
    with pytest.raises(ValueError, match="unknown configs"):
        _run(matrix, only=["four"])


def test_failed_preflight_or_smoke_skips_the_config(tmp_path):
    matrix = _matrix(tmp_path, [("ok", "oracle", ["open"]), ("broken", "oracle", ["open"])])
    log = []
    report = m.run_matrix(
        matrix, prices=PRICES, backend_factory=_factory, smoke=False, log=log.append,
        checks=lambda c: ["ANTHROPIC_API_KEY is not set"] if c.name == "broken" else [],
    )
    assert report.ran == {"ok": 2} and "ANTHROPIC_API_KEY" in report.skipped["broken"]

    class Down(PlannerBackend):
        def generate(self, messages, tools):
            raise ConnectionError("server down")

    report = m.run_matrix(
        matrix, only=["broken"], prices=PRICES, checks=lambda c: [], log=log.append,
        backend_factory=lambda c: Down("fake", c.model),
    )
    assert "smoke run failed: ConnectionError: server down" in report.skipped["broken"]
    assert read_records(tmp_path / "results" / "smoke" / "broken.jsonl")  # recorded, and counted by the cap


def test_open_and_closed_runs_of_a_task_get_the_same_fault(tmp_path):
    matrix = _matrix(tmp_path, [("pair", "oracle", ["open", "closed"])], tasks=["pick-place-000", "tower-000"])
    config = matrix.configs[0]
    object.__setattr__(config, "faults", 1.0)  # every eligible run gets a fault
    _run(matrix)
    by_mode = {}
    for r in read_records(config.out):
        by_mode.setdefault(r["mode"], {})[(r["task_id"], r["repeat"])] = (r["fault_type"], r["fault_injected"])
    assert by_mode["open"] == by_mode["closed"] and all(kind for kind, _ in by_mode["open"].values())


def test_calibration_records_tokens_per_mode_and_the_plan_uses_them(tmp_path):
    results = tmp_path / "results"
    backend = FakeBackend(oracle_policy, model="paid-a", input_tokens=1200, output_tokens=300, repair=oracle_repair)
    entry = m.calibrate(backend, results, n_tasks=2, max_usd=5.0, prices=PRICES, log=lambda _: None)
    assert set(entry["modes"]) == {"open", "closed"}
    assert entry["modes"]["open"]["input_tokens_per_run"] == 1200 and entry["modes"]["open"]["output_tokens_per_run"] == 300
    assert len(entry["tasks"]) == 4 and {t["mode"] for t in entry["tasks"]} == {"open", "closed"}
    assert m.calibration_tasks(2) == ["pick-place-000", "stack-000"]  # one per family first
    saved = json.loads((results / m.CALIBRATION_FILE).read_text())
    assert saved["models"]["paid-a"]["modes"]["closed"]["runs"] == 2

    matrix = _matrix(tmp_path, [("paid", "paid-a", ["open"])])
    [e], text = m.plan(matrix, prices=PRICES)
    assert e.basis == "calibrated" and e.input_tokens == 2 * 1200 and e.cost_usd == pytest.approx(2 * 0.15)
    assert "calibrated" in text


def test_preflight_messages(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert "ANTHROPIC_API_KEY is not set" in m.check_anthropic()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret")
    assert m.check_anthropic() is None
    message = m.check_ollama("gemma4:12b", "http://127.0.0.1:9")
    assert "not reachable" in message and "ollama serve" in message
    assert m.check_menagerie() is None
    assert m.check_writable(tmp_path / "out") is None


def test_cli_plan_and_run(tmp_path, capsys, monkeypatch):
    matrix = _matrix(tmp_path, [("free", "oracle", ["open"])])
    matrix_main(["plan", str(matrix.path)])
    assert "total: 2 runs" in capsys.readouterr().out
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    matrix_main(["run", str(matrix.path), "--no-smoke"])
    out = capsys.readouterr().out
    assert "done: 2 task runs" in out and len(read_records(matrix.configs[0].out)) == 2
