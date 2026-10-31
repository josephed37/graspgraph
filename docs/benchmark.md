# The benchmark

How the week-4 benchmark is designed, and how to run it end to end on your own machine. Real runs need a local Ollama and an Anthropic API key; nothing here is run in CI.

## Design

### Configs

Defined in [`experiments/main.toml`](../experiments/main.toml), one `[[config]]` per row. Edit it freely; `graspgraph-matrix plan` shows what an edit costs.

| config | backend / model | modes | faults | repeats | tasks | runs |
|---|---|---|---|---|---|---:|
| `gemma-main` | ollama `gemma4:12b`, thinking off | open, closed | 0.3, seed 0 | 3 | all 110 | 660 |
| `gemma-think` | ollama `gemma4:12b`, thinking on | open, closed | 0.3, seed 0 | 1 | all 110 | 220 |
| `haiku-main` | anthropic `claude-haiku-5-5` | open, closed | 0.3, seed 0 | 3 | all 110 | 660 |
| `sonnet-subset` | anthropic `claude-sonnet-5-5` | closed | 0.3, seed 0 | 1 | 50-task subset | 50 |

**Claude settings** are set explicitly in `planner/backends.py` (`ANTHROPIC_REASONING`), and each config can override them (`effort`, `thinking`, `max_tokens`):
- **Sonnet 5.5** defaults to effort "high", which would blow the budget, so it runs at **effort "low"**. Its thinking **cannot be disabled** (langchain-anthropic rejects `thinking: disabled` for it), so it uses adaptive thinking at that low effort, with a 2048-token output cap.
- **Haiku 5.5** lists no effort levels, so it runs with **thinking disabled** and no effort setting.

### Faults and pairing

`faults = 0.3` gives about 30% of the act tasks a seeded fault: `drop`, `displace` or `knock` (see the README's "Closed-loop planning and recovery"). Whether a run is faulted, the fault type and the step it hits come from `(task_id, repeat, fault_seed)`. So:
- the open and closed runs of a task and repeat get the **identical** fault;
- every model sees the same faults.

That is what makes the comparisons *paired*.

### Repeats

Models sample at their provider's default temperature, so a task can pass on one run and fail on the next. With 3 repeats, per-task success is the mean over the repeats, and overall rates count every run.

### The Sonnet subset

[`experiments/subsets/sonnet50.txt`](../experiments/subsets/sonnet50.txt) holds seeds 0, 2, 4, 6 and 8 of every family: 5 tasks per family, 50 in all, fixed and committed. Sonnet runs only those, closed loop, once. Its comparison with Haiku uses the 50 shared tasks (Haiku's repeat 0).

### Statistics

All in `bench/stats.py`, using numpy and the standard library only.
- **Success and recovery rates:** 95% Wilson score intervals, which behave well at 0%, 100% and small n.
- **Paired comparisons:**
  - **Which runs are compared:** runs that share task and repeat (open vs closed, thinking on vs off, model vs model).
  - **The difference:** A − B in success rate, with a paired bootstrap 95% CI (10,000 resamples, fixed seed).
  - **The p-value:** McNemar's exact test on the runs where only one side succeeded.
- **Recovery rate:** success among the runs that actually had a fault injected.

### Budget

The global cap is **$8.00** (`max_usd` in the matrix file, or `--max-usd`). It is computed from the cost recorded in every result file under `results/`: config runs, smoke runs and calibration. So a resumed or repeated `run` never double-counts and never spends past the cap. Paid configs ask before they start (or take `--yes`) and show their estimate and the remaining budget. The planned spend is about $6.

## Runbook

The steps take about 8 hours in total, nearly all of it Gemma. The estimates come from `graspgraph-matrix plan`; re-check them after calibration.

### 1. Update and install (1 minute, free)

```bash
git pull
uv sync
ls third_party/mujoco_menagerie/franka_emika_panda/scene.xml   # or clone Menagerie, see README
```

### 2. Check Ollama (2 minutes, free)

```bash
ollama pull gemma4:12b
ollama run gemma4:12b "hi" && ollama ps     # PROCESSOR should say 100% GPU
```

The planner asks for an 8192-token context on every request. Make it the server default too, so Ollama doesn't reload the model with a different context:

```bash
sudo systemctl edit ollama
#   [Service]
#   Environment="OLLAMA_CONTEXT_LENGTH=8192"
sudo systemctl daemon-reload && sudo systemctl restart ollama
ollama ps                                    # CONTEXT should read 8192 after the next request
```

### 3. Set the API key, and optionally LangSmith (1 minute, free)

```bash
export ANTHROPIC_API_KEY=...                 # never committed, never printed
# optional tracing: nothing is sent unless both of these are set
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=...
export LANGSMITH_PROJECT=graspgraph
```

### 4. Calibrate Haiku and Sonnet (5 minutes, about $0.05 to $0.30)

Each runs 10 tasks, one per family, in both modes, with faults as in the matrix, under a $0.30 cap. The measured tokens, calls and latency go to `results/calibration.json`, and `plan` uses them from then on.

```bash
uv run graspgraph-matrix calibrate --model claude-haiku-5-5 --tasks 10
uv run graspgraph-matrix calibrate --model claude-sonnet-5-5 --tasks 10
```

If Haiku rejects `thinking: disabled`, the calibration's records show `llm_error` with the API message. Set `thinking = "omit"` for `haiku-main` in the matrix file and calibrate again.

### 5. Plan and check the cost (10 seconds, free)

```bash
uv run graspgraph-matrix plan experiments/main.toml
```

Check that the total is under the cap and that the Haiku and Sonnet rows now say `calibrated`.

### 6. Run (about 7 to 8 hours, about $1 to $6)

```bash
uv run graspgraph-matrix run experiments/main.toml --yes
```

- **Order:** configs run cheapest first: `gemma-main`, then `gemma-think` (both free), then Haiku, then Sonnet.
- **Smoke run:** each backend first gets a 1-task smoke run (`results/smoke/`). A config whose preflight or smoke run fails is skipped with the reason, and the rest carry on.
- **Interruptions:** just run the same command again. Finished runs are skipped and the cap still counts everything already spent.
- **Part of the matrix:** use `--only gemma-main` or `--skip sonnet-subset` to run just some configs. Leave out `--yes` to be asked before each paid config.

### 7. Report (1 minute, free)

```bash
uv run python scripts/report.py results/ --out reports/ --matrix experiments/main.toml
```

This writes:
- **`reports/results.md`:** the headline table with CIs, per-family success, paired comparisons, outcome types, and how to reproduce (with the recorded git commit and package versions).
- **`reports/failures.md`:** up to three real examples per failure type.
- **Three PNG charts.**

It works on partial results; configs not run yet are listed.

## Checking the pipeline offline

[`experiments/fake.toml`](../experiments/fake.toml) is the same matrix with every model replaced by the scripted fake backend. It needs no Ollama, key or money, and finishes in about 10 minutes:

```bash
uv run graspgraph-matrix run experiments/fake.toml
uv run python scripts/report.py results/fake --out reports/fake
```

Everything it produces is labelled **FAKE DATA - not real results**. Don't publish it.
