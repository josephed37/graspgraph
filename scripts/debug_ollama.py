"""Show what Ollama really returns for a benchmark task, before LangChain parses it.

    uv run python scripts/debug_ollama.py --task-ids tower-001 tower-007 --repeats 3
    uv run python scripts/debug_ollama.py --task-ids arrange-004 --stream off --temperature 0

For each attempt it prints the text, thinking, tool calls and done_reason Ollama sent, and
saves every raw chunk to outputs/ollama_debug/. Use it on tasks that scored no_tool_calls
or degenerate_output to see what the model actually produced.
"""

import argparse
import json
from pathlib import Path

from graspgraph.planner.backends import DEFAULT_MODELS
from graspgraph.planner.ollama_debug import build_request, raw_chat
from graspgraph.planner.plans import PLAN_PROTOCOLS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-ids", nargs="+", required=True)
    parser.add_argument("--model", default=DEFAULT_MODELS["ollama"])
    parser.add_argument("--plan-protocol", choices=PLAN_PROTOCOLS, default="submit_plan")
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--stream", choices=["on", "off", "both"], default="both", help="the benchmark streams (on)")
    parser.add_argument("--ollama-url")
    parser.add_argument("--out", type=Path, default=Path("outputs/ollama_debug"))
    args = parser.parse_args()

    modes = {"on": [True], "off": [False], "both": [True, False]}[args.stream]
    args.out.mkdir(parents=True, exist_ok=True)
    for task_id in args.task_ids:
        client, params = build_request(
            task_id, args.model, args.plan_protocol, args.think, args.max_tokens, args.temperature, args.ollama_url
        )
        for repeat in range(args.repeats):
            for stream in modes:
                result = raw_chat(client, params, stream)
                path = args.out / f"{task_id}_r{repeat}_{'stream' if stream else 'nostream'}.json"
                path.write_text(json.dumps(result, indent=1, default=str))
                calls = [c.get("function", {}).get("name") for c in result["tool_calls"]]
                print(
                    f"{task_id} r{repeat} stream={'on ' if stream else 'off'} done={result['done_reason']} "
                    f"tokens={result['output_tokens']} tool_calls={calls}"
                )
                for label in ("content", "thinking"):
                    if result[label]:
                        print(f"    {label}: {result[label][:400]!r}")
                print(f"    raw -> {path}")


if __name__ == "__main__":
    main()
