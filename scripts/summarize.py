"""Print a summary table for each benchmark results file, then compare them side by side.

    uv run python scripts/summarize.py results/*.jsonl
    uv run python scripts/summarize.py results/a_open.jsonl results/a_closed.jsonl   # open vs closed
"""

import argparse
from pathlib import Path

from graspgraph.bench.summary import compare, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    files = parser.parse_args().files
    for path in files:
        print(summarize(path) + "\n")
    comparison = compare(files)
    if comparison:
        print(comparison)


if __name__ == "__main__":
    main()
