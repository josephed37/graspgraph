"""Print a summary table for each benchmark results file.

    uv run python scripts/summarize.py results/*.jsonl
"""

import argparse
from pathlib import Path

from graspgraph.bench.summary import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    for path in parser.parse_args().files:
        print(summarize(path) + "\n")


if __name__ == "__main__":
    main()
