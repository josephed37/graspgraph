"""Write the benchmark report: results.md, failures.md and charts, from a results directory.

    uv run python scripts/report.py results/ --out reports/
    uv run python scripts/report.py results/ --out reports/ --matrix experiments/main.toml   # notes missing configs

Reads the config files (*.jsonl) directly in the results directory; smoke runs and
calibration sit in subdirectories and are left out. Results from the fake backend are
labelled "FAKE DATA - not real results" in every file.
"""

import argparse
from pathlib import Path

from graspgraph.bench.matrix import load_matrix
from graspgraph.bench.report import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--matrix", type=Path, help="matrix file, to list configs that have no results yet")
    args = parser.parse_args()
    expected = [c.name for c in load_matrix(args.matrix).configs] if args.matrix else None
    matrix_path = str(args.matrix) if args.matrix else "experiments/main.toml"
    for path in write_report(args.results, args.out, expected, matrix_path):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
