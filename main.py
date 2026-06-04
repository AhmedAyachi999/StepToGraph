from __future__ import annotations

import argparse
from pathlib import Path

from step_graph import analysis_to_json, classify_step_edges
from visualization import write_edge_visualization


DEFAULT_OUTPUT_DIR = Path("visualization/output")
DATASET_DIR = Path("step_datasets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize convex and concave STEP edges.")
    parser.add_argument(
        "step_file",
        nargs="?",
        type=Path,
        help="STEP file to analyze. Defaults to the first step_datasets/Par*.STEP file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the generated 3D edge viewer.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional JSON file for the edge classification result.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    step_file = args.step_file or _default_step_file()

    analysis = classify_step_edges(step_file)
    views = write_edge_visualization(step_file, analysis, args.output_dir)

    print(f"STEP file: {Path(analysis.source_file).name}")
    print(f"Shared edges: {analysis.edge_count}")
    print(f"Convex edges: {analysis.convex_count}")
    print(f"Concave edges: {analysis.concave_count}")
    print(f"Neutral edges ignored by viewer: {analysis.neutral_count}")
    print(f"Wrote edge viewer: {views['html'].resolve()}")

    if args.json_output:
        args.json_output.write_text(analysis_to_json(analysis), encoding="utf-8")
        print(f"Wrote JSON: {args.json_output.resolve()}")

    return 0


def _default_step_file() -> Path:
    matches = sorted(DATASET_DIR.glob("Par*.STEP"))
    if not matches:
        raise FileNotFoundError(f"No Par*.STEP files found in {DATASET_DIR.resolve()}")
    return matches[0]


if __name__ == "__main__":
    raise SystemExit(main())
