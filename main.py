from __future__ import annotations

import argparse
from pathlib import Path

from features.cleaning_simulation import (
    CleaningSimulationParameters,
    export_heatmap_html,
    simulate_cleaning,
    simulation_to_json,
)
from features.edge_classification import analysis_to_json, classify_step_edges


DATASET_DIR = Path("step_datasets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify STEP edges and optionally simulate surface cleaning.")
    parser.add_argument(
        "step_file",
        nargs="?",
        type=Path,
        help="STEP file to analyze. Defaults to the first step_datasets/Par*.STEP file.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional JSON file for the edge classification result.",
    )
    parser.add_argument(
        "--simulate-cleaning",
        action="store_true",
        help="Run the sampled surface cleaning and redeposition simulation with top and bottom spray.",
    )
    parser.add_argument(
        "--cleaning-json-output",
        type=Path,
        help="Optional JSON file for cleaning simulation samples and scores.",
    )
    parser.add_argument(
        "--heatmap-output",
        type=Path,
        help="Optional standalone HTML heatmap for the cleaning simulation.",
    )
    parser.add_argument(
        "--mesh-deflection",
        type=float,
        default=CleaningSimulationParameters.mesh_linear_deflection,
        help="STEP mesh linear deflection used by the cleaning simulation.",
    )
    parser.add_argument(
        "--flow-steps",
        type=int,
        default=CleaningSimulationParameters.flow_steps,
        help="Number of water-flow propagation steps used by the cleaning simulation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    step_file = args.step_file or _default_step_file()

    analysis = classify_step_edges(step_file)
    print(f"STEP file: {Path(analysis.source_file).name}")
    print(f"Shared edges: {analysis.edge_count}")
    print(f"Convex edges: {analysis.convex_count}")
    print(f"Concave edges: {analysis.concave_count}")
    print(f"Neutral edges: {analysis.neutral_count}")

    if args.json_output:
        args.json_output.write_text(analysis_to_json(analysis), encoding="utf-8")
        print(f"Wrote JSON: {args.json_output.resolve()}")

    if args.simulate_cleaning or args.cleaning_json_output or args.heatmap_output:
        parameters = CleaningSimulationParameters(
            mesh_linear_deflection=args.mesh_deflection,
            flow_steps=args.flow_steps,
        )
        cleaning = simulate_cleaning(step_file, parameters=parameters)
        summary = cleaning.summary()
        print(f"Cleaning samples: {summary['sample_count']}")
        print(f"Max hot spot score: {summary['max_hotspot_score']}")
        print(f"Hot spots >= 0.65: {summary['hotspot_count_0_65']}")

        if args.cleaning_json_output:
            args.cleaning_json_output.write_text(simulation_to_json(cleaning), encoding="utf-8")
            print(f"Wrote cleaning JSON: {args.cleaning_json_output.resolve()}")

        if args.heatmap_output:
            export_heatmap_html(cleaning, args.heatmap_output)
            print(f"Wrote heatmap: {args.heatmap_output.resolve()}")

    return 0


def _default_step_file() -> Path:
    matches = sorted(DATASET_DIR.glob("Par*.STEP"))
    if not matches:
        raise FileNotFoundError(f"No Par*.STEP files found in {DATASET_DIR.resolve()}")
    return matches[0]


if __name__ == "__main__":
    raise SystemExit(main())
