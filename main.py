from __future__ import annotations

import argparse
from pathlib import Path

from step_graph import export_colored_model, export_feature_candidates, graph_to_json, build_attributed_adjacency_graph, save_graph_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an attributed adjacency graph from a STEP B-rep file."
    )
    parser.add_argument(
        "step_file",
        nargs="?",
        default="s14-08.stp",
        help="Path to the STEP file to parse.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="aag.json",
        help="Path for the JSON graph output.",
    )
    parser.add_argument(
        "--plot-output",
        default="aag.svg",
        help="Path for the SVG graph plot output.",
    )
    parser.add_argument(
        "--features-dir",
        default="feature_candidates",
        help="Folder where decomposed feature candidate files will be written.",
    )
    parser.add_argument(
        "--colored-step-output",
        default="colored_features.step",
        help="Path for the colored STEP export of the original solid.",
    )
    parser.add_argument(
        "--colored-gltf-output",
        default="colored_features.gltf",
        help="Path for the colored GLTF export of the original solid.",
    )
    parser.add_argument(
        "--colored-html-output",
        default="colored_features.html",
        help="Path for a simple HTML viewer for the colored GLTF.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    graph = build_attributed_adjacency_graph(args.step_file)
    graph_json = graph_to_json(graph)
    output_path = Path(args.output)
    output_path.write_text(graph_json, encoding="utf-8")
    print(f"Wrote graph JSON to {output_path.resolve()}")
    plot_path = save_graph_plot(graph, args.plot_output)
    print(f"Wrote graph plot to {plot_path.resolve()}")
    feature_dir = export_feature_candidates(graph, args.features_dir)
    print(f"Wrote feature candidates to {feature_dir.resolve()}")
    colored = export_colored_model(
        args.step_file,
        graph,
        args.colored_step_output,
        args.colored_gltf_output,
        args.colored_html_output,
    )
    print(f"Wrote colored STEP to {colored['step'].resolve()}")
    print(f"Wrote colored GLTF to {colored['gltf'].resolve()}")
    print(f"Wrote colored viewer to {colored['html'].resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
