from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from features.cleaning_simulation import (
    CleaningSample,
    CleaningSimulationParameters,
    SurfaceMesh,
    step_to_surface_mesh,
    simulate_cleaning_on_mesh,
)
from features.cleaning_simulation.math_utils import clamp, dot, norm, normalize, scale, sub
from features.edge_classification import classify_step_edges
from features.hole_finding import FACE_FORM_TYPES, FaceFormFinder


DATASET_DIR = Path("step_datasets")
DEFAULT_OUTPUT_DIR = Path("cache") / "cleaning_retention"
STEP_PATTERNS = ("*.step", "*.stp", "*.STEP", "*.STP")
FACE_CSV_NAME = "face_particle_retention.csv"
FORM_CSV_NAME = "form_particle_retention.csv"
TOP_SURFACE_CSV_NAME = "object_surface_particle_retention.csv"
NEIGHBOR_CSV_NAME = "dirty_surface_neighbor_context.csv"
FAILURE_CSV_NAME = "cleaning_retention_failures.csv"
ALL_AXIS_WATER_DIRECTIONS = (
    (0.0, 0.0, -1.0),
    (0.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 1.0, 0.0),
)
RETAINED_PARTICLE_MARKER_THRESHOLD = 0.02


@dataclass
class RetentionAggregate:
    datasets: set[str] = field(default_factory=set)
    faces: set[tuple[str, int]] = field(default_factory=set)
    sample_count: int = 0
    surface_area: float = 0.0
    retained_particle_mass: float = 0.0
    retained_particle_marker_count: int = 0
    remaining_dust_sum: float = 0.0
    max_remaining_dust: float = 0.0
    exposure_area_sum: float = 0.0
    water_dose_area_sum: float = 0.0
    cleaning_dose_area_sum: float = 0.0
    hotspot_score_area_sum: float = 0.0
    redeposition_area_sum: float = 0.0
    poor_drainage_area_sum: float = 0.0
    concavity_area_sum: float = 0.0
    hiddenness_area_sum: float = 0.0

    def add_sample(self, dataset: str, face_id: int, sample: CleaningSample) -> None:
        area = max(sample.area, 0.0)
        self.datasets.add(dataset)
        self.faces.add((dataset, face_id))
        self.sample_count += 1
        self.surface_area += area
        self.retained_particle_mass += sample.remaining_dust * area
        if sample.remaining_dust >= RETAINED_PARTICLE_MARKER_THRESHOLD:
            self.retained_particle_marker_count += 1
        self.remaining_dust_sum += sample.remaining_dust
        self.max_remaining_dust = max(self.max_remaining_dust, sample.remaining_dust)
        self.exposure_area_sum += sample.exposure * area
        self.water_dose_area_sum += sample.water_dose * area
        self.cleaning_dose_area_sum += sample.cleaning_dose * area
        self.hotspot_score_area_sum += sample.hotspot_score * area
        self.redeposition_area_sum += sample.redeposition * area
        self.poor_drainage_area_sum += sample.poor_drainage * area
        self.concavity_area_sum += sample.concavity * area
        self.hiddenness_area_sum += sample.hiddenness * area

    def add_aggregate(self, other: "RetentionAggregate") -> None:
        self.datasets.update(other.datasets)
        self.faces.update(other.faces)
        self.sample_count += other.sample_count
        self.surface_area += other.surface_area
        self.retained_particle_mass += other.retained_particle_mass
        self.retained_particle_marker_count += other.retained_particle_marker_count
        self.remaining_dust_sum += other.remaining_dust_sum
        self.max_remaining_dust = max(self.max_remaining_dust, other.max_remaining_dust)
        self.exposure_area_sum += other.exposure_area_sum
        self.water_dose_area_sum += other.water_dose_area_sum
        self.cleaning_dose_area_sum += other.cleaning_dose_area_sum
        self.hotspot_score_area_sum += other.hotspot_score_area_sum
        self.redeposition_area_sum += other.redeposition_area_sum
        self.poor_drainage_area_sum += other.poor_drainage_area_sum
        self.concavity_area_sum += other.concavity_area_sum
        self.hiddenness_area_sum += other.hiddenness_area_sum

    @property
    def retained_particle_ratio(self) -> float:
        if self.surface_area <= 1e-12:
            return 0.0
        return self.retained_particle_mass / self.surface_area

    @property
    def mean_remaining_dust(self) -> float:
        if self.sample_count == 0:
            return 0.0
        return self.remaining_dust_sum / self.sample_count

    def area_weighted_mean(self, total: float) -> float:
        if self.surface_area <= 1e-12:
            return 0.0
        return total / self.surface_area


@dataclass
class BoundaryAggregate:
    convex_count: int = 0
    concave_count: int = 0
    neutral_count: int = 0

    @property
    def total_count(self) -> int:
        return self.convex_count + self.concave_count + self.neutral_count

    @property
    def label(self) -> str:
        if self.total_count == 0:
            return "unknown"
        if self.convex_count and self.concave_count:
            return "mixed"
        if self.concave_count:
            return "concave"
        if self.convex_count:
            return "convex"
        return "neutral"

    def add(self, convexity: str) -> None:
        if convexity == "convex":
            self.convex_count += 1
        elif convexity == "concave":
            self.concave_count += 1
        else:
            self.neutral_count += 1


@dataclass
class MeshBoundaryAggregate:
    sample_edge_count: int = 0
    normal_dot_sum: float = 0.0
    angle_score_sum: float = 0.0
    concavity_score_sum: float = 0.0

    def add(self, normal_dot: float, angle_score: float, concavity_score: float) -> None:
        self.sample_edge_count += 1
        self.normal_dot_sum += normal_dot
        self.angle_score_sum += angle_score
        self.concavity_score_sum += concavity_score

    @property
    def mean_normal_dot(self) -> float:
        return self.mean(self.normal_dot_sum)

    @property
    def mean_angle_score(self) -> float:
        return self.mean(self.angle_score_sum)

    @property
    def mean_concavity_score(self) -> float:
        return self.mean(self.concavity_score_sum)

    @property
    def label(self) -> str:
        if self.sample_edge_count == 0:
            return "unknown"
        if self.mean_concavity_score >= 0.05:
            return "mesh_concave"
        if self.mean_angle_score >= 0.05:
            return "mesh_sharp_nonconcave"
        return "mesh_smooth_or_coplanar"

    def mean(self, total: float) -> float:
        if self.sample_edge_count == 0:
            return 0.0
        return total / self.sample_edge_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the cleaning simulation for every STEP dataset and report which "
            "analytic face forms retain the most particles."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Directory containing STEP files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV reports are written.",
    )
    parser.add_argument(
        "--mesh-deflection",
        type=float,
        default=CleaningSimulationParameters.mesh_linear_deflection,
        help="STEP mesh linear deflection used by the cleaning simulation.",
    )
    parser.add_argument(
        "--mesh-angular-deflection",
        type=float,
        default=CleaningSimulationParameters.mesh_angular_deflection,
        help="STEP mesh angular deflection used by the cleaning simulation.",
    )
    parser.add_argument(
        "--flow-steps",
        type=int,
        default=CleaningSimulationParameters.flow_steps,
        help="Number of water-flow propagation steps.",
    )
    parser.add_argument(
        "--water-force",
        type=float,
        default=CleaningSimulationParameters.water_force,
        help="Multiplier for direct water dose.",
    )
    parser.add_argument(
        "--spray-mode",
        choices=("top-bottom", "all-axis"),
        default="top-bottom",
        help="Spray from top/bottom only or from all six axis directions like the desktop UI.",
    )
    parser.add_argument(
        "--top-surfaces-per-object",
        type=int,
        default=5,
        help="Number of highest-retention surfaces to write per STEP object.",
    )
    parser.add_argument(
        "--neighbor-scope",
        choices=("dirty", "all"),
        default="dirty",
        help="Write neighbor rows only around dirty faces or around every face.",
    )
    parser.add_argument(
        "--cleaned-threshold",
        type=float,
        default=0.2,
        help="A neighboring face is labeled cleaned when its retained particle ratio is at or below this value.",
    )
    parser.add_argument(
        "--dirty-threshold",
        type=float,
        default=0.2,
        help="A target face is labeled dirty when its retained particle ratio is at or above this value.",
    )
    parser.add_argument(
        "--skip-edge-convexity",
        action="store_true",
        help="Skip exact convex/concave edge classification and only use mesh adjacency.",
    )
    parser.add_argument(
        "--skip-face-csv",
        action="store_true",
        help="Do not write the full per-face CSV. Useful for large ABC runs.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Only process the first N STEP files after sorting.",
    )
    parser.add_argument(
        "--sample-files",
        type=int,
        default=None,
        help="Deterministically sample this many STEP files from the full sorted dataset.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used with --sample-files.",
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        default=None,
        help="Skip STEP files larger than this size before max/sample selection.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress after this many STEP files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    step_files = find_step_files(args.dataset_dir)
    if args.max_file_size_mb is not None:
        max_bytes = max(0.0, args.max_file_size_mb) * 1024 * 1024
        step_files = [path for path in step_files if path.stat().st_size <= max_bytes]
    if args.sample_files is not None:
        sample_count = max(0, args.sample_files)
        if sample_count < len(step_files):
            rng = random.Random(args.sample_seed)
            step_files = sorted(
                rng.sample(step_files, sample_count),
                key=lambda path: str(path).casefold(),
            )
    elif args.max_files is not None:
        step_files = step_files[: max(0, args.max_files)]
    if not step_files:
        raise FileNotFoundError(f"No STEP files found in {args.dataset_dir.resolve()}")

    parameters = CleaningSimulationParameters(
        mesh_linear_deflection=args.mesh_deflection,
        mesh_angular_deflection=args.mesh_angular_deflection,
        flow_steps=args.flow_steps,
        water_force=args.water_force,
        water_directions=water_directions_for_mode(args.spray_mode),
    )
    face_rows, form_rows, top_surface_rows, neighbor_rows, failure_rows = analyze_datasets(
        step_files,
        args.dataset_dir,
        parameters,
        top_surfaces_per_object=args.top_surfaces_per_object,
        neighbor_scope=args.neighbor_scope,
        cleaned_threshold=args.cleaned_threshold,
        dirty_threshold=args.dirty_threshold,
        include_edge_convexity=not args.skip_edge_convexity,
        collect_face_rows=not args.skip_face_csv,
        progress_every=args.progress_every,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    face_csv = args.output_dir / FACE_CSV_NAME
    form_csv = args.output_dir / FORM_CSV_NAME
    top_surface_csv = args.output_dir / TOP_SURFACE_CSV_NAME
    neighbor_csv = args.output_dir / NEIGHBOR_CSV_NAME
    failure_csv = args.output_dir / FAILURE_CSV_NAME
    if not args.skip_face_csv:
        write_csv(face_csv, face_rows, FACE_COLUMNS)
    write_csv(form_csv, form_rows, FORM_COLUMNS)
    write_csv(top_surface_csv, top_surface_rows, FACE_COLUMNS)
    write_csv(neighbor_csv, neighbor_rows, NEIGHBOR_COLUMNS)
    write_csv(failure_csv, failure_rows, FAILURE_COLUMNS)

    print(f"Processed {len(step_files)} STEP files.")
    if not args.skip_face_csv:
        print(f"Wrote face retention table: {face_csv.resolve()}")
    print(f"Wrote form retention table: {form_csv.resolve()}")
    print(f"Wrote top dirty surfaces table: {top_surface_csv.resolve()}")
    print(f"Wrote dirty surface neighbor table: {neighbor_csv.resolve()}")
    if failure_rows:
        print(f"Wrote failures table: {failure_csv.resolve()}")
    print_top_forms(form_rows)
    return 0


def find_step_files(dataset_dir: Path) -> list[Path]:
    matches: dict[str, Path] = {}
    for pattern in STEP_PATTERNS:
        for path in dataset_dir.rglob(pattern):
            if path.is_file():
                matches[str(path.resolve()).casefold()] = path
    return sorted(matches.values(), key=lambda path: str(path).casefold())


def water_directions_for_mode(mode: str) -> tuple[tuple[float, float, float], ...]:
    if mode == "all-axis":
        return ALL_AXIS_WATER_DIRECTIONS
    return CleaningSimulationParameters.water_directions


def analyze_datasets(
    step_files: Iterable[Path],
    dataset_dir: Path,
    parameters: CleaningSimulationParameters,
    *,
    top_surfaces_per_object: int,
    neighbor_scope: str,
    cleaned_threshold: float,
    dirty_threshold: float,
    include_edge_convexity: bool,
    collect_face_rows: bool,
    progress_every: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    step_files = list(step_files)
    all_face_rows: list[dict[str, object]] = []
    top_surface_rows: list[dict[str, object]] = []
    neighbor_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    form_aggregates: dict[str, RetentionAggregate] = defaultdict(RetentionAggregate)
    finder = FaceFormFinder()

    for index, step_path in enumerate(step_files, start=1):
        dataset_name = dataset_label(step_path, dataset_dir)
        if index == 1 or index == len(step_files) or index % max(1, progress_every) == 0:
            print(f"[{index}/{len(step_files)}] Processing {dataset_name}...")

        try:
            form_by_face = {
                form.face_id: form.form_type
                for form in finder.find_step(step_path).forms
            }
            mesh = step_to_surface_mesh(step_path, parameters=parameters)
            result = simulate_cleaning_on_mesh(mesh, parameters=parameters)
        except Exception as exc:
            failure_rows.append(
                {
                    "stage": "object_analysis",
                    "dataset": dataset_name,
                    "path": str(step_path),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            continue

        boundaries = classify_boundaries(step_path, include_edge_convexity, failure_rows, dataset_name)
        face_neighbors, directed_flow_counts, mesh_boundaries = mesh_face_relations(mesh, result.samples)
        face_aggregates: dict[int, RetentionAggregate] = defaultdict(RetentionAggregate)

        for sample in result.samples:
            face_aggregates[sample.face_id].add_sample(dataset_name, sample.face_id, sample)

        file_retained_mass = sum(aggregate.retained_particle_mass for aggregate in face_aggregates.values())
        file_rows: list[dict[str, object]] = []
        for face_id, aggregate in sorted(face_aggregates.items()):
            form_type = form_by_face.get(face_id, "unknown")
            form_aggregates[form_type].add_aggregate(aggregate)
            file_rows.append(face_row(dataset_name, face_id, form_type, aggregate, file_retained_mass))

        add_rank(
            file_rows,
            rank_column="face_retention_rank_in_file",
            sort_columns=("retained_particle_marker_count", "retained_particle_mass"),
        )
        if collect_face_rows:
            all_face_rows.extend(file_rows)
        top_surface_rows.extend(
            row
            for row in file_rows
            if int(row["face_retention_rank_in_file"]) <= max(0, top_surfaces_per_object)
            and int(row["retained_particle_marker_count"]) > 0
        )
        neighbor_rows.extend(
            neighbor_context_rows(
                dataset_name,
                file_rows,
                face_aggregates,
                form_by_face,
                face_neighbors,
                directed_flow_counts,
                boundaries,
                mesh_boundaries,
                top_surfaces_per_object=max(0, top_surfaces_per_object),
                neighbor_scope=neighbor_scope,
                cleaned_threshold=cleaned_threshold,
                dirty_threshold=dirty_threshold,
            )
        )

    total_retained_mass = sum(aggregate.retained_particle_mass for aggregate in form_aggregates.values())
    form_rows = [
        form_row(form_type, aggregate, total_retained_mass)
        for form_type, aggregate in sorted(
            form_aggregates.items(),
            key=lambda item: form_sort_key(item[0]),
        )
    ]
    add_rank(
        form_rows,
        rank_column="form_retention_rank_overall",
        sort_columns=("retained_particle_marker_count", "retained_particle_mass"),
    )
    form_rows.sort(key=lambda row: int(row["form_retention_rank_overall"]))
    return all_face_rows, form_rows, top_surface_rows, neighbor_rows, failure_rows


def dataset_label(step_path: Path, dataset_dir: Path) -> str:
    try:
        return step_path.relative_to(dataset_dir).as_posix()
    except ValueError:
        return step_path.name


def classify_boundaries(
    step_path: Path,
    include_edge_convexity: bool,
    failure_rows: list[dict[str, object]],
    dataset_name: str,
) -> dict[tuple[int, int], BoundaryAggregate]:
    if not include_edge_convexity:
        return {}

    boundaries: dict[tuple[int, int], BoundaryAggregate] = defaultdict(BoundaryAggregate)
    try:
        edge_analysis = classify_step_edges(step_path)
    except Exception as exc:
        failure_rows.append(
            {
                "stage": "edge_classification",
                "dataset": dataset_name,
                "path": str(step_path),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        return boundaries

    for edge in edge_analysis.edges:
        boundaries[pair_key(edge.source_face, edge.target_face)].add(edge.convexity)
    return boundaries


def mesh_face_relations(
    mesh: SurfaceMesh,
    samples: list[CleaningSample],
) -> tuple[
    dict[int, set[int]],
    dict[tuple[int, int], int],
    dict[tuple[int, int], MeshBoundaryAggregate],
]:
    face_neighbors: dict[int, set[int]] = defaultdict(set)
    mesh_boundaries: dict[tuple[int, int], MeshBoundaryAggregate] = defaultdict(MeshBoundaryAggregate)
    samples_by_id = {sample.id: sample for sample in samples}
    directed_flow_counts: dict[tuple[int, int], int] = defaultdict(int)

    for triangle in mesh.triangles:
        for neighbor_id in mesh.neighbors.get(triangle.id, []):
            neighbor = mesh.triangles[neighbor_id]
            if triangle.face_id == neighbor.face_id:
                continue
            face_neighbors[triangle.face_id].add(neighbor.face_id)
            face_neighbors[neighbor.face_id].add(triangle.face_id)
            if neighbor_id <= triangle.id:
                continue
            normal_dot = dot(triangle.normal, neighbor.normal)
            angle_score = clamp((1.0 - normal_dot) / 0.7)
            neighbor_offset = sub(neighbor.point, triangle.point)
            concavity_score = 0.0
            if norm(neighbor_offset) > 1e-12:
                toward_neighbor = normalize(neighbor_offset)
                toward_triangle = scale(toward_neighbor, -1.0)
                concavity_score = angle_score * max(
                    max(0.0, dot(triangle.normal, toward_neighbor)),
                    max(0.0, dot(neighbor.normal, toward_triangle)),
                )
            mesh_boundaries[pair_key(triangle.face_id, neighbor.face_id)].add(
                normal_dot,
                angle_score,
                clamp(concavity_score),
            )

    for sample in samples:
        if sample.downstream_id is None:
            continue
        downstream = samples_by_id.get(sample.downstream_id)
        if downstream is None or downstream.face_id == sample.face_id:
            continue
        directed_flow_counts[(sample.face_id, downstream.face_id)] += 1

    return face_neighbors, directed_flow_counts, mesh_boundaries


def neighbor_context_rows(
    dataset: str,
    file_rows: list[dict[str, object]],
    face_aggregates: dict[int, RetentionAggregate],
    form_by_face: dict[int, str],
    face_neighbors: dict[int, set[int]],
    directed_flow_counts: dict[tuple[int, int], int],
    boundaries: dict[tuple[int, int], BoundaryAggregate],
    mesh_boundaries: dict[tuple[int, int], MeshBoundaryAggregate],
    *,
    top_surfaces_per_object: int,
    neighbor_scope: str,
    cleaned_threshold: float,
    dirty_threshold: float,
) -> list[dict[str, object]]:
    rows_by_face = {int(row["face_id"]): row for row in file_rows}
    if neighbor_scope == "all":
        dirty_faces = [int(row["face_id"]) for row in file_rows]
    else:
        dirty_faces = [
            int(row["face_id"])
            for row in file_rows
            if int(row["face_retention_rank_in_file"]) <= top_surfaces_per_object
            and int(row["retained_particle_marker_count"]) > 0
        ]
    neighbor_rows: list[dict[str, object]] = []

    for dirty_face_id in dirty_faces:
        dirty_row = rows_by_face[dirty_face_id]
        dirty_aggregate = face_aggregates[dirty_face_id]
        dirty_retention = dirty_aggregate.retained_particle_ratio
        for neighbor_face_id in sorted(face_neighbors.get(dirty_face_id, ())):
            neighbor_aggregate = face_aggregates.get(neighbor_face_id)
            if neighbor_aggregate is None:
                continue
            key = pair_key(dirty_face_id, neighbor_face_id)
            boundary = boundaries.get(key, BoundaryAggregate())
            mesh_boundary = mesh_boundaries.get(key, MeshBoundaryAggregate())
            neighbor_retention = neighbor_aggregate.retained_particle_ratio
            neighbor_rows.append(
                {
                    "dataset": dataset,
                    "dirty_face_id": dirty_face_id,
                    "dirty_form_type": form_by_face.get(dirty_face_id, "unknown"),
                    "dirty_rank_in_object": dirty_row["face_retention_rank_in_file"],
                    "dirty_retained_particle_ratio": dirty_retention,
                    "dirty_retained_particle_mass": dirty_aggregate.retained_particle_mass,
                    "dirty_is_above_threshold": int(dirty_retention >= dirty_threshold),
                    "dirty_area_weighted_exposure": dirty_aggregate.area_weighted_mean(
                        dirty_aggregate.exposure_area_sum
                    ),
                    "dirty_area_weighted_water_dose": dirty_aggregate.area_weighted_mean(
                        dirty_aggregate.water_dose_area_sum
                    ),
                    "dirty_area_weighted_cleaning_dose": dirty_aggregate.area_weighted_mean(
                        dirty_aggregate.cleaning_dose_area_sum
                    ),
                    "dirty_area_weighted_hotspot_score": dirty_aggregate.area_weighted_mean(
                        dirty_aggregate.hotspot_score_area_sum
                    ),
                    "dirty_area_weighted_concavity": dirty_aggregate.area_weighted_mean(
                        dirty_aggregate.concavity_area_sum
                    ),
                    "dirty_area_weighted_hiddenness": dirty_aggregate.area_weighted_mean(
                        dirty_aggregate.hiddenness_area_sum
                    ),
                    "neighbor_face_id": neighbor_face_id,
                    "neighbor_form_type": form_by_face.get(neighbor_face_id, "unknown"),
                    "neighbor_retained_particle_ratio": neighbor_retention,
                    "neighbor_retained_particle_mass": neighbor_aggregate.retained_particle_mass,
                    "neighbor_is_cleaned": int(neighbor_retention <= cleaned_threshold),
                    "neighbor_is_cleaner_than_dirty_face": int(neighbor_retention < dirty_retention),
                    "neighbor_area_weighted_exposure": neighbor_aggregate.area_weighted_mean(
                        neighbor_aggregate.exposure_area_sum
                    ),
                    "neighbor_area_weighted_water_dose": neighbor_aggregate.area_weighted_mean(
                        neighbor_aggregate.water_dose_area_sum
                    ),
                    "neighbor_area_weighted_cleaning_dose": neighbor_aggregate.area_weighted_mean(
                        neighbor_aggregate.cleaning_dose_area_sum
                    ),
                    "neighbor_area_weighted_hotspot_score": neighbor_aggregate.area_weighted_mean(
                        neighbor_aggregate.hotspot_score_area_sum
                    ),
                    "neighbor_area_weighted_concavity": neighbor_aggregate.area_weighted_mean(
                        neighbor_aggregate.concavity_area_sum
                    ),
                    "neighbor_area_weighted_hiddenness": neighbor_aggregate.area_weighted_mean(
                        neighbor_aggregate.hiddenness_area_sum
                    ),
                    "retention_ratio_delta_dirty_minus_neighbor": dirty_retention - neighbor_retention,
                    "flow_from_dirty_to_neighbor_count": directed_flow_counts.get(
                        (dirty_face_id, neighbor_face_id),
                        0,
                    ),
                    "flow_from_neighbor_to_dirty_count": directed_flow_counts.get(
                        (neighbor_face_id, dirty_face_id),
                        0,
                    ),
                    "mesh_boundary_label": mesh_boundary.label,
                    "mesh_boundary_sample_edge_count": mesh_boundary.sample_edge_count,
                    "mesh_boundary_mean_normal_dot": mesh_boundary.mean_normal_dot,
                    "mesh_boundary_mean_angle_score": mesh_boundary.mean_angle_score,
                    "mesh_boundary_mean_concavity_score": mesh_boundary.mean_concavity_score,
                    "boundary_convexity": boundary.label,
                    "boundary_convex_edge_count": boundary.convex_count,
                    "boundary_concave_edge_count": boundary.concave_count,
                    "boundary_neutral_edge_count": boundary.neutral_count,
                    "boundary_edge_count": boundary.total_count,
                }
            )

    return neighbor_rows


def pair_key(face_a: int, face_b: int) -> tuple[int, int]:
    return (face_a, face_b) if face_a <= face_b else (face_b, face_a)


def face_row(
    dataset: str,
    face_id: int,
    form_type: str,
    aggregate: RetentionAggregate,
    file_retained_mass: float,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "face_id": face_id,
        "form_type": form_type,
        "sample_count": aggregate.sample_count,
        "surface_area": aggregate.surface_area,
        "retained_particle_mass": aggregate.retained_particle_mass,
        "retained_particle_marker_count": aggregate.retained_particle_marker_count,
        "retained_particle_ratio": aggregate.retained_particle_ratio,
        "retained_particle_share_in_file": ratio(aggregate.retained_particle_mass, file_retained_mass),
        "face_retention_rank_in_file": 0,
        "mean_remaining_dust": aggregate.mean_remaining_dust,
        "max_remaining_dust": aggregate.max_remaining_dust,
        "area_weighted_exposure": aggregate.area_weighted_mean(aggregate.exposure_area_sum),
        "area_weighted_water_dose": aggregate.area_weighted_mean(aggregate.water_dose_area_sum),
        "area_weighted_cleaning_dose": aggregate.area_weighted_mean(aggregate.cleaning_dose_area_sum),
        "area_weighted_hotspot_score": aggregate.area_weighted_mean(aggregate.hotspot_score_area_sum),
        "area_weighted_redeposition": aggregate.area_weighted_mean(aggregate.redeposition_area_sum),
        "area_weighted_poor_drainage": aggregate.area_weighted_mean(aggregate.poor_drainage_area_sum),
        "area_weighted_concavity": aggregate.area_weighted_mean(aggregate.concavity_area_sum),
        "area_weighted_hiddenness": aggregate.area_weighted_mean(aggregate.hiddenness_area_sum),
    }


def form_row(
    form_type: str,
    aggregate: RetentionAggregate,
    total_retained_mass: float,
) -> dict[str, object]:
    return {
        "form_type": form_type,
        "dataset_count": len(aggregate.datasets),
        "face_count": len(aggregate.faces),
        "sample_count": aggregate.sample_count,
        "surface_area": aggregate.surface_area,
        "retained_particle_mass": aggregate.retained_particle_mass,
        "retained_particle_marker_count": aggregate.retained_particle_marker_count,
        "retained_particle_ratio": aggregate.retained_particle_ratio,
        "retained_particle_share_total": ratio(aggregate.retained_particle_mass, total_retained_mass),
        "form_retention_rank_overall": 0,
        "mean_remaining_dust": aggregate.mean_remaining_dust,
        "max_remaining_dust": aggregate.max_remaining_dust,
        "area_weighted_exposure": aggregate.area_weighted_mean(aggregate.exposure_area_sum),
        "area_weighted_water_dose": aggregate.area_weighted_mean(aggregate.water_dose_area_sum),
        "area_weighted_cleaning_dose": aggregate.area_weighted_mean(aggregate.cleaning_dose_area_sum),
        "area_weighted_hotspot_score": aggregate.area_weighted_mean(aggregate.hotspot_score_area_sum),
        "area_weighted_redeposition": aggregate.area_weighted_mean(aggregate.redeposition_area_sum),
        "area_weighted_poor_drainage": aggregate.area_weighted_mean(aggregate.poor_drainage_area_sum),
        "area_weighted_concavity": aggregate.area_weighted_mean(aggregate.concavity_area_sum),
        "area_weighted_hiddenness": aggregate.area_weighted_mean(aggregate.hiddenness_area_sum),
    }


def add_rank(
    rows: list[dict[str, object]],
    *,
    rank_column: str,
    sort_columns: tuple[str, str],
) -> None:
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row[sort_columns[0]]),
            -float(row[sort_columns[1]]),
            str(row.get("dataset", "")),
            str(row.get("form_type", "")),
            int(row.get("face_id", 0)),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row[rank_column] = rank


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: csv_value(row[column]) for column in columns})


def print_top_forms(form_rows: list[dict[str, object]], limit: int = 6) -> None:
    print("Top retained-particle forms:")
    for row in sorted(form_rows, key=lambda item: int(item["form_retention_rank_overall"]))[:limit]:
        print(
            f"  {row['form_retention_rank_overall']}. {row['form_type']}: "
            f"ratio={float(row['retained_particle_ratio']):.4f}, "
            f"retained_mass={float(row['retained_particle_mass']):.4f}, "
            f"faces={row['face_count']}"
        )


def ratio(value: float, total: float) -> float:
    if total <= 1e-12:
        return 0.0
    return value / total


def csv_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def form_sort_key(form_type: str) -> tuple[int, str]:
    if form_type in FACE_FORM_TYPES:
        return FACE_FORM_TYPES.index(form_type), form_type
    return len(FACE_FORM_TYPES), form_type


FACE_COLUMNS = [
    "dataset",
    "face_id",
    "form_type",
    "sample_count",
    "surface_area",
    "retained_particle_mass",
    "retained_particle_marker_count",
    "retained_particle_ratio",
    "retained_particle_share_in_file",
    "face_retention_rank_in_file",
    "mean_remaining_dust",
    "max_remaining_dust",
    "area_weighted_exposure",
    "area_weighted_water_dose",
    "area_weighted_cleaning_dose",
    "area_weighted_hotspot_score",
    "area_weighted_redeposition",
    "area_weighted_poor_drainage",
    "area_weighted_concavity",
    "area_weighted_hiddenness",
]

FORM_COLUMNS = [
    "form_type",
    "dataset_count",
    "face_count",
    "sample_count",
    "surface_area",
    "retained_particle_mass",
    "retained_particle_marker_count",
    "retained_particle_ratio",
    "retained_particle_share_total",
    "form_retention_rank_overall",
    "mean_remaining_dust",
    "max_remaining_dust",
    "area_weighted_exposure",
    "area_weighted_water_dose",
    "area_weighted_cleaning_dose",
    "area_weighted_hotspot_score",
    "area_weighted_redeposition",
    "area_weighted_poor_drainage",
    "area_weighted_concavity",
    "area_weighted_hiddenness",
]

NEIGHBOR_COLUMNS = [
    "dataset",
    "dirty_face_id",
    "dirty_form_type",
    "dirty_rank_in_object",
    "dirty_retained_particle_ratio",
    "dirty_retained_particle_mass",
    "dirty_is_above_threshold",
    "dirty_area_weighted_exposure",
    "dirty_area_weighted_water_dose",
    "dirty_area_weighted_cleaning_dose",
    "dirty_area_weighted_hotspot_score",
    "dirty_area_weighted_concavity",
    "dirty_area_weighted_hiddenness",
    "neighbor_face_id",
    "neighbor_form_type",
    "neighbor_retained_particle_ratio",
    "neighbor_retained_particle_mass",
    "neighbor_is_cleaned",
    "neighbor_is_cleaner_than_dirty_face",
    "neighbor_area_weighted_exposure",
    "neighbor_area_weighted_water_dose",
    "neighbor_area_weighted_cleaning_dose",
    "neighbor_area_weighted_hotspot_score",
    "neighbor_area_weighted_concavity",
    "neighbor_area_weighted_hiddenness",
    "retention_ratio_delta_dirty_minus_neighbor",
    "flow_from_dirty_to_neighbor_count",
    "flow_from_neighbor_to_dirty_count",
    "mesh_boundary_label",
    "mesh_boundary_sample_edge_count",
    "mesh_boundary_mean_normal_dot",
    "mesh_boundary_mean_angle_score",
    "mesh_boundary_mean_concavity_score",
    "boundary_convexity",
    "boundary_convex_edge_count",
    "boundary_concave_edge_count",
    "boundary_neutral_edge_count",
    "boundary_edge_count",
]

FAILURE_COLUMNS = [
    "stage",
    "dataset",
    "path",
    "error_type",
    "error_message",
]


if __name__ == "__main__":
    raise SystemExit(main())
