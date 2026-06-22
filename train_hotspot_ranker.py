from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


DEFAULT_DATA_DIR = Path("cache") / "abc_cleaning_retention_2000_sample2mb_all_axis_wf05_rich"
DEFAULT_FACE_CSV = DEFAULT_DATA_DIR / "face_particle_retention.csv"
DEFAULT_NEIGHBOR_CSV = DEFAULT_DATA_DIR / "dirty_surface_neighbor_context.csv"
DEFAULT_FORM_CSV = DEFAULT_DATA_DIR / "form_particle_retention.csv"
DEFAULT_MODEL_OUTPUT = Path("cache") / "lightgbm_hotspot_model.txt"
DEFAULT_PREDICTIONS_OUTPUT = Path("cache") / "hotspot_ranker_predictions_wf05_markers.csv"
DEFAULT_HOLDOUT_PREDICTIONS_OUTPUT = Path("cache") / "hotspot_ranker_holdout_predictions_wf05_markers.csv"
DEFAULT_IMPORTANCE_OUTPUT = Path("cache") / "hotspot_ranker_feature_importance_wf05_markers.csv"
TARGET_COLUMN = "retained_particle_marker_count"
RELEVANCE_COLUMN = "hotspot_relevance"
SCORE_COLUMN = "predicted_hotspot_score"
FEATURE_COLUMNS = [
    "form_type",
    "sample_count",
    "form_retained_particle_ratio",
    "form_retained_particle_share_total",
    "form_face_count",
    "form_dataset_count",
    "form_retention_rank_overall",
    "surface_area",
    "area_weighted_exposure",
    "area_weighted_water_dose",
    "area_weighted_cleaning_dose",
    "area_weighted_hotspot_score",
    "area_weighted_redeposition",
    "area_weighted_poor_drainage",
    "area_weighted_concavity",
    "area_weighted_hiddenness",
    "boundary_count",
    "boundary_concave_count",
    "boundary_sharp_nonconcave_count",
    "boundary_smooth_count",
    "boundary_mean_angle_score",
    "boundary_max_angle_score",
    "boundary_mean_concavity_score",
    "boundary_max_concavity_score",
    "exact_boundary_count",
    "exact_boundary_convex_count",
    "exact_boundary_concave_count",
    "exact_boundary_neutral_count",
    "neighbor_count",
    "neighbor_surface_area_mean",
    "neighbor_surface_area_max",
    "neighbor_form_retained_particle_ratio_mean",
    "neighbor_form_retained_particle_ratio_max",
    "neighbor_form_retained_particle_share_total_mean",
    "neighbor_form_retained_particle_share_total_max",
    "neighbor_form_retention_rank_mean",
    "neighbor_form_retention_rank_min",
    "neighbor_area_weighted_concavity_mean",
    "neighbor_area_weighted_concavity_max",
    "neighbor_area_weighted_cleaning_dose_mean",
    "neighbor_area_weighted_cleaning_dose_min",
    "neighbor_area_weighted_hotspot_score_mean",
    "neighbor_area_weighted_hotspot_score_max",
    "neighbor_boundary_concave_count",
    "neighbor_boundary_convex_count",
    "neighbor_boundary_neutral_count",
    "neighbor_mesh_concave_count",
    "neighbor_mesh_sharp_nonconcave_count",
]


@dataclass
class BoundaryAggregate:
    count: int = 0
    concave_count: int = 0
    sharp_nonconcave_count: int = 0
    smooth_count: int = 0
    angle_sum: float = 0.0
    concavity_sum: float = 0.0
    max_angle: float = 0.0
    max_concavity: float = 0.0

    def add(self, label: str, sample_count: int, angle_score: float, concavity_score: float) -> None:
        count = max(1, sample_count)
        self.count += count
        self.angle_sum += angle_score * count
        self.concavity_sum += concavity_score * count
        self.max_angle = max(self.max_angle, angle_score)
        self.max_concavity = max(self.max_concavity, concavity_score)
        if label == "mesh_concave":
            self.concave_count += count
        elif label == "mesh_sharp_nonconcave":
            self.sharp_nonconcave_count += count
        else:
            self.smooth_count += count

    def row(self) -> dict[str, float | int]:
        if self.count == 0:
            return empty_boundary_features()
        return {
            "boundary_count": self.count,
            "boundary_concave_count": self.concave_count,
            "boundary_sharp_nonconcave_count": self.sharp_nonconcave_count,
            "boundary_smooth_count": self.smooth_count,
            "boundary_mean_angle_score": self.angle_sum / self.count,
            "boundary_max_angle_score": self.max_angle,
            "boundary_mean_concavity_score": self.concavity_sum / self.count,
            "boundary_max_concavity_score": self.max_concavity,
        }


@dataclass
class ExactBoundaryAggregate:
    count: int = 0
    convex_count: int = 0
    concave_count: int = 0
    neutral_count: int = 0

    def add(self, convex_count: int, concave_count: int, neutral_count: int) -> None:
        self.convex_count += max(0, convex_count)
        self.concave_count += max(0, concave_count)
        self.neutral_count += max(0, neutral_count)
        self.count += max(0, convex_count) + max(0, concave_count) + max(0, neutral_count)

    def row(self) -> dict[str, float | int]:
        return {
            "exact_boundary_count": self.count,
            "exact_boundary_convex_count": self.convex_count,
            "exact_boundary_concave_count": self.concave_count,
            "exact_boundary_neutral_count": self.neutral_count,
        }


@dataclass
class NeighborFeatureAggregate:
    count: int = 0
    surface_area_sum: float = 0.0
    surface_area_max: float = 0.0
    form_ratio_sum: float = 0.0
    form_ratio_max: float = 0.0
    form_share_sum: float = 0.0
    form_share_max: float = 0.0
    form_rank_sum: float = 0.0
    form_rank_min: float | None = None
    concavity_sum: float = 0.0
    concavity_max: float = 0.0
    cleaning_dose_sum: float = 0.0
    cleaning_dose_min: float | None = None
    hotspot_score_sum: float = 0.0
    hotspot_score_max: float = 0.0
    boundary_concave_count: int = 0
    boundary_convex_count: int = 0
    boundary_neutral_count: int = 0
    mesh_concave_count: int = 0
    mesh_sharp_nonconcave_count: int = 0

    def add(self, neighbor: dict[str, float], relation: pd.Series) -> None:
        self.count += 1
        surface_area = float(neighbor.get("surface_area", 0.0))
        form_ratio = float(neighbor.get("form_retained_particle_ratio", 0.0))
        form_share = float(neighbor.get("form_retained_particle_share_total", 0.0))
        form_rank = float(neighbor.get("form_retention_rank_overall", 0.0))
        concavity = float(neighbor.get("area_weighted_concavity", 0.0))
        cleaning_dose = float(neighbor.get("area_weighted_cleaning_dose", 0.0))
        hotspot_score = float(neighbor.get("area_weighted_hotspot_score", 0.0))

        self.surface_area_sum += surface_area
        self.surface_area_max = max(self.surface_area_max, surface_area)
        self.form_ratio_sum += form_ratio
        self.form_ratio_max = max(self.form_ratio_max, form_ratio)
        self.form_share_sum += form_share
        self.form_share_max = max(self.form_share_max, form_share)
        self.form_rank_sum += form_rank
        self.form_rank_min = form_rank if self.form_rank_min is None else min(self.form_rank_min, form_rank)
        self.concavity_sum += concavity
        self.concavity_max = max(self.concavity_max, concavity)
        self.cleaning_dose_sum += cleaning_dose
        self.cleaning_dose_min = (
            cleaning_dose if self.cleaning_dose_min is None else min(self.cleaning_dose_min, cleaning_dose)
        )
        self.hotspot_score_sum += hotspot_score
        self.hotspot_score_max = max(self.hotspot_score_max, hotspot_score)

        convexity = str(relation.get("boundary_convexity", "unknown"))
        if convexity == "concave":
            self.boundary_concave_count += 1
        elif convexity == "convex":
            self.boundary_convex_count += 1
        elif convexity == "neutral":
            self.boundary_neutral_count += 1

        mesh_label = str(relation.get("mesh_boundary_label", ""))
        if mesh_label == "mesh_concave":
            self.mesh_concave_count += 1
        elif mesh_label == "mesh_sharp_nonconcave":
            self.mesh_sharp_nonconcave_count += 1

    def row(self) -> dict[str, float | int]:
        if self.count == 0:
            return empty_neighbor_features()
        return {
            "neighbor_count": self.count,
            "neighbor_surface_area_mean": self.surface_area_sum / self.count,
            "neighbor_surface_area_max": self.surface_area_max,
            "neighbor_form_retained_particle_ratio_mean": self.form_ratio_sum / self.count,
            "neighbor_form_retained_particle_ratio_max": self.form_ratio_max,
            "neighbor_form_retained_particle_share_total_mean": self.form_share_sum / self.count,
            "neighbor_form_retained_particle_share_total_max": self.form_share_max,
            "neighbor_form_retention_rank_mean": self.form_rank_sum / self.count,
            "neighbor_form_retention_rank_min": self.form_rank_min or 0.0,
            "neighbor_area_weighted_concavity_mean": self.concavity_sum / self.count,
            "neighbor_area_weighted_concavity_max": self.concavity_max,
            "neighbor_area_weighted_cleaning_dose_mean": self.cleaning_dose_sum / self.count,
            "neighbor_area_weighted_cleaning_dose_min": self.cleaning_dose_min or 0.0,
            "neighbor_area_weighted_hotspot_score_mean": self.hotspot_score_sum / self.count,
            "neighbor_area_weighted_hotspot_score_max": self.hotspot_score_max,
            "neighbor_boundary_concave_count": self.boundary_concave_count,
            "neighbor_boundary_convex_count": self.boundary_convex_count,
            "neighbor_boundary_neutral_count": self.boundary_neutral_count,
            "neighbor_mesh_concave_count": self.mesh_concave_count,
            "neighbor_mesh_sharp_nonconcave_count": self.mesh_sharp_nonconcave_count,
        }


@dataclass(frozen=True)
class SplitInfo:
    name: str
    train_rows: int
    test_rows: int
    train_datasets: int
    test_datasets: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LightGBM LambdaRank hotspot model grouped by STEP object."
    )
    parser.add_argument("--face-csv", type=Path, default=DEFAULT_FACE_CSV)
    parser.add_argument("--neighbor-csv", type=Path, default=DEFAULT_NEIGHBOR_CSV)
    parser.add_argument("--form-csv", type=Path, default=DEFAULT_FORM_CSV)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS_OUTPUT)
    parser.add_argument("--holdout-predictions-output", type=Path, default=DEFAULT_HOLDOUT_PREDICTIONS_OUTPUT)
    parser.add_argument("--importance-output", type=Path, default=DEFAULT_IMPORTANCE_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--holdout-test-size",
        type=float,
        default=0.0,
        help="When > 0, evaluate once on this fraction of held-out STEP objects instead of cross-validation.",
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--max-faces-per-object",
        type=int,
        default=2000,
        help="Cap very large ranking groups after keeping all relevant faces and sampled lower-ranked faces.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = build_training_frame(args.face_csv, args.neighbor_csv, args.form_csv, top_k=args.top_k)
    frame = limit_faces_per_object(
        frame,
        max_faces_per_object=args.max_faces_per_object,
        random_state=args.random_state,
    )
    prepared = prepare_frame(frame)
    split_info: SplitInfo | None = None
    if args.holdout_test_size > 0.0:
        predictions, split_info = holdout_predict(
            prepared,
            test_size=args.holdout_test_size,
            random_state=args.random_state,
        )
        predictions_output = args.holdout_predictions_output
    else:
        predictions = cross_val_predict(prepared, folds=args.folds, random_state=args.random_state)
        predictions_output = args.predictions_output

    final_frame = sort_for_ranking(prepared)
    model = make_ranker(args.random_state)
    model.fit(
        final_frame[FEATURE_COLUMNS],
        final_frame[RELEVANCE_COLUMN],
        group=group_sizes(final_frame),
    )
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(args.model_output))
    write_predictions(predictions, predictions_output)
    write_importance(model, args.importance_output)

    metrics = evaluate(predictions, top_k=args.top_k)
    print_summary(prepared, metrics, args, predictions_output, split_info)
    return 0


def build_training_frame(face_csv: Path, neighbor_csv: Path, form_csv: Path, *, top_k: int) -> pd.DataFrame:
    if not face_csv.exists():
        raise FileNotFoundError(f"Missing face CSV: {face_csv}")
    if not neighbor_csv.exists():
        raise FileNotFoundError(f"Missing neighbor CSV: {neighbor_csv}")
    if not form_csv.exists():
        raise FileNotFoundError(f"Missing form CSV: {form_csv}")

    face_frame = pd.read_csv(face_csv)
    form_priors = load_form_priors(form_csv)
    face_features = face_feature_lookup(face_frame, form_priors)
    context_features = load_context_features(neighbor_csv, face_features)
    rows = []
    for _, row in face_frame.iterrows():
        dataset = str(row["dataset"])
        face_id = int(row["face_id"])
        form_type = str(row["form_type"])
        rows.append(
            {
                "dataset": dataset,
                "face_id": face_id,
                "form_type": form_type,
                "sample_count": float(row.get("sample_count", 0.0)),
                "surface_area": float(row.get("surface_area", 0.0)),
                TARGET_COLUMN: float(row[TARGET_COLUMN]),
                "face_retention_rank_in_file": float(row.get("face_retention_rank_in_file", 0.0)),
                "area_weighted_exposure": float(row.get("area_weighted_exposure", 0.0)),
                "area_weighted_water_dose": float(row.get("area_weighted_water_dose", 0.0)),
                "area_weighted_cleaning_dose": float(row.get("area_weighted_cleaning_dose", 0.0)),
                "area_weighted_hotspot_score": float(row.get("area_weighted_hotspot_score", 0.0)),
                "area_weighted_redeposition": float(row.get("area_weighted_redeposition", 0.0)),
                "area_weighted_poor_drainage": float(row.get("area_weighted_poor_drainage", 0.0)),
                "area_weighted_concavity": float(row.get("area_weighted_concavity", 0.0)),
                "area_weighted_hiddenness": float(row.get("area_weighted_hiddenness", 0.0)),
                **form_priors.get(form_type, empty_form_prior()),
                **context_features.get((dataset, face_id), empty_context_features()),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No ranking rows were built from the input CSV files.")
    add_derived_features(frame)
    frame[RELEVANCE_COLUMN] = relevance_labels(frame, top_k=top_k)
    return frame


def add_derived_features(frame: pd.DataFrame) -> None:
    surface_area = pd.to_numeric(frame["surface_area"], errors="coerce").fillna(0.0)
    sample_count = pd.to_numeric(frame["sample_count"], errors="coerce").fillna(0.0)
    total_area = frame.groupby("dataset")["surface_area"].transform("sum").replace(0.0, np.nan)
    mean_area = frame.groupby("dataset")["surface_area"].transform("mean").replace(0.0, np.nan)
    total_samples = frame.groupby("dataset")["sample_count"].transform("sum").replace(0.0, np.nan)
    frame["face_area_share"] = (surface_area / total_area).fillna(0.0)
    frame["face_area_to_object_mean"] = (surface_area / mean_area).fillna(0.0)
    frame["sample_share"] = (sample_count / total_samples).fillna(0.0)

    exposure = pd.to_numeric(frame["area_weighted_exposure"], errors="coerce").fillna(0.0)
    water_dose = pd.to_numeric(frame["area_weighted_water_dose"], errors="coerce").fillna(0.0)
    cleaning_dose = pd.to_numeric(frame["area_weighted_cleaning_dose"], errors="coerce").fillna(0.0)
    hotspot_score = pd.to_numeric(frame["area_weighted_hotspot_score"], errors="coerce").fillna(0.0)
    redeposition = pd.to_numeric(frame["area_weighted_redeposition"], errors="coerce").fillna(0.0)
    poor_drainage = pd.to_numeric(frame["area_weighted_poor_drainage"], errors="coerce").fillna(0.0)
    hiddenness = pd.to_numeric(frame["area_weighted_hiddenness"], errors="coerce").fillna(0.0)
    concavity = pd.to_numeric(frame["area_weighted_concavity"], errors="coerce").fillna(0.0)
    neighbor_count = pd.to_numeric(frame["neighbor_count"], errors="coerce").fillna(0.0)
    neighbor_area = pd.to_numeric(frame["neighbor_surface_area_mean"], errors="coerce").fillna(0.0)
    neighbor_hotspot = pd.to_numeric(frame["neighbor_area_weighted_hotspot_score_mean"], errors="coerce").fillna(0.0)
    neighbor_cleaning = pd.to_numeric(frame["neighbor_area_weighted_cleaning_dose_mean"], errors="coerce").fillna(0.0)
    neighbor_concavity = pd.to_numeric(frame["neighbor_area_weighted_concavity_mean"], errors="coerce").fillna(0.0)

    frame["water_to_exposure_ratio"] = safe_divide(water_dose, exposure)
    frame["cleaning_to_water_ratio"] = safe_divide(cleaning_dose, water_dose)
    frame["hotspot_to_cleaning_ratio"] = safe_divide(hotspot_score, cleaning_dose)
    frame["redeposition_to_cleaning_ratio"] = safe_divide(redeposition, cleaning_dose)
    frame["drainage_hiddenness_product"] = poor_drainage * hiddenness
    frame["hotspot_minus_neighbor_mean"] = hotspot_score - neighbor_hotspot
    frame["cleaning_minus_neighbor_mean"] = cleaning_dose - neighbor_cleaning
    frame["concavity_minus_neighbor_mean"] = concavity - neighbor_concavity
    frame["area_to_neighbor_mean"] = safe_divide(surface_area, neighbor_area)

    boundary_concave = pd.to_numeric(frame["boundary_concave_count"], errors="coerce").fillna(0.0)
    boundary_total = pd.to_numeric(frame["boundary_count"], errors="coerce").fillna(0.0)
    exact_concave = pd.to_numeric(frame["exact_boundary_concave_count"], errors="coerce").fillna(0.0)
    exact_total = pd.to_numeric(frame["exact_boundary_count"], errors="coerce").fillna(0.0)
    neighbor_mesh_concave = pd.to_numeric(frame["neighbor_mesh_concave_count"], errors="coerce").fillna(0.0)
    frame["boundary_concave_fraction"] = safe_divide(boundary_concave, boundary_total)
    frame["exact_boundary_concave_fraction"] = safe_divide(exact_concave, exact_total)
    frame["neighbor_mesh_concave_fraction"] = safe_divide(neighbor_mesh_concave, neighbor_count)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load_form_priors(path: Path) -> dict[str, dict[str, float]]:
    frame = pd.read_csv(path)
    priors: dict[str, dict[str, float]] = {}
    for _, row in frame.iterrows():
        form_type = str(row["form_type"])
        priors[form_type] = {
            "form_retained_particle_ratio": float(row.get("retained_particle_ratio", 0.0)),
            "form_retained_particle_share_total": float(row.get("retained_particle_share_total", 0.0)),
            "form_face_count": float(row.get("face_count", 0.0)),
            "form_dataset_count": float(row.get("dataset_count", 0.0)),
            "form_retention_rank_overall": float(row.get("form_retention_rank_overall", 0.0)),
        }
    return priors


def face_feature_lookup(
    face_frame: pd.DataFrame,
    form_priors: dict[str, dict[str, float]],
) -> dict[tuple[str, int], dict[str, float]]:
    lookup: dict[tuple[str, int], dict[str, float]] = {}
    for _, row in face_frame.iterrows():
        dataset = str(row["dataset"])
        face_id = int(row["face_id"])
        form_type = str(row["form_type"])
        lookup[(dataset, face_id)] = {
            "surface_area": float(row.get("surface_area", 0.0)),
            "area_weighted_concavity": float(row.get("area_weighted_concavity", 0.0)),
            "area_weighted_cleaning_dose": float(row.get("area_weighted_cleaning_dose", 0.0)),
            "area_weighted_hotspot_score": float(row.get("area_weighted_hotspot_score", 0.0)),
            **form_priors.get(form_type, empty_form_prior()),
        }
    return lookup


def load_context_features(
    path: Path,
    face_features: dict[tuple[str, int], dict[str, float]],
) -> dict[tuple[str, int], dict[str, float | int]]:
    rows = pd.read_csv(path)
    boundary_aggregates: dict[tuple[str, int], BoundaryAggregate] = {}
    exact_aggregates: dict[tuple[str, int], ExactBoundaryAggregate] = {}
    neighbor_aggregates: dict[tuple[str, int], NeighborFeatureAggregate] = {}
    for _, row in rows.iterrows():
        dataset = str(row["dataset"])
        face_id = int(row["dirty_face_id"])
        neighbor_face_id = int(row["neighbor_face_id"])
        key = (dataset, face_id)
        add_boundary(boundary_aggregates, key, row)
        add_exact_boundary(exact_aggregates, key, row)
        neighbor = face_features.get((dataset, neighbor_face_id))
        if neighbor is not None:
            neighbor_aggregates.setdefault(key, NeighborFeatureAggregate()).add(neighbor, row)

    context: dict[tuple[str, int], dict[str, float | int]] = {}
    for key in set(boundary_aggregates) | set(exact_aggregates) | set(neighbor_aggregates):
        values = empty_context_features()
        values.update(boundary_aggregates.get(key, BoundaryAggregate()).row())
        values.update(exact_aggregates.get(key, ExactBoundaryAggregate()).row())
        values.update(neighbor_aggregates.get(key, NeighborFeatureAggregate()).row())
        context[key] = values
    return context


def add_boundary(
    aggregates: dict[tuple[str, int], BoundaryAggregate],
    key: tuple[str, int],
    row: pd.Series,
) -> None:
    aggregate = aggregates.setdefault(key, BoundaryAggregate())
    aggregate.add(
        label=str(row.get("mesh_boundary_label", "")),
        sample_count=int(row.get("mesh_boundary_sample_edge_count", 0) or 0),
        angle_score=float(row.get("mesh_boundary_mean_angle_score", 0.0) or 0.0),
        concavity_score=float(row.get("mesh_boundary_mean_concavity_score", 0.0) or 0.0),
    )


def add_exact_boundary(
    aggregates: dict[tuple[str, int], ExactBoundaryAggregate],
    key: tuple[str, int],
    row: pd.Series,
) -> None:
    aggregate = aggregates.setdefault(key, ExactBoundaryAggregate())
    aggregate.add(
        convex_count=int(row.get("boundary_convex_edge_count", 0) or 0),
        concave_count=int(row.get("boundary_concave_edge_count", 0) or 0),
        neutral_count=int(row.get("boundary_neutral_edge_count", 0) or 0),
    )


def relevance_labels(frame: pd.DataFrame, *, top_k: int) -> pd.Series:
    labels = pd.Series(0, index=frame.index, dtype=int)
    for _, group in frame.groupby("dataset", sort=False):
        positive_group = group[group[TARGET_COLUMN] > 0]
        if positive_group.empty:
            continue
        ranked = positive_group.sort_values(
            [TARGET_COLUMN, "surface_area", "face_id"],
            ascending=[False, False, True],
        )
        count = len(ranked)
        top_10 = max(top_k + 1, int(np.ceil(count * 0.10)))
        top_25 = max(top_10 + 1, int(np.ceil(count * 0.25)))
        top_50 = max(top_25 + 1, int(np.ceil(count * 0.50)))
        for position, index in enumerate(ranked.index, start=1):
            if position <= min(top_k, count):
                labels.loc[index] = 4
            elif position <= min(top_10, count):
                labels.loc[index] = 3
            elif position <= min(top_25, count):
                labels.loc[index] = 2
            elif position <= min(top_50, count):
                labels.loc[index] = 1
    return labels


def limit_faces_per_object(
    frame: pd.DataFrame,
    *,
    max_faces_per_object: int,
    random_state: int,
) -> pd.DataFrame:
    if max_faces_per_object <= 0:
        return frame
    groups = []
    for dataset, group in frame.groupby("dataset", sort=False):
        if len(group) <= max_faces_per_object:
            groups.append(group)
            continue

        ranked = group.sort_values(
            [RELEVANCE_COLUMN, TARGET_COLUMN, "surface_area", "face_id"],
            ascending=[False, False, False, True],
        )
        must_keep = ranked[ranked[RELEVANCE_COLUMN] >= 3]
        remaining_slots = max(0, max_faces_per_object - len(must_keep))
        lower_ranked = ranked[ranked[RELEVANCE_COLUMN] <= 0]
        if remaining_slots >= len(lower_ranked):
            kept = ranked
        else:
            sampled = lower_ranked.sample(
                n=remaining_slots,
                random_state=random_state,
            )
            kept = pd.concat([must_keep, sampled], ignore_index=False)
        groups.append(kept.sort_values("face_id"))
        print(f"Capped {dataset}: {len(group)} -> {len(groups[-1])} faces")
    return pd.concat(groups, ignore_index=True)


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["form_type"] = prepared["form_type"].fillna("unknown").astype("category")
    for column in FEATURE_COLUMNS:
        if column == "form_type":
            continue
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce").fillna(0.0)
    prepared[TARGET_COLUMN] = pd.to_numeric(prepared[TARGET_COLUMN], errors="coerce").fillna(0.0)
    prepared[RELEVANCE_COLUMN] = pd.to_numeric(prepared[RELEVANCE_COLUMN], errors="coerce").fillna(0).astype(int)
    return prepared


def cross_val_predict(frame: pd.DataFrame, *, folds: int, random_state: int) -> pd.DataFrame:
    dataset_groups = frame["dataset"].astype(str)
    split_count = min(max(2, folds), dataset_groups.nunique())
    predictions = frame[["dataset", "face_id", "form_type", TARGET_COLUMN, RELEVANCE_COLUMN]].copy()
    predictions[SCORE_COLUMN] = np.nan
    predictions["fold"] = -1

    splitter = GroupKFold(n_splits=split_count)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(frame[FEATURE_COLUMNS], frame[RELEVANCE_COLUMN], dataset_groups),
        start=1,
    ):
        train_frame = sort_for_ranking(frame.iloc[train_index])
        test_frame = frame.iloc[test_index]
        model = make_ranker(random_state + fold)
        model.fit(
            train_frame[FEATURE_COLUMNS],
            train_frame[RELEVANCE_COLUMN],
            group=group_sizes(train_frame),
        )
        predictions.loc[test_frame.index, SCORE_COLUMN] = model.predict(test_frame[FEATURE_COLUMNS])
        predictions.loc[test_frame.index, "fold"] = fold
    return predictions


def holdout_predict(
    frame: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, SplitInfo]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("--holdout-test-size must be greater than 0 and less than 1.")
    dataset_groups = frame["dataset"].astype(str)
    if dataset_groups.nunique() < 2:
        raise ValueError("Holdout evaluation needs at least two STEP objects.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_index, test_index = next(
        splitter.split(frame[FEATURE_COLUMNS], frame[RELEVANCE_COLUMN], dataset_groups)
    )
    train_frame = sort_for_ranking(frame.iloc[train_index])
    test_frame = frame.iloc[test_index].copy()

    model = make_ranker(random_state)
    model.fit(
        train_frame[FEATURE_COLUMNS],
        train_frame[RELEVANCE_COLUMN],
        group=group_sizes(train_frame),
    )

    predictions = test_frame[["dataset", "face_id", "form_type", TARGET_COLUMN, RELEVANCE_COLUMN]].copy()
    predictions[SCORE_COLUMN] = model.predict(test_frame[FEATURE_COLUMNS])
    predictions["split"] = "test"
    split_info = SplitInfo(
        name="grouped holdout",
        train_rows=len(train_frame),
        test_rows=len(test_frame),
        train_datasets=train_frame["dataset"].nunique(),
        test_datasets=test_frame["dataset"].nunique(),
    )
    return predictions, split_info


def sort_for_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["dataset", "face_id"]).copy()


def group_sizes(frame: pd.DataFrame) -> list[int]:
    return frame.groupby("dataset", sort=False).size().astype(int).tolist()


def make_ranker(random_state: int) -> LGBMRanker:
    return LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=320,
        learning_rate=0.035,
        num_leaves=15,
        min_child_samples=8,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=0.2,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def evaluate(predictions: pd.DataFrame, *, top_k: int) -> dict[str, float]:
    hits = 0
    possible_hits = 0
    exact_matches = 0
    evaluated_datasets = 0
    precision_at_k: list[float] = []
    ndcg_at_k: list[float] = []
    for _, group in predictions.groupby("dataset", sort=False):
        positive_group = group[group[TARGET_COLUMN] > 0]
        if positive_group.empty:
            continue
        k = min(top_k, len(positive_group), len(group))
        actual_index = set(positive_group.nlargest(k, TARGET_COLUMN).index)
        predicted_index = set(group.nlargest(k, SCORE_COLUMN).index)
        overlap = len(actual_index & predicted_index)
        hits += overlap
        possible_hits += len(actual_index)
        evaluated_datasets += 1
        if overlap == len(actual_index):
            exact_matches += 1
        precision_at_k.append(overlap / k)
        ndcg_at_k.append(ndcg(group[RELEVANCE_COLUMN].to_numpy(), group[SCORE_COLUMN].to_numpy(), k))
    return {
        "mean_precision_at_k": float(np.mean(precision_at_k)) if precision_at_k else 0.0,
        "mean_ndcg_at_k": float(np.mean(ndcg_at_k)) if ndcg_at_k else 0.0,
        "exact_match_at_k": exact_matches / evaluated_datasets if evaluated_datasets else 0.0,
        "evaluated_datasets": float(evaluated_datasets),
        "hits": float(hits),
        "possible_hits": float(possible_hits),
    }


def ndcg(relevance: np.ndarray, scores: np.ndarray, k: int) -> float:
    predicted_order = np.argsort(scores)[::-1][:k]
    ideal_order = np.argsort(relevance)[::-1][:k]
    ideal = dcg(relevance[ideal_order])
    if ideal <= 1e-12:
        return 0.0
    return dcg(relevance[predicted_order]) / ideal


def dcg(relevance: np.ndarray) -> float:
    gains = (2.0 ** relevance.astype(float)) - 1.0
    discounts = np.log2(np.arange(len(relevance), dtype=float) + 2.0)
    return float(np.sum(gains / discounts))


def write_predictions(predictions: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.sort_values(
        ["dataset", SCORE_COLUMN],
        ascending=[True, False],
    ).to_csv(path, index=False)


def write_importance(model: LGBMRanker, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    importance = pd.DataFrame(
        {
            "feature": model.booster_.feature_name(),
            "gain_importance": model.booster_.feature_importance(importance_type="gain"),
            "split_importance": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values(["gain_importance", "split_importance"], ascending=False)
    importance.to_csv(path, index=False)


def print_summary(
    frame: pd.DataFrame,
    metrics: dict[str, float],
    args: argparse.Namespace,
    predictions_output: Path,
    split_info: SplitInfo | None,
) -> None:
    print("LightGBM LambdaRank hotspot model")
    print(f"Rows: {len(frame)}")
    print(f"Datasets: {frame['dataset'].nunique()}")
    print(f"Target: {TARGET_COLUMN}")
    print(f"Features: {', '.join(FEATURE_COLUMNS)}")
    print()
    if split_info is None:
        print(f"Evaluation: grouped {args.folds}-fold cross-validation")
    else:
        print(f"Evaluation: {split_info.name}")
        print(f"  Train: {split_info.train_datasets} STEP objects, {split_info.train_rows} rows")
        print(f"  Test:  {split_info.test_datasets} STEP objects, {split_info.test_rows} rows")
    print(f"Top-{args.top_k} ranking")
    print(f"  Mean top-{args.top_k} face accuracy: {metrics['mean_precision_at_k']:.4f}")
    print(f"  Exact object top-{args.top_k} match: {metrics['exact_match_at_k']:.4f}")
    print(f"  Mean NDCG@{args.top_k}:             {metrics['mean_ndcg_at_k']:.4f}")
    print(f"  Evaluated objects: {int(metrics['evaluated_datasets'])}")
    print(f"  Hits: {int(metrics['hits'])}/{int(metrics['possible_hits'])}")
    print()
    print(f"Wrote model: {args.model_output.resolve()}")
    print(f"Wrote predictions: {predictions_output.resolve()}")
    print(f"Wrote feature importance: {args.importance_output.resolve()}")


def empty_form_prior() -> dict[str, float]:
    return {
        "form_retained_particle_ratio": 0.0,
        "form_retained_particle_share_total": 0.0,
        "form_face_count": 0.0,
        "form_dataset_count": 0.0,
        "form_retention_rank_overall": 0.0,
    }


def empty_boundary_features() -> dict[str, float | int]:
    return {
        "boundary_count": 0,
        "boundary_concave_count": 0,
        "boundary_sharp_nonconcave_count": 0,
        "boundary_smooth_count": 0,
        "boundary_mean_angle_score": 0.0,
        "boundary_max_angle_score": 0.0,
        "boundary_mean_concavity_score": 0.0,
        "boundary_max_concavity_score": 0.0,
    }


def empty_exact_boundary_features() -> dict[str, float | int]:
    return {
        "exact_boundary_count": 0,
        "exact_boundary_convex_count": 0,
        "exact_boundary_concave_count": 0,
        "exact_boundary_neutral_count": 0,
    }


def empty_neighbor_features() -> dict[str, float | int]:
    return {
        "neighbor_count": 0,
        "neighbor_surface_area_mean": 0.0,
        "neighbor_surface_area_max": 0.0,
        "neighbor_form_retained_particle_ratio_mean": 0.0,
        "neighbor_form_retained_particle_ratio_max": 0.0,
        "neighbor_form_retained_particle_share_total_mean": 0.0,
        "neighbor_form_retained_particle_share_total_max": 0.0,
        "neighbor_form_retention_rank_mean": 0.0,
        "neighbor_form_retention_rank_min": 0.0,
        "neighbor_area_weighted_concavity_mean": 0.0,
        "neighbor_area_weighted_concavity_max": 0.0,
        "neighbor_area_weighted_cleaning_dose_mean": 0.0,
        "neighbor_area_weighted_cleaning_dose_min": 0.0,
        "neighbor_area_weighted_hotspot_score_mean": 0.0,
        "neighbor_area_weighted_hotspot_score_max": 0.0,
        "neighbor_boundary_concave_count": 0,
        "neighbor_boundary_convex_count": 0,
        "neighbor_boundary_neutral_count": 0,
        "neighbor_mesh_concave_count": 0,
        "neighbor_mesh_sharp_nonconcave_count": 0,
    }


def empty_context_features() -> dict[str, float | int]:
    return empty_boundary_features() | empty_exact_boundary_features() | empty_neighbor_features()


if __name__ == "__main__":
    raise SystemExit(main())
