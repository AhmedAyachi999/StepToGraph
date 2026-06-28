"""Predict stayed-dirty face probabilities for a new STEP file."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from features.cleaning_simulation import (
    CleaningSample,
    CleaningSimulationParameters,
    simulate_cleaning_on_mesh,
    step_to_surface_mesh,
)
from features.cleaning_simulation.math_utils import clamp, dot, norm, normalize, scale, sub
from features.cleaning_simulation.models import SurfaceMesh
from features.edge_classification import classify_step_edges
from features.hole_finding import FaceFormFinder


DEFAULT_HOTSPOT_MODEL_PATH = Path("cache") / "lightgbm_hotspot_model.txt"
DEFAULT_FORM_PRIOR_CSV = Path("data") / "training" / "cleaning_retention_wf05" / "form_particle_retention.csv"
TRAINED_WATER_FORCE = 0.5
TRAINED_WATER_DIRECTIONS = (
    (-1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 0.0, 1.0),
)
MODEL_FEATURE_COLUMNS = [
    "area_weighted_exposure",
    "sample_count",
    "area_weighted_cleaning_dose",
    "area_weighted_water_dose",
    "neighbor_area_weighted_cleaning_dose_mean",
    "area_weighted_poor_drainage",
    "area_weighted_hiddenness",
    "neighbor_area_weighted_hotspot_score_max",
    "neighbor_area_weighted_cleaning_dose_min",
    "neighbor_area_weighted_hotspot_score_mean",
]


@dataclass(frozen=True)
class HotspotPrediction:
    face_id: int
    form_type: str
    dirty_probability: float
    rank: int

    @property
    def predicted_retained_particle_ratio(self) -> float:
        return self.dirty_probability


@dataclass
class _FaceFeatureAggregate:
    sample_count: int = 0
    surface_area: float = 0.0
    exposure_area_sum: float = 0.0
    water_dose_area_sum: float = 0.0
    cleaning_dose_area_sum: float = 0.0
    hotspot_score_area_sum: float = 0.0
    redeposition_area_sum: float = 0.0
    poor_drainage_area_sum: float = 0.0
    concavity_area_sum: float = 0.0
    hiddenness_area_sum: float = 0.0

    def add_sample(self, sample: CleaningSample) -> None:
        area = max(sample.area, 0.0)
        self.sample_count += 1
        self.surface_area += area
        self.exposure_area_sum += sample.exposure * area
        self.water_dose_area_sum += sample.water_dose * area
        self.cleaning_dose_area_sum += sample.cleaning_dose * area
        self.hotspot_score_area_sum += sample.hotspot_score * area
        self.redeposition_area_sum += sample.redeposition * area
        self.poor_drainage_area_sum += sample.poor_drainage * area
        self.concavity_area_sum += sample.concavity * area
        self.hiddenness_area_sum += sample.hiddenness * area

    def area_weighted_mean(self, total: float) -> float:
        if self.surface_area <= 1e-12:
            return 0.0
        return total / self.surface_area

    def row(self, face_id: int, form_type: str) -> dict[str, Any]:
        return {
            "face_id": face_id,
            "form_type": form_type,
            "sample_count": self.sample_count,
            "surface_area": self.surface_area,
            "area_weighted_exposure": self.area_weighted_mean(self.exposure_area_sum),
            "area_weighted_water_dose": self.area_weighted_mean(self.water_dose_area_sum),
            "area_weighted_cleaning_dose": self.area_weighted_mean(self.cleaning_dose_area_sum),
            "area_weighted_hotspot_score": self.area_weighted_mean(self.hotspot_score_area_sum),
            "area_weighted_redeposition": self.area_weighted_mean(self.redeposition_area_sum),
            "area_weighted_poor_drainage": self.area_weighted_mean(self.poor_drainage_area_sum),
            "area_weighted_concavity": self.area_weighted_mean(self.concavity_area_sum),
            "area_weighted_hiddenness": self.area_weighted_mean(self.hiddenness_area_sum),
        }


@dataclass
class _BoundaryFeatureAggregate:
    boundary_count: int = 0
    concave_count: int = 0
    sharp_nonconcave_count: int = 0
    smooth_count: int = 0
    angle_score_sum: float = 0.0
    concavity_score_sum: float = 0.0
    max_angle_score: float = 0.0
    max_concavity_score: float = 0.0

    def add(self, angle_score: float, concavity_score: float) -> None:
        self.boundary_count += 1
        self.angle_score_sum += angle_score
        self.concavity_score_sum += concavity_score
        self.max_angle_score = max(self.max_angle_score, angle_score)
        self.max_concavity_score = max(self.max_concavity_score, concavity_score)
        if concavity_score >= 0.05:
            self.concave_count += 1
        elif angle_score >= 0.05:
            self.sharp_nonconcave_count += 1
        else:
            self.smooth_count += 1

    def row(self) -> dict[str, float | int]:
        if self.boundary_count == 0:
            return _empty_boundary_features()
        return {
            "boundary_count": self.boundary_count,
            "boundary_concave_count": self.concave_count,
            "boundary_sharp_nonconcave_count": self.sharp_nonconcave_count,
            "boundary_smooth_count": self.smooth_count,
            "boundary_mean_angle_score": self.angle_score_sum / self.boundary_count,
            "boundary_max_angle_score": self.max_angle_score,
            "boundary_mean_concavity_score": self.concavity_score_sum / self.boundary_count,
            "boundary_max_concavity_score": self.max_concavity_score,
        }


@dataclass
class _ExactBoundaryFeatureAggregate:
    count: int = 0
    convex_count: int = 0
    concave_count: int = 0
    neutral_count: int = 0

    def add(self, convexity: str) -> None:
        self.count += 1
        if convexity == "convex":
            self.convex_count += 1
        elif convexity == "concave":
            self.concave_count += 1
        else:
            self.neutral_count += 1

    def row(self) -> dict[str, float | int]:
        return {
            "exact_boundary_count": self.count,
            "exact_boundary_convex_count": self.convex_count,
            "exact_boundary_concave_count": self.concave_count,
            "exact_boundary_neutral_count": self.neutral_count,
        }


@dataclass
class _NeighborFeatureAggregate:
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

    def add(
        self,
        neighbor: dict[str, Any],
        form_priors: dict[str, dict[str, float]],
        mesh_boundary_label: str,
        exact_boundary_label: str,
    ) -> None:
        self.count += 1
        form_type = str(neighbor.get("form_type", "unknown"))
        priors = _form_prior_features(form_type, form_priors)
        surface_area = _float_value(neighbor.get("surface_area"))
        form_ratio = _float_value(priors.get("form_retained_particle_ratio"))
        form_share = _float_value(priors.get("form_retained_particle_share_total"))
        form_rank = _float_value(priors.get("form_retention_rank_overall"))
        concavity = _float_value(neighbor.get("area_weighted_concavity"))
        cleaning_dose = _float_value(neighbor.get("area_weighted_cleaning_dose"))
        hotspot_score = _float_value(neighbor.get("area_weighted_hotspot_score"))

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

        if exact_boundary_label == "concave":
            self.boundary_concave_count += 1
        elif exact_boundary_label == "convex":
            self.boundary_convex_count += 1
        elif exact_boundary_label == "neutral":
            self.boundary_neutral_count += 1

        if mesh_boundary_label == "mesh_concave":
            self.mesh_concave_count += 1
        elif mesh_boundary_label == "mesh_sharp_nonconcave":
            self.mesh_sharp_nonconcave_count += 1

    def row(self) -> dict[str, float | int]:
        if self.count == 0:
            return _empty_neighbor_features()
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


def predict_step_hotspots(
    step_path: str | Path,
    *,
    model_path: str | Path = DEFAULT_HOTSPOT_MODEL_PATH,
    parameters: CleaningSimulationParameters | None = None,
) -> list[HotspotPrediction]:
    """Predict face-level hotspot risk for a STEP file with the trained LightGBM model."""
    lightgbm, pandas = _load_ml_dependencies()
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"LightGBM model not found: {model_path}")

    feature_rows = step_hotspot_feature_rows(step_path, parameters=parameters)
    if not feature_rows:
        return []

    frame = pandas.DataFrame(feature_rows)
    booster = lightgbm.Booster(model_file=str(model_path))
    feature_columns = booster.feature_name() or MODEL_FEATURE_COLUMNS
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing model feature columns: {', '.join(missing)}")

    _prepare_frame_for_lightgbm(frame, booster, feature_columns, pandas)
    raw_predictions = booster.predict(frame[feature_columns])
    ranked_rows = sorted(
        zip(feature_rows, raw_predictions),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [
        HotspotPrediction(
            face_id=int(row["face_id"]),
            form_type=str(row["form_type"]),
            dirty_probability=float(prediction),
            rank=rank,
        )
        for rank, (row, prediction) in enumerate(ranked_rows, start=1)
    ]


def step_hotspot_feature_rows(
    step_path: str | Path,
    *,
    parameters: CleaningSimulationParameters | None = None,
) -> list[dict[str, Any]]:
    parameters = parameters or default_prediction_parameters()
    step_path = Path(step_path)
    form_by_face = {
        form.face_id: form.form_type
        for form in FaceFormFinder().find_step(step_path).forms
    }
    mesh = step_to_surface_mesh(step_path, parameters=parameters)
    result = simulate_cleaning_on_mesh(mesh, parameters=parameters)
    aggregates: dict[int, _FaceFeatureAggregate] = {}
    face_neighbors, boundary_features, mesh_boundary_labels = _mesh_context_features(mesh)
    exact_boundary_features, exact_boundary_labels = _exact_boundary_context(step_path)
    form_priors = _load_form_priors()

    for sample in result.samples:
        aggregates.setdefault(sample.face_id, _FaceFeatureAggregate()).add_sample(sample)

    rows_by_face: dict[int, dict[str, Any]] = {}
    for face_id, aggregate in sorted(aggregates.items()):
        form_type = form_by_face.get(face_id, "unknown")
        row = aggregate.row(face_id, form_type)
        row.update(_form_prior_features(form_type, form_priors))
        row.update(boundary_features.get(face_id, _empty_boundary_features()))
        row.update(exact_boundary_features.get(face_id, _empty_exact_boundary_features()))
        rows_by_face[face_id] = row
    neighbor_features = _neighbor_features(
        rows_by_face,
        face_neighbors,
        mesh_boundary_labels,
        exact_boundary_labels,
        form_priors,
    )
    rows = []
    for face_id, row in sorted(rows_by_face.items()):
        row.update(neighbor_features.get(face_id, _empty_neighbor_features()))
        rows.append(row)
    _add_derived_features(rows)
    return rows


def default_prediction_parameters() -> CleaningSimulationParameters:
    return CleaningSimulationParameters(
        water_directions=TRAINED_WATER_DIRECTIONS,
        water_force=TRAINED_WATER_FORCE,
    )


def _add_derived_features(rows: list[dict[str, Any]]) -> None:
    total_area = sum(_float_value(row.get("surface_area")) for row in rows)
    total_samples = sum(_float_value(row.get("sample_count")) for row in rows)
    mean_area = total_area / len(rows) if rows else 0.0
    for row in rows:
        surface_area = _float_value(row.get("surface_area"))
        sample_count = _float_value(row.get("sample_count"))
        exposure = _float_value(row.get("area_weighted_exposure"))
        water_dose = _float_value(row.get("area_weighted_water_dose"))
        cleaning_dose = _float_value(row.get("area_weighted_cleaning_dose"))
        hotspot_score = _float_value(row.get("area_weighted_hotspot_score"))
        redeposition = _float_value(row.get("area_weighted_redeposition"))
        poor_drainage = _float_value(row.get("area_weighted_poor_drainage"))
        hiddenness = _float_value(row.get("area_weighted_hiddenness"))
        concavity = _float_value(row.get("area_weighted_concavity"))
        neighbor_count = _float_value(row.get("neighbor_count"))
        neighbor_area = _float_value(row.get("neighbor_surface_area_mean"))
        neighbor_hotspot = _float_value(row.get("neighbor_area_weighted_hotspot_score_mean"))
        neighbor_cleaning = _float_value(row.get("neighbor_area_weighted_cleaning_dose_mean"))
        neighbor_concavity = _float_value(row.get("neighbor_area_weighted_concavity_mean"))
        boundary_count = _float_value(row.get("boundary_count"))
        boundary_concave = _float_value(row.get("boundary_concave_count"))
        exact_boundary_count = _float_value(row.get("exact_boundary_count"))
        exact_boundary_concave = _float_value(row.get("exact_boundary_concave_count"))
        neighbor_mesh_concave = _float_value(row.get("neighbor_mesh_concave_count"))

        row["face_area_share"] = _safe_divide(surface_area, total_area)
        row["face_area_to_object_mean"] = _safe_divide(surface_area, mean_area)
        row["sample_share"] = _safe_divide(sample_count, total_samples)
        row["water_to_exposure_ratio"] = _safe_divide(water_dose, exposure)
        row["cleaning_to_water_ratio"] = _safe_divide(cleaning_dose, water_dose)
        row["hotspot_to_cleaning_ratio"] = _safe_divide(hotspot_score, cleaning_dose)
        row["redeposition_to_cleaning_ratio"] = _safe_divide(redeposition, cleaning_dose)
        row["drainage_hiddenness_product"] = poor_drainage * hiddenness
        row["hotspot_minus_neighbor_mean"] = hotspot_score - neighbor_hotspot
        row["cleaning_minus_neighbor_mean"] = cleaning_dose - neighbor_cleaning
        row["concavity_minus_neighbor_mean"] = concavity - neighbor_concavity
        row["area_to_neighbor_mean"] = _safe_divide(surface_area, neighbor_area)
        row["boundary_concave_fraction"] = _safe_divide(boundary_concave, boundary_count)
        row["exact_boundary_concave_fraction"] = _safe_divide(exact_boundary_concave, exact_boundary_count)
        row["neighbor_mesh_concave_fraction"] = _safe_divide(neighbor_mesh_concave, neighbor_count)


def _mesh_context_features(
    mesh: SurfaceMesh,
) -> tuple[dict[int, set[int]], dict[int, dict[str, float | int]], dict[tuple[int, int], str]]:
    aggregates: dict[int, _BoundaryFeatureAggregate] = {}
    face_neighbors: dict[int, set[int]] = {}
    pair_scores: dict[tuple[int, int], tuple[float, float]] = {}
    for triangle in mesh.triangles:
        for neighbor_id in mesh.neighbors.get(triangle.id, []):
            if neighbor_id <= triangle.id:
                continue
            neighbor = mesh.triangles[neighbor_id]
            if triangle.face_id == neighbor.face_id:
                continue
            face_neighbors.setdefault(triangle.face_id, set()).add(neighbor.face_id)
            face_neighbors.setdefault(neighbor.face_id, set()).add(triangle.face_id)

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
            concavity_score = clamp(concavity_score)
            aggregates.setdefault(triangle.face_id, _BoundaryFeatureAggregate()).add(
                angle_score,
                concavity_score,
            )
            aggregates.setdefault(neighbor.face_id, _BoundaryFeatureAggregate()).add(
                angle_score,
                concavity_score,
            )
            pair_key = _face_pair_key(triangle.face_id, neighbor.face_id)
            previous_angle, previous_concavity = pair_scores.get(pair_key, (0.0, 0.0))
            pair_scores[pair_key] = (
                max(previous_angle, angle_score),
                max(previous_concavity, concavity_score),
            )
    pair_labels = {
        key: _mesh_boundary_label(angle_score, concavity_score)
        for key, (angle_score, concavity_score) in pair_scores.items()
    }
    return face_neighbors, {face_id: aggregate.row() for face_id, aggregate in aggregates.items()}, pair_labels


def _exact_boundary_context(
    step_path: Path,
) -> tuple[dict[int, dict[str, float | int]], dict[tuple[int, int], str]]:
    try:
        analysis = classify_step_edges(step_path)
    except Exception:
        return {}, {}
    aggregates: dict[int, _ExactBoundaryFeatureAggregate] = {}
    pair_labels: dict[tuple[int, int], str] = {}
    for edge in analysis.edges:
        aggregates.setdefault(edge.source_face, _ExactBoundaryFeatureAggregate()).add(edge.convexity)
        aggregates.setdefault(edge.target_face, _ExactBoundaryFeatureAggregate()).add(edge.convexity)
        pair_labels[_face_pair_key(edge.source_face, edge.target_face)] = edge.convexity
    return {face_id: aggregate.row() for face_id, aggregate in aggregates.items()}, pair_labels


def _neighbor_features(
    rows_by_face: dict[int, dict[str, Any]],
    face_neighbors: dict[int, set[int]],
    mesh_boundary_labels: dict[tuple[int, int], str],
    exact_boundary_labels: dict[tuple[int, int], str],
    form_priors: dict[str, dict[str, float]],
) -> dict[int, dict[str, float | int]]:
    aggregates: dict[int, _NeighborFeatureAggregate] = {}
    for face_id, neighbor_ids in face_neighbors.items():
        aggregate = aggregates.setdefault(face_id, _NeighborFeatureAggregate())
        for neighbor_id in sorted(neighbor_ids):
            neighbor = rows_by_face.get(neighbor_id)
            if neighbor is None:
                continue
            aggregate.add(
                neighbor,
                form_priors,
                mesh_boundary_labels.get(_face_pair_key(face_id, neighbor_id), ""),
                exact_boundary_labels.get(_face_pair_key(face_id, neighbor_id), ""),
            )
    return {face_id: aggregate.row() for face_id, aggregate in aggregates.items()}


def _face_pair_key(face_a: int, face_b: int) -> tuple[int, int]:
    return (face_a, face_b) if face_a <= face_b else (face_b, face_a)


def _mesh_boundary_label(angle_score: float, concavity_score: float) -> str:
    if concavity_score >= 0.05:
        return "mesh_concave"
    if angle_score >= 0.05:
        return "mesh_sharp_nonconcave"
    return "mesh_smooth_or_coplanar"


def _load_form_priors() -> dict[str, dict[str, float]]:
    if not DEFAULT_FORM_PRIOR_CSV.exists():
        return {}
    priors: dict[str, dict[str, float]] = {}
    with DEFAULT_FORM_PRIOR_CSV.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            form_type = str(row.get("form_type", "unknown"))
            priors[form_type] = {
                "form_retained_particle_ratio": _float_value(row.get("retained_particle_ratio")),
                "form_retained_particle_share_total": _float_value(row.get("retained_particle_share_total")),
                "form_face_count": _float_value(row.get("face_count")),
                "form_dataset_count": _float_value(row.get("dataset_count")),
                "form_retention_rank_overall": _float_value(row.get("form_retention_rank_overall")),
            }
    return priors


def _form_prior_features(form_type: str, priors: dict[str, dict[str, float]]) -> dict[str, float]:
    return priors.get(
        form_type,
        {
            "form_retained_particle_ratio": 0.0,
            "form_retained_particle_share_total": 0.0,
            "form_face_count": 0.0,
            "form_dataset_count": 0.0,
            "form_retention_rank_overall": 0.0,
        },
    )


def _empty_boundary_features() -> dict[str, float | int]:
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


def _empty_exact_boundary_features() -> dict[str, float | int]:
    return {
        "exact_boundary_count": 0,
        "exact_boundary_convex_count": 0,
        "exact_boundary_concave_count": 0,
        "exact_boundary_neutral_count": 0,
    }


def _empty_neighbor_features() -> dict[str, float | int]:
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


def _load_ml_dependencies() -> tuple[Any, Any]:
    try:
        import lightgbm
        import pandas
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM hotspot prediction requires the ML dependencies. "
            "Install requirements-ml.txt into the Python environment that runs desktop_ui.py."
        ) from exc
    return lightgbm, pandas


def _prepare_frame_for_lightgbm(frame: Any, booster: Any, feature_columns: list[str], pandas: Any) -> None:
    categories = getattr(booster, "pandas_categorical", None) or []
    category_by_column = {
        column: category_values
        for column, category_values in zip(
            [column for column in feature_columns if column == "form_type"],
            categories,
        )
    }
    if "form_type" in feature_columns:
        frame["form_type"] = pandas.Categorical(
            frame["form_type"].fillna("unknown").astype(str),
            categories=category_by_column.get("form_type"),
        )
    for column in feature_columns:
        if column == "form_type":
            continue
        frame[column] = pandas.to_numeric(frame[column], errors="coerce")


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _float_value(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return 0.0
    return numerator / denominator
