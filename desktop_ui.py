import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from typing import Any

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCC.Core.gp import gp_Pnt
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Display.tkDisplay import tkViewer3d
from OCC.Extend.DataExchange import read_step_file

from features.cleaning_simulation import (
    CleaningSample,
    CleaningSimulationParameters,
    CleaningSimulationResult,
    SurfaceMesh,
    simulate_cleaning_on_mesh,
    step_to_surface_mesh,
)
from features.edge_classification import edge_shape_groups
from features.hotspot_prediction import DEFAULT_HOTSPOT_MODEL_PATH, HotspotPrediction, predict_step_hotspots
from features.hole_finding import (
    FACE_FORM_LABELS,
    FACE_FORM_TYPES,
    FaceFormAnalysis,
    FaceFormFinder,
    InwardCylinderCandidate,
    InwardCylinderHeuristic,
)


DEFAULT_STEP_FILE = Path("step_datasets/perfect_L_no_holes.step")
BASE_COLOR = Quantity_Color(0.72, 0.75, 0.80, Quantity_TOC_RGB)
INWARD_CYLINDER_COLOR = Quantity_Color(0.95, 0.44, 0.08, Quantity_TOC_RGB)
DUST_COLOR = Quantity_Color(0.32, 0.25, 0.17, Quantity_TOC_RGB)
MODEL_HOTSPOT_COLOR = Quantity_Color(1.0, 0.0, 0.72, Quantity_TOC_RGB)
RETAINED_PARTICLE_RANK_COLORS = (
    Quantity_Color(1.0, 0.05, 0.0, Quantity_TOC_RGB),
    Quantity_Color(1.0, 0.55, 0.0, Quantity_TOC_RGB),
    Quantity_Color(1.0, 0.90, 0.0, Quantity_TOC_RGB),
    Quantity_Color(0.30, 0.85, 0.10, Quantity_TOC_RGB),
    Quantity_Color(0.0, 0.72, 0.90, Quantity_TOC_RGB),
    Quantity_Color(0.05, 0.28, 1.0, Quantity_TOC_RGB),
    Quantity_Color(0.55, 0.20, 0.90, Quantity_TOC_RGB),
    Quantity_Color(0.40, 0.40, 0.40, Quantity_TOC_RGB),
)
HOTSPOT_THRESHOLD = 0.65
LEARNED_HOTSPOT_TOP_K = 4
RETAINED_PARTICLE_RANK_LIMIT = len(RETAINED_PARTICLE_RANK_COLORS)
RETAINED_PARTICLE_ACCEPTED_RANK_LIMIT = 4
REFERENCE_WATER_FORCE = 0.5
REMAINING_DUST_THRESHOLD = 0.02
MAX_PARTICLE_MARKERS = 1200
DIRECTION_OPTIONS = (
    ("Top", (0.0, 0.0, -1.0)),
    ("Bottom", (0.0, 0.0, 1.0)),
    ("From +X", (-1.0, 0.0, 0.0)),
    ("From -X", (1.0, 0.0, 0.0)),
    ("From +Y", (0.0, -1.0, 0.0)),
    ("From -Y", (0.0, 1.0, 0.0)),
)
SPRAY_DIRECTION_OPTIONS = (
    ("+X", "Spray +X", (-1.0, 0.0, 0.0)),
    ("-X", "Spray -X", (1.0, 0.0, 0.0)),
    ("+Y", "Spray +Y", (0.0, -1.0, 0.0)),
    ("-Y", "Spray -Y", (0.0, 1.0, 0.0)),
    ("+Z", "Spray +Z", (0.0, 0.0, -1.0)),
    ("-Z", "Spray -Z", (0.0, 0.0, 1.0)),
)
EDGE_COLORS = {
    "convex": Quantity_Color(0.0, 0.65, 0.18, Quantity_TOC_RGB),
    "concave": Quantity_Color(0.02, 0.18, 0.95, Quantity_TOC_RGB),
    "neutral": Quantity_Color(0.55, 0.58, 0.62, Quantity_TOC_RGB),
}
FACE_FORM_COLORS = {
    "plane": Quantity_Color(0.86, 0.76, 0.18, Quantity_TOC_RGB),
    "cylinder": Quantity_Color(0.08, 0.58, 0.72, Quantity_TOC_RGB),
    "cone": Quantity_Color(0.72, 0.28, 0.50, Quantity_TOC_RGB),
    "sphere": Quantity_Color(0.32, 0.66, 0.28, Quantity_TOC_RGB),
    "torus": Quantity_Color(0.55, 0.38, 0.78, Quantity_TOC_RGB),
    "other": Quantity_Color(0.74, 0.36, 0.12, Quantity_TOC_RGB),
}


def view_step_file(filename: str | Path) -> None:
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"STEP file not found: {path}")

    shape = read_step_file(str(path))
    edge_groups = edge_shape_groups(path)
    cylinder_summary, cylinder_shapes = _inward_cylinder_data(path)
    form_summary, form_shape_groups, face_shapes = _face_form_data(path)
    feature_summary = f"{cylinder_summary}   {form_summary}"
    mesh_parameters = CleaningSimulationParameters()
    cleaning_cache: dict[Any, SurfaceMesh | CleaningSimulationResult] = {}

    root = tk.Tk()
    root.title(f"STEP Hotspot Viewer - {path.name}")
    root.geometry("1180x820")

    toolbar = tk.Frame(root, padx=8, pady=6)
    toolbar.pack(side=tk.TOP, fill=tk.X)
    statusbar = tk.Frame(root, padx=8, pady=4)
    statusbar.pack(side=tk.BOTTOM, fill=tk.X)
    status = tk.StringVar(value=_status_text(path, edge_groups, feature_summary, "solid only"))
    model_status = tk.StringVar(value=f"Model: {DEFAULT_HOTSPOT_MODEL_PATH}")
    water_force_var = tk.StringVar(value=f"{REFERENCE_WATER_FORCE:.3f}")
    cleaning_view_active = {"value": False}
    cleaning_refresh_after_id = {"id": None}

    viewer = tkViewer3d(root)
    viewer.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    viewer.wait_visibility()
    display = viewer._display

    def choose_step_file() -> None:
        selected = filedialog.askopenfilename(
            title="Open STEP file",
            filetypes=(
                ("STEP files", "*.step *.stp *.STEP *.STP"),
                ("All files", "*.*"),
            ),
        )
        if not selected:
            return
        try:
            load_step_file(Path(selected))
        except Exception as exc:
            status.set(f"{Path(selected).name}   upload error: {exc}")

    def load_step_file(new_path: Path) -> None:
        nonlocal path, shape, edge_groups
        nonlocal cylinder_summary, cylinder_shapes
        nonlocal form_summary, form_shape_groups, face_shapes
        nonlocal feature_summary, cleaning_cache

        if not new_path.is_file():
            raise FileNotFoundError(f"STEP file not found: {new_path}")

        status.set(f"{new_path.name}   loading STEP file...")
        model_status.set(f"Model: {DEFAULT_HOTSPOT_MODEL_PATH}")
        root.update_idletasks()

        new_shape = read_step_file(str(new_path))
        new_edge_groups = edge_shape_groups(new_path)
        new_cylinder_summary, new_cylinder_shapes = _inward_cylinder_data(new_path)
        new_form_summary, new_form_shape_groups, new_face_shapes = _face_form_data(new_path)

        path = new_path
        shape = new_shape
        edge_groups = new_edge_groups
        cylinder_summary = new_cylinder_summary
        cylinder_shapes = new_cylinder_shapes
        form_summary = new_form_summary
        form_shape_groups = new_form_shape_groups
        face_shapes = new_face_shapes
        feature_summary = f"{cylinder_summary}   {form_summary}"
        cleaning_cache = {}
        cleaning_view_active["value"] = False
        if cleaning_refresh_after_id["id"] is not None:
            root.after_cancel(cleaning_refresh_after_id["id"])
            cleaning_refresh_after_id["id"] = None

        root.title(f"STEP Hotspot Viewer - {path.name}")
        display.EraseAll()
        display.DisplayShape(shape, color=BASE_COLOR, transparency=0.15, update=False)
        display.FitAll()
        status.set(_status_text(path, edge_groups, feature_summary, "solid only"))
        display.Repaint()

    def show_colored_edges() -> None:
        display.EraseAll()
        display.DisplayShape(shape, color=BASE_COLOR, transparency=0.55, update=False)
        for edge_type, color in EDGE_COLORS.items():
            for ais in display.DisplayShape(edge_groups[edge_type], color=color, update=False):
                display.Context.SetWidth(ais, 4.0, False)
        status.set(_status_text(path, edge_groups, feature_summary, "colored edges"))
        display.Repaint()

    def show_inward_cylinders() -> None:
        display.EraseAll()
        display.DisplayShape(shape, color=BASE_COLOR, transparency=0.70, update=False)
        if cylinder_shapes:
            display.DisplayShape(cylinder_shapes, color=INWARD_CYLINDER_COLOR, transparency=0.05, update=False)
        status.set(_status_text(path, edge_groups, feature_summary, "inward cylinders"))
        display.Repaint()

    def show_face_form(form_type: str) -> None:
        display.EraseAll()
        display.DisplayShape(shape, color=BASE_COLOR, transparency=0.72, update=False)
        form_shapes = form_shape_groups.get(form_type, [])
        if form_shapes:
            display.DisplayShape(form_shapes, color=FACE_FORM_COLORS[form_type], transparency=0.02, update=False)
        label = FACE_FORM_LABELS[form_type]
        status.set(_status_text(path, edge_groups, feature_summary, f"{label} faces"))
        display.Repaint()

    def surface_mesh() -> SurfaceMesh:
        cached = cleaning_cache.get("mesh")
        if isinstance(cached, SurfaceMesh):
            return cached
        status.set(_status_text(path, edge_groups, feature_summary, "meshing surface samples..."))
        root.update_idletasks()
        mesh = step_to_surface_mesh(path, parameters=mesh_parameters)
        cleaning_cache["mesh"] = mesh
        return mesh

    def active_water_directions() -> tuple[tuple[float, float, float], ...]:
        return tuple(direction for _, _, direction in SPRAY_DIRECTION_OPTIONS)

    def selected_cleaning_parameters() -> CleaningSimulationParameters:
        water_force = _parse_water_force(water_force_var.get())
        if water_force is None:
            raise ValueError("enter a numeric water force from 0.000 to 1.000")
        water_force = _clamp(water_force, 0.0, 1.0)
        return CleaningSimulationParameters(
            water_directions=active_water_directions(),
            water_force=water_force,
        )

    def reference_cleaning_parameters() -> CleaningSimulationParameters:
        return CleaningSimulationParameters(
            water_directions=active_water_directions(),
            water_force=REFERENCE_WATER_FORCE,
        )

    def cleaning_result() -> CleaningSimulationResult:
        return cleaning_result_for(selected_cleaning_parameters(), "simulating cleaning...")

    def reference_cleaning_result() -> CleaningSimulationResult:
        return cleaning_result_for(
            reference_cleaning_parameters(),
            f"simulating retained particles at force={REFERENCE_WATER_FORCE:.3f}...",
        )

    def cleaning_result_for(parameters: CleaningSimulationParameters, progress_text: str) -> CleaningSimulationResult:
        cache_key = _cleaning_cache_key(parameters)
        cached = cleaning_cache.get(cache_key)
        if isinstance(cached, CleaningSimulationResult):
            return cached
        status.set(_status_text(path, edge_groups, feature_summary, progress_text))
        root.update_idletasks()
        result = simulate_cleaning_on_mesh(surface_mesh(), parameters=parameters)
        cleaning_cache[cache_key] = result
        return result

    def show_particles() -> None:
        try:
            mesh = surface_mesh()
        except Exception as exc:
            status.set(_status_text(path, edge_groups, feature_summary, f"particle error: {exc}"))
            return

        display.EraseAll()
        display.DisplayShape(shape, color=BASE_COLOR, transparency=0.62, update=False)
        radius = _marker_radius([triangle.point for triangle in mesh.triangles], scale=0.004)
        for triangle in _limited_sequence(mesh.triangles, MAX_PARTICLE_MARKERS):
            display.DisplayShape(_sphere_at(triangle.point, radius), color=DUST_COLOR, update=False)
        shown = min(len(mesh.triangles), MAX_PARTICLE_MARKERS)
        status.set(
            _status_text(
                path,
                edge_groups,
                feature_summary,
                f"particles introduced: {shown}/{len(mesh.triangles)} surface samples",
            )
        )
        display.Repaint()

    def show_cleaning_result() -> None:
        cleaning_view_active["value"] = True
        try:
            result = cleaning_result()
        except Exception as exc:
            status.set(_status_text(path, edge_groups, feature_summary, f"cleaning error: {exc}"))
            return

        _clear_display(display)
        display.DisplayShape(shape, color=BASE_COLOR, transparency=0.82, update=False)

        retained_scores = _face_retained_particle_scores(result)
        retained_faces = _selected_retained_particle_faces(retained_scores)
        retained_face_set = set(retained_faces)
        for rank_index, face_id in enumerate(retained_faces):
            face_shape = face_shapes.get(face_id)
            if face_shape is None:
                continue
            display.DisplayShape(
                face_shape,
                color=_retained_rank_color(rank_index),
                transparency=0.0,
                update=False,
            )

        remaining_samples = _remaining_particles(result)
        radius = _marker_radius([sample.point for sample in result.samples], scale=0.006)
        for sample in _limited_sequence(remaining_samples, MAX_PARTICLE_MARKERS):
            marker_radius = radius * (0.65 + sample.remaining_dust)
            marker_color = (
                _retained_sample_color(sample, retained_faces)
                if sample.face_id in retained_face_set
                else _remaining_dust_color(sample.remaining_dust)
            )
            display.DisplayShape(
                _sphere_at(sample.point, marker_radius),
                color=marker_color,
                update=False,
            )

        parameters = selected_cleaning_parameters()
        status.set(
            _status_text(
                path,
                edge_groups,
                feature_summary,
                (
                    f"simulation force={parameters.water_force:.3f} dirs={_direction_summary(parameters)}: "
                    f"{len(remaining_samples)} particles stayed, "
                    f"{len(retained_faces)} retained faces, "
                    f"order={_retained_rank_summary(retained_faces, retained_scores)}"
                ),
            )
        )
        display.Repaint()

    def show_simulation_hotspots() -> None:
        cleaning_view_active["value"] = False
        try:
            result = reference_cleaning_result()
        except Exception as exc:
            status.set(_status_text(path, edge_groups, feature_summary, f"simulation error: {exc}"))
            return

        _clear_display(display)
        display.DisplayShape(shape, color=BASE_COLOR, transparency=0.84, update=False)

        retained_scores = _face_retained_particle_scores(result)
        retained_faces = _selected_retained_particle_faces(retained_scores)
        for rank_index, face_id in enumerate(retained_faces):
            face_shape = face_shapes.get(face_id)
            if face_shape is None:
                continue
            display.DisplayShape(
                face_shape,
                color=_retained_rank_color(rank_index),
                transparency=0.0,
                update=False,
            )

        parameters = reference_cleaning_parameters()
        status.set(
            _status_text(
                path,
                edge_groups,
                feature_summary,
                (
                    f"too many particles stayed: {len(retained_faces)} faces through green, "
                    f"force={parameters.water_force:.3f}, dirs={_direction_summary(parameters)}, "
                    f"order={_retained_rank_summary(retained_faces, retained_scores)}"
                ),
            )
        )
        display.Repaint()

    def show_model_vs_retained_particles() -> None:
        cleaning_view_active["value"] = False
        model_status.set("Learned model: predicting and comparing with retained particles...")
        status.set(_status_text(path, edge_groups, feature_summary, "comparing model vs retained particles..."))
        root.update_idletasks()

        try:
            predictions = predict_step_hotspots(path, model_path=DEFAULT_HOTSPOT_MODEL_PATH)
            result = reference_cleaning_result()
        except Exception as exc:
            model_status.set(f"Comparison error: {exc}")
            status.set(_status_text(path, edge_groups, feature_summary, f"comparison error: {exc}"))
            return

        selected_predictions = _selected_learned_hotspots(predictions)
        retained_scores = _face_retained_particle_scores(result)
        retained_faces = _selected_retained_particle_faces(retained_scores)
        accepted_retained_faces = _accepted_retained_particle_faces(retained_faces)
        retained_face_set = set(retained_faces)
        accepted_retained_face_set = set(accepted_retained_faces)
        predicted_face_set = {prediction.face_id for prediction in selected_predictions}
        overlap_count = len(accepted_retained_face_set & predicted_face_set)
        max_model_score = max((prediction.predicted_retained_particle_ratio for prediction in predictions), default=0.0)

        _clear_display(display)
        display.DisplayShape(shape, color=BASE_COLOR, transparency=0.88, update=False)
        for rank_index, face_id in enumerate(retained_faces):
            face_shape = face_shapes.get(face_id)
            if face_shape is not None:
                display.DisplayShape(
                    face_shape,
                    color=_retained_rank_color(rank_index),
                    transparency=0.0,
                    update=False,
                )
        for prediction in selected_predictions:
            face_shape = face_shapes.get(prediction.face_id)
            if face_shape is not None:
                display.DisplayShape(
                    face_shape,
                    color=MODEL_HOTSPOT_COLOR,
                    transparency=0.0,
                    update=False,
                )

        retained_samples = _retained_particle_samples(result, retained_face_set)
        if retained_samples:
            radius = _marker_radius([sample.point for sample in result.samples], scale=0.005)
            for sample in _limited_sequence(retained_samples, MAX_PARTICLE_MARKERS):
                display.DisplayShape(
                    _sphere_at(sample.point, radius * (0.65 + sample.remaining_dust)),
                    color=_retained_sample_color(sample, retained_faces),
                    update=False,
                )

        model_status.set(
            f"Pink=model {len(selected_predictions)} faces; "
            f"rank colors=particles stayed at force {REFERENCE_WATER_FORCE:.3f} "
            f"dirs={_direction_summary(reference_cleaning_parameters())}: {len(retained_faces)} faces"
        )
        status.set(
            _status_text(
                path,
                edge_groups,
                feature_summary,
                (
                    f"compare: pink={len(selected_predictions)} model faces, "
                    f"rank colors={len(retained_faces)} retained faces at force={REFERENCE_WATER_FORCE:.3f}, "
                    f"accepted through green={len(accepted_retained_faces)} faces, "
                    f"dirs={_direction_summary(reference_cleaning_parameters())}, accepted overlap={overlap_count}, "
                    f"order={_retained_rank_summary(retained_faces, retained_scores)}"
                ),
            )
        )
        display.Repaint()

    def show_learned_hotspots() -> None:
        cleaning_view_active["value"] = False
        try:
            parameters = selected_cleaning_parameters()
        except ValueError as exc:
            model_status.set(f"Model error: {exc}")
            status.set(_status_text(path, edge_groups, feature_summary, f"model error: {exc}"))
            return

        model_status.set("Model: extracting features and predicting...")
        status.set(
            _status_text(
                path,
                edge_groups,
                feature_summary,
                f"model predicting with force={parameters.water_force:.3f} dirs={_direction_summary(parameters)}...",
            )
        )
        root.update_idletasks()

        try:
            predictions = predict_step_hotspots(path, model_path=DEFAULT_HOTSPOT_MODEL_PATH, parameters=parameters)
        except Exception as exc:
            model_status.set(f"Model error: {exc}")
            status.set(_status_text(path, edge_groups, feature_summary, f"model error: {exc}"))
            return

        if not predictions:
            model_status.set("Model: no mesh faces were available for prediction")
            status.set(_status_text(path, edge_groups, feature_summary, "no model hotspots"))
            return

        selected = _selected_learned_hotspots(predictions)
        max_prediction = max(prediction.predicted_retained_particle_ratio for prediction in predictions)

        _clear_display(display)
        display.DisplayShape(shape, color=BASE_COLOR, transparency=0.88, update=False)
        shown = 0
        for prediction in selected:
            face_shape = face_shapes.get(prediction.face_id)
            if face_shape is None:
                continue
            display.DisplayShape(
                face_shape,
                color=MODEL_HOTSPOT_COLOR,
                transparency=0.0,
                update=False,
            )
            shown += 1

        summary = _prediction_summary(predictions)
        model_status.set(f"Model: {summary}")
        status.set(
            _status_text(
                path,
                edge_groups,
                feature_summary,
                (
                    f"model-only pink hotspots: {shown} faces, "
                    f"force={parameters.water_force:.3f}, max model score={max_prediction:.6f}"
                ),
            )
        )
        display.Repaint()

    def schedule_cleaning_refresh(*_args) -> None:
        if not cleaning_view_active["value"]:
            return
        if cleaning_refresh_after_id["id"] is not None:
            root.after_cancel(cleaning_refresh_after_id["id"])
        cleaning_refresh_after_id["id"] = root.after(350, refresh_cleaning_view)

    def refresh_cleaning_view() -> None:
        cleaning_refresh_after_id["id"] = None
        if _parse_water_force(water_force_var.get()) is None:
            status.set(_status_text(path, edge_groups, feature_summary, "cleaning error: enter a numeric water force"))
            return
        show_cleaning_result()

    tk.Button(toolbar, text="Open STEP File", command=choose_step_file).pack(side=tk.LEFT, padx=(0, 12))
    tk.Button(toolbar, text="Predict Hotspots", command=show_learned_hotspots).pack(side=tk.LEFT, padx=(0, 12))
    tk.Button(toolbar, text="Run Cleaning Simulation", command=show_cleaning_result).pack(side=tk.LEFT, padx=(0, 18))
    tk.Label(toolbar, text="Water Force").pack(side=tk.LEFT, padx=(0, 6))
    force_entry = tk.Entry(
        toolbar,
        width=8,
        textvariable=water_force_var,
    )
    force_entry.pack(side=tk.LEFT, padx=(0, 6))
    force_entry.bind("<Return>", lambda _event: schedule_cleaning_refresh())
    force_entry.bind("<FocusOut>", lambda _event: schedule_cleaning_refresh())
    water_force_var.trace_add("write", schedule_cleaning_refresh)
    tk.Label(toolbar, text="Range 0.000-1.000").pack(side=tk.LEFT, padx=(0, 18))
    tk.Label(
        toolbar,
        text=f"LightGBM LambdaRank: {DEFAULT_HOTSPOT_MODEL_PATH}",
    ).pack(side=tk.LEFT, padx=(0, 12))
    tk.Label(statusbar, textvariable=status, anchor="w").pack(side=tk.TOP, fill=tk.X)
    tk.Label(statusbar, textvariable=model_status, anchor="w").pack(side=tk.TOP, fill=tk.X)

    display.DisplayShape(shape, color=BASE_COLOR, transparency=0.15, update=True)
    root.mainloop()


def _inward_cylinder_data(path: Path) -> tuple[str, list]:
    try:
        analysis = InwardCylinderHeuristic().find_step(path)
    except Exception as exc:
        return f"inward cylinder heuristic error: {exc}", []

    face_shapes = []
    seen_face_ids: set[int] = set()
    for candidate in analysis.candidates:
        for face_id in candidate.face_ids:
            if face_id in seen_face_ids:
                continue
            shape = analysis.face_shapes.get(face_id)
            if shape is not None:
                face_shapes.append(shape)
                seen_face_ids.add(face_id)

    return f"inward cylinders: {len(analysis.candidates)}{_cylinder_summary(analysis.candidates)}", face_shapes


def _face_form_data(path: Path) -> tuple[str, dict[str, list], dict[int, Any]]:
    try:
        analysis = FaceFormFinder().find_step(path)
    except Exception as exc:
        return f"forms error: {exc}", {form_type: [] for form_type in FACE_FORM_TYPES}, {}

    return _form_summary(analysis), analysis.shape_groups(), analysis.face_shapes


def _face_hotspot_scores(result: CleaningSimulationResult) -> dict[int, float]:
    scores: dict[int, float] = {}
    for sample in result.samples:
        scores[sample.face_id] = max(scores.get(sample.face_id, 0.0), sample.hotspot_score)
    return scores


def _face_retained_particle_scores(result: CleaningSimulationResult) -> dict[int, float]:
    retained_count_by_face: dict[int, float] = {}
    for sample in result.samples:
        if sample.remaining_dust < REMAINING_DUST_THRESHOLD:
            continue
        retained_count_by_face[sample.face_id] = retained_count_by_face.get(sample.face_id, 0.0) + 1.0
    return retained_count_by_face


def _selected_retained_particle_faces(face_scores: dict[int, float]) -> list[int]:
    if not face_scores:
        return []
    ranked = sorted(face_scores, key=lambda face_id: (-face_scores[face_id], face_id))
    return ranked[:RETAINED_PARTICLE_ACCEPTED_RANK_LIMIT]


def _accepted_retained_particle_faces(retained_faces: list[int]) -> list[int]:
    return retained_faces


def _retained_rank_color(rank_index: int) -> Quantity_Color:
    if rank_index < len(RETAINED_PARTICLE_RANK_COLORS):
        return RETAINED_PARTICLE_RANK_COLORS[rank_index]
    return RETAINED_PARTICLE_RANK_COLORS[-1]


def _retained_sample_color(sample: CleaningSample, retained_faces: list[int]) -> Quantity_Color:
    try:
        rank_index = retained_faces.index(sample.face_id)
    except ValueError:
        rank_index = len(RETAINED_PARTICLE_RANK_COLORS) - 1
    return _retained_rank_color(rank_index)


def _retained_rank_summary(face_ids: list[int], face_scores: dict[int, float]) -> str:
    color_names = ("red", "orange", "yellow", "green", "cyan", "blue", "violet", "gray")
    parts = []
    for rank, face_id in enumerate(face_ids, start=1):
        color_name = color_names[min(rank - 1, len(color_names) - 1)]
        parts.append(f"{rank}:{color_name}:F{face_id}={int(face_scores.get(face_id, 0))}")
    return "; ".join(parts)


def _hotspot_samples(result: CleaningSimulationResult, cutoff: float) -> list[CleaningSample]:
    samples = [sample for sample in result.samples if sample.hotspot_score >= cutoff]
    return samples or result.top_hotspots(80)


def _remaining_particles(result: CleaningSimulationResult) -> list[CleaningSample]:
    return [
        sample
        for sample in result.samples
        if sample.remaining_dust >= REMAINING_DUST_THRESHOLD
    ]


def _retained_particle_samples(result: CleaningSimulationResult, face_ids: set[int]) -> list[CleaningSample]:
    return [
        sample
        for sample in result.samples
        if sample.face_id in face_ids and sample.remaining_dust >= REMAINING_DUST_THRESHOLD
    ]


def _hotspot_cutoff(result: CleaningSimulationResult) -> float:
    max_score = result.max_hotspot_score
    if max_score <= 1e-9:
        return 1.0
    if max_score >= HOTSPOT_THRESHOLD:
        return HOTSPOT_THRESHOLD
    return max_score * 0.85


def _hotspot_color(score: float, cutoff: float, max_score: float) -> Quantity_Color:
    score = _clamp((score - cutoff) / max(max_score - cutoff, 1e-9))
    red = 0.9
    green = 0.78 - 0.58 * score
    blue = 0.08 - 0.04 * score
    return Quantity_Color(red, green, blue, Quantity_TOC_RGB)


def _remaining_dust_color(remaining_dust: float) -> Quantity_Color:
    remaining_dust = _clamp(remaining_dust)
    red = 0.78 - 0.32 * remaining_dust
    green = 0.54 - 0.34 * remaining_dust
    blue = 0.18 - 0.10 * remaining_dust
    return Quantity_Color(red, green, blue, Quantity_TOC_RGB)


def _selected_learned_hotspots(predictions: list[HotspotPrediction]) -> list[HotspotPrediction]:
    return predictions[:LEARNED_HOTSPOT_TOP_K]


def _clear_display(display: Any) -> None:
    display.EraseAll()
    try:
        display.Context.RemoveAll(False)
    except AttributeError:
        pass


def _prediction_summary(predictions: list[HotspotPrediction], limit: int = 5) -> str:
    top = predictions[:limit]
    summary = ", ".join(
        f"F{prediction.face_id}={prediction.predicted_retained_particle_ratio:.6f}"
        for prediction in top
    )
    return f"top predictions {summary}"


def _cleaning_cache_key(parameters: CleaningSimulationParameters) -> tuple[float, tuple[tuple[float, float, float], ...]]:
    return round(parameters.water_force, 3), parameters.water_directions


def _direction_summary(parameters: CleaningSimulationParameters) -> str:
    all_directions = tuple(direction for _, _, direction in SPRAY_DIRECTION_OPTIONS)
    if set(parameters.water_directions) == set(all_directions):
        return "all"

    labels = []
    for key, _label, direction in SPRAY_DIRECTION_OPTIONS:
        if direction in parameters.water_directions:
            labels.append(key)
    return ",".join(labels) if labels else "none"


def _parse_water_force(text: str) -> float | None:
    try:
        return float(text.strip().replace(",", "."))
    except ValueError:
        return None


def _sphere_at(point: tuple[float, float, float], radius: float):
    return BRepPrimAPI_MakeSphere(gp_Pnt(point[0], point[1], point[2]), radius).Shape()


def _marker_radius(points: list[tuple[float, float, float]], *, scale: float) -> float:
    if not points:
        return 1.0
    lows = [min(point[axis] for point in points) for axis in range(3)]
    highs = [max(point[axis] for point in points) for axis in range(3)]
    diagonal = sum((highs[axis] - lows[axis]) ** 2 for axis in range(3)) ** 0.5
    return max(diagonal * scale, 0.05)


def _limited_sequence(items: list[Any], limit: int) -> list[Any]:
    if len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(index * step)] for index in range(limit)]


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _cylinder_summary(candidates: list[InwardCylinderCandidate], limit: int = 180) -> str:
    if not candidates:
        return ""
    text = "   " + "; ".join(
        f"C{candidate.id} dia={candidate.diameter:.3g} span={_span_text(candidate)} conf={candidate.confidence:.2g}"
        for candidate in candidates
    )
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _span_text(candidate: InwardCylinderCandidate) -> str:
    return "unknown" if candidate.axial_span is None else f"{candidate.axial_span:.3g}"


def _form_summary(analysis: FaceFormAnalysis) -> str:
    counts = " ".join(f"{form_type}={analysis.count(form_type)}" for form_type in FACE_FORM_TYPES)
    return f"forms: {counts}"


def _status_text(path: Path, edge_groups: dict[str, list], feature_summary: str, mode: str) -> str:
    return f"{path.name}   {mode}"


def main() -> int:
    view_step_file(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_STEP_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
