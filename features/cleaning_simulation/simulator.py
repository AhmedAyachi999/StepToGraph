from __future__ import annotations

import html
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Core.TopLoc import TopLoc_Location
from occwl.compound import Compound


Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class SurfaceMeshTriangle:
    id: int
    face_id: int
    vertices: tuple[Point3, Point3, Point3]
    point: Point3
    normal: Point3
    area: float


@dataclass(frozen=True)
class SurfaceMesh:
    source_file: str
    triangles: list[SurfaceMeshTriangle]
    neighbors: dict[int, list[int]]


@dataclass(frozen=True)
class CleaningSimulationParameters:
    mesh_linear_deflection: float = 0.8
    mesh_angular_deflection: float = 0.5
    gravity_direction: Point3 = (0.0, 0.0, -1.0)
    water_directions: tuple[Point3, ...] = ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0))
    water_force: float = 1.0
    flow_steps: int = 28
    flow_retention: float = 0.82
    cleaning_rate: float = 0.28
    deposition_rate: float = 0.2
    vertex_merge_tolerance: float = 1e-5

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CleaningSample:
    id: int
    face_id: int
    point: Point3
    normal: Point3
    area: float
    exposure: float
    slope: float
    poor_drainage: float
    concavity: float
    hiddenness: float
    water_dose: float
    cleaning_dose: float
    remaining_dust: float
    redeposition: float
    hotspot_score: float
    downstream_id: int | None


@dataclass(frozen=True)
class CleaningSimulationResult:
    source_file: str
    parameters: dict[str, Any]
    samples: list[CleaningSample]

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def max_hotspot_score(self) -> float:
        return max((sample.hotspot_score for sample in self.samples), default=0.0)

    def top_hotspots(self, limit: int = 12) -> list[CleaningSample]:
        return sorted(self.samples, key=lambda sample: sample.hotspot_score, reverse=True)[:limit]

    def summary(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "max_hotspot_score": round(self.max_hotspot_score, 4),
            "mean_hotspot_score": _mean(sample.hotspot_score for sample in self.samples),
            "mean_cleaning_dose": _mean(sample.cleaning_dose for sample in self.samples),
            "mean_remaining_dust": _mean(sample.remaining_dust for sample in self.samples),
            "mean_redeposition": _mean(sample.redeposition for sample in self.samples),
            "remaining_particle_count_0_2": sum(sample.remaining_dust >= 0.2 for sample in self.samples),
            "hotspot_count_0_65": sum(sample.hotspot_score >= 0.65 for sample in self.samples),
        }


def simulate_cleaning(
    step_path: str | Path,
    *,
    parameters: CleaningSimulationParameters | None = None,
) -> CleaningSimulationResult:
    parameters = parameters or CleaningSimulationParameters()
    mesh = step_to_surface_mesh(step_path, parameters=parameters)
    return simulate_cleaning_on_mesh(mesh, parameters=parameters)


def step_to_surface_mesh(
    step_path: str | Path,
    *,
    parameters: CleaningSimulationParameters | None = None,
) -> SurfaceMesh:
    parameters = parameters or CleaningSimulationParameters()
    step_path = Path(step_path)
    compound = Compound.load_from_step(step_path)
    BRepMesh_IncrementalMesh(
        compound.topods_shape(),
        parameters.mesh_linear_deflection,
        False,
        parameters.mesh_angular_deflection,
        True,
    )

    triangles: list[SurfaceMeshTriangle] = []
    triangle_vertex_keys: dict[int, tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = {}

    for face_id, face in enumerate(compound.faces(), start=1):
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face.topods_shape(), location)
        if triangulation is None:
            continue

        transform = location.Transformation()
        nodes = {
            node_index: _point(triangulation.Node(node_index).Transformed(transform))
            for node_index in range(1, triangulation.NbNodes() + 1)
        }

        reversed_face = face.topods_shape().Orientation() == TopAbs_REVERSED
        for triangle_index in range(1, triangulation.NbTriangles() + 1):
            node_ids = triangulation.Triangle(triangle_index).Get()
            vertices = tuple(nodes[node_id] for node_id in node_ids)
            normal, area = _triangle_normal_and_area(vertices)
            if area <= 1e-12:
                continue
            if reversed_face:
                normal = _scale(normal, -1.0)

            triangle_id = len(triangles)
            triangles.append(
                SurfaceMeshTriangle(
                    id=triangle_id,
                    face_id=face_id,
                    vertices=vertices,
                    point=_centroid(vertices),
                    normal=normal,
                    area=area,
                )
            )
            triangle_vertex_keys[triangle_id] = tuple(
                _vertex_key(vertex, parameters.vertex_merge_tolerance) for vertex in vertices
            )

    return SurfaceMesh(
        source_file=str(step_path.resolve()),
        triangles=triangles,
        neighbors=_build_neighbors(triangle_vertex_keys),
    )


def simulate_cleaning_on_mesh(
    mesh: SurfaceMesh,
    *,
    parameters: CleaningSimulationParameters | None = None,
) -> CleaningSimulationResult:
    parameters = parameters or CleaningSimulationParameters()
    if not mesh.triangles:
        return CleaningSimulationResult(
            source_file=mesh.source_file,
            parameters=parameters.to_json_dict(),
            samples=[],
        )

    gravity = _normalize(parameters.gravity_direction)
    water_directions = [_normalize(direction) for direction in parameters.water_directions]
    downstream, slopes = _downstream_neighbors(mesh, gravity)
    exposures = [_spray_exposure(triangle.normal, water_directions) for triangle in mesh.triangles]
    concavities = _concavity_scores(mesh)
    poor_drainage = [
        _poor_drainage_score(downstream_id, slope)
        for downstream_id, slope in zip(downstream, slopes)
    ]
    hiddenness = [
        _clamp(0.75 * (1.0 - exposure) + 0.25 * concavity)
        for exposure, concavity in zip(exposures, concavities)
    ]

    water_dose, redeposition, remaining_dust = _simulate_water_and_soil(
        mesh,
        downstream,
        exposures,
        poor_drainage,
        concavities,
        hiddenness,
        parameters,
    )
    water_dose_normalized = _normalize_values(water_dose)
    redeposition_normalized = _normalize_values(redeposition)

    samples: list[CleaningSample] = []
    for triangle in mesh.triangles:
        index = triangle.id
        low_dose = 1.0 - water_dose_normalized[index]
        geometry_risk = _clamp(0.55 * hiddenness[index] + 0.45 * concavities[index])
        drainage_risk = poor_drainage[index]
        hotspot_score = _clamp(
            0.43 * low_dose
            + 0.25 * drainage_risk
            + 0.2 * geometry_risk
            + 0.12 * redeposition_normalized[index]
        )
        cleaning_dose = _clamp(water_dose_normalized[index] * (1.0 - 0.35 * hiddenness[index]))
        samples.append(
            CleaningSample(
                id=triangle.id,
                face_id=triangle.face_id,
                point=_rounded_point(triangle.point),
                normal=_rounded_point(triangle.normal),
                area=round(triangle.area, 6),
                exposure=round(exposures[index], 6),
                slope=round(slopes[index], 6),
                poor_drainage=round(poor_drainage[index], 6),
                concavity=round(concavities[index], 6),
                hiddenness=round(hiddenness[index], 6),
                water_dose=round(water_dose_normalized[index], 6),
                cleaning_dose=round(cleaning_dose, 6),
                remaining_dust=round(remaining_dust[index], 6),
                redeposition=round(redeposition_normalized[index], 6),
                hotspot_score=round(hotspot_score, 6),
                downstream_id=downstream[index],
            )
        )

    return CleaningSimulationResult(
        source_file=mesh.source_file,
        parameters=parameters.to_json_dict(),
        samples=samples,
    )


def simulation_to_json(result: CleaningSimulationResult) -> str:
    payload = asdict(result)
    payload["summary"] = result.summary()
    payload["top_hotspots"] = [asdict(sample) for sample in result.top_hotspots()]
    return json.dumps(payload, indent=2)


def export_heatmap_html(
    result: CleaningSimulationResult,
    output_path: str | Path,
    *,
    metric: str = "hotspot_score",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_heatmap_html(result, metric), encoding="utf-8")


def _simulate_water_and_soil(
    mesh: SurfaceMesh,
    downstream: list[int | None],
    exposures: list[float],
    poor_drainage: list[float],
    concavities: list[float],
    hiddenness: list[float],
    parameters: CleaningSimulationParameters,
) -> tuple[list[float], list[float], list[float]]:
    count = len(mesh.triangles)
    water_force = max(0.0, parameters.water_force)
    dust = [triangle.area for triangle in mesh.triangles]
    direct_water = [exposure * triangle.area * water_force for exposure, triangle in zip(exposures, mesh.triangles)]
    water_dose = [0.0] * count
    redeposition = [0.0] * count
    incoming_water = [0.0] * count
    incoming_soil = [0.0] * count

    for _ in range(max(1, parameters.flow_steps)):
        next_water = [0.0] * count
        next_soil = [0.0] * count

        for index, triangle in enumerate(mesh.triangles):
            water = direct_water[index] + incoming_water[index]
            if water <= 1e-12 and incoming_soil[index] <= 1e-12:
                continue

            area = max(triangle.area, 1e-12)
            dose_increment = water / area
            water_dose[index] += dose_increment
            removed = dust[index] * (1.0 - math.exp(-parameters.cleaning_rate * dose_increment))
            dust[index] = max(0.0, dust[index] - removed)

            slow_geometry = _clamp(
                0.55 * poor_drainage[index]
                + 0.25 * concavities[index]
                + 0.2 * hiddenness[index]
            )
            deposit_fraction = _clamp(
                parameters.deposition_rate * (0.25 + slow_geometry),
                upper=0.85,
            )
            dirty_water = incoming_soil[index] + removed
            deposited = dirty_water * deposit_fraction
            redeposition[index] += deposited / area

            remaining_soil = dirty_water - deposited
            downstream_id = downstream[index]
            flow_fraction = parameters.flow_retention * (1.0 - 0.45 * poor_drainage[index])
            if downstream_id is None or flow_fraction <= 1e-6:
                redeposition[index] += remaining_soil / area
                continue

            next_water[downstream_id] += water * flow_fraction
            next_soil[downstream_id] += remaining_soil * flow_fraction

        incoming_water = next_water
        incoming_soil = next_soil

    remaining_dust = [
        dust[index] / max(triangle.area, 1e-12)
        for index, triangle in enumerate(mesh.triangles)
    ]
    return water_dose, redeposition, remaining_dust


def _spray_exposure(normal: Point3, water_directions: list[Point3]) -> float:
    if not water_directions:
        return 0.0
    return max(_clamp(_dot(normal, _scale(direction, -1.0))) for direction in water_directions)


def _downstream_neighbors(mesh: SurfaceMesh, gravity: Point3) -> tuple[list[int | None], list[float]]:
    downstream: list[int | None] = []
    slopes: list[float] = []

    for triangle in mesh.triangles:
        best_neighbor: int | None = None
        best_slope = 0.0
        for neighbor_id in mesh.neighbors.get(triangle.id, []):
            neighbor = mesh.triangles[neighbor_id]
            offset = _sub(neighbor.point, triangle.point)
            distance = _norm(offset)
            if distance <= 1e-12:
                continue
            drop = _dot(offset, gravity)
            slope = drop / distance
            if drop > 1e-8 and slope > best_slope:
                best_neighbor = neighbor_id
                best_slope = slope
        downstream.append(best_neighbor)
        slopes.append(_clamp(best_slope))

    return downstream, slopes


def _poor_drainage_score(downstream_id: int | None, slope: float) -> float:
    if downstream_id is None:
        return 1.0
    return _clamp(1.0 - slope / 0.35)


def _concavity_scores(mesh: SurfaceMesh) -> list[float]:
    scores = [0.0] * len(mesh.triangles)

    for triangle in mesh.triangles:
        for neighbor_id in mesh.neighbors.get(triangle.id, []):
            if neighbor_id <= triangle.id:
                continue
            neighbor = mesh.triangles[neighbor_id]
            angle_score = _clamp((1.0 - _dot(triangle.normal, neighbor.normal)) / 0.7)
            if angle_score <= 0.0:
                continue

            toward_neighbor = _normalize(_sub(neighbor.point, triangle.point))
            toward_triangle = _scale(toward_neighbor, -1.0)
            triangle_inside_turn = max(0.0, _dot(triangle.normal, toward_neighbor))
            neighbor_inside_turn = max(0.0, _dot(neighbor.normal, toward_triangle))
            scores[triangle.id] = max(scores[triangle.id], _clamp(angle_score * triangle_inside_turn))
            scores[neighbor.id] = max(scores[neighbor.id], _clamp(angle_score * neighbor_inside_turn))

    return scores


def _build_neighbors(
    triangle_vertex_keys: dict[int, tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]],
) -> dict[int, list[int]]:
    edges: dict[tuple[tuple[int, int, int], tuple[int, int, int]], list[int]] = {}
    for triangle_id, vertex_keys in triangle_vertex_keys.items():
        for start, end in ((0, 1), (1, 2), (2, 0)):
            edge_key = tuple(sorted((vertex_keys[start], vertex_keys[end])))
            edges.setdefault(edge_key, []).append(triangle_id)

    neighbors = {triangle_id: set() for triangle_id in triangle_vertex_keys}
    for triangle_ids in edges.values():
        if len(triangle_ids) < 2:
            continue
        for source in triangle_ids:
            neighbors[source].update(target for target in triangle_ids if target != source)

    return {triangle_id: sorted(values) for triangle_id, values in neighbors.items()}


def _heatmap_html(result: CleaningSimulationResult, metric: str) -> str:
    if not result.samples:
        body = "<p>No mesh samples were generated.</p>"
        return _html_document("Cleaning heatmap", body)

    if not hasattr(result.samples[0], metric):
        raise ValueError(f"Unknown heatmap metric: {metric}")

    axis_a, axis_b = _projection_axes(result.samples)
    bounds_a = _bounds(getattr(sample, "point")[axis_a] for sample in result.samples)
    bounds_b = _bounds(getattr(sample, "point")[axis_b] for sample in result.samples)
    width, height, pad = 980, 680, 36
    metric_values = [float(getattr(sample, metric)) for sample in result.samples]
    metric_min, metric_max = min(metric_values), max(metric_values)

    circles = []
    for sample, metric_value in zip(result.samples, metric_values):
        x = _scale_to_canvas(sample.point[axis_a], bounds_a, pad, width - pad)
        y = height - _scale_to_canvas(sample.point[axis_b], bounds_b, pad, height - pad)
        normalized = 0.0 if metric_max == metric_min else (metric_value - metric_min) / (metric_max - metric_min)
        radius = 3.0 + 4.0 * normalized
        circles.append(
            "<circle "
            f"cx=\"{x:.2f}\" cy=\"{y:.2f}\" r=\"{radius:.2f}\" "
            f"fill=\"{_heat_color(normalized)}\" opacity=\"0.86\">"
            f"<title>sample {sample.id} face {sample.face_id} {html.escape(metric)}={metric_value:.4f} "
            f"cleaning={sample.cleaning_dose:.4f} remaining={sample.remaining_dust:.4f} "
            f"redeposition={sample.redeposition:.4f}</title>"
            "</circle>"
        )

    top_rows = "\n".join(
        "<tr>"
        f"<td>{sample.id}</td>"
        f"<td>{sample.face_id}</td>"
        f"<td>{sample.hotspot_score:.4f}</td>"
        f"<td>{sample.cleaning_dose:.4f}</td>"
        f"<td>{sample.remaining_dust:.4f}</td>"
        f"<td>{sample.poor_drainage:.4f}</td>"
        f"<td>{sample.concavity:.4f}</td>"
        f"<td>{sample.hiddenness:.4f}</td>"
        f"<td>{sample.redeposition:.4f}</td>"
        "</tr>"
        for sample in result.top_hotspots(10)
    )

    summary_rows = "\n".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in result.summary().items()
    )

    body = f"""
<header>
  <h1>Cleaning heatmap</h1>
  <p>{html.escape(Path(result.source_file).name)} projected on {_axis_name(axis_a)}/{_axis_name(axis_b)}. Metric: {html.escape(metric)}.</p>
</header>
<main>
  <section>
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="Cleaning heatmap">
      <rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc" />
      {''.join(circles)}
    </svg>
  </section>
  <section class="grid">
    <table>
      <caption>Summary</caption>
      <tbody>{summary_rows}</tbody>
    </table>
    <table>
      <caption>Top hot spots</caption>
      <thead>
        <tr><th>sample</th><th>face</th><th>hotspot</th><th>cleaning</th><th>remaining</th><th>drainage</th><th>concavity</th><th>hidden</th><th>redeposition</th></tr>
      </thead>
      <tbody>{top_rows}</tbody>
    </table>
  </section>
</main>
"""
    return _html_document("Cleaning heatmap", body)


def _html_document(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      color: #172033;
      background: #ffffff;
    }}
    header, main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 20px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 28px;
    }}
    p {{
      margin: 0;
      color: #526071;
    }}
    svg {{
      width: 100%;
      height: auto;
      border: 1px solid #d8dee8;
      background: #f8fafc;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(260px, 0.8fr) minmax(0, 1.5fr);
      gap: 18px;
      align-items: start;
      margin-top: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    caption {{
      text-align: left;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    th, td {{
      border-bottom: 1px solid #e4e8f0;
      padding: 7px 8px;
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child, caption {{
      text-align: left;
    }}
    @media (max-width: 760px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def _projection_axes(samples: list[CleaningSample]) -> tuple[int, int]:
    ranges = []
    for axis in range(3):
        low, high = _bounds(sample.point[axis] for sample in samples)
        ranges.append((high - low, axis))
    return tuple(axis for _, axis in sorted(ranges, reverse=True)[:2])  # type: ignore[return-value]


def _scale_to_canvas(value: float, bounds: tuple[float, float], low: float, high: float) -> float:
    start, end = bounds
    if abs(end - start) <= 1e-12:
        return (low + high) * 0.5
    return low + (value - start) / (end - start) * (high - low)


def _axis_name(axis: int) -> str:
    return ("X", "Y", "Z")[axis]


def _heat_color(value: float) -> str:
    stops = (
        (0.0, (35, 84, 163)),
        (0.35, (40, 154, 142)),
        (0.68, (245, 196, 66)),
        (1.0, (196, 43, 35)),
    )
    value = _clamp(value)
    for index in range(len(stops) - 1):
        start_value, start_color = stops[index]
        end_value, end_color = stops[index + 1]
        if value <= end_value:
            amount = (value - start_value) / max(end_value - start_value, 1e-12)
            color = tuple(
                round(start_color[channel] + (end_color[channel] - start_color[channel]) * amount)
                for channel in range(3)
            )
            return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
    return "#c42b23"


def _normalize_values(values: list[float]) -> list[float]:
    low = min(values, default=0.0)
    high = max(values, default=0.0)
    if high - low <= 1e-12:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]




def _bounds(values: Any) -> tuple[float, float]:
    values = list(values)
    if not values:
        return 0.0, 0.0
    return min(values), max(values)


def _mean(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _triangle_normal_and_area(vertices: tuple[Point3, Point3, Point3]) -> tuple[Point3, float]:
    a, b, c = vertices
    cross = _cross(_sub(b, a), _sub(c, a))
    length = _norm(cross)
    if length <= 1e-12:
        return (0.0, 0.0, 1.0), 0.0
    return _scale(cross, 1.0 / length), 0.5 * length


def _centroid(vertices: tuple[Point3, Point3, Point3]) -> Point3:
    return (
        sum(vertex[0] for vertex in vertices) / 3.0,
        sum(vertex[1] for vertex in vertices) / 3.0,
        sum(vertex[2] for vertex in vertices) / 3.0,
    )


def _vertex_key(point: Point3, tolerance: float) -> tuple[int, int, int]:
    tolerance = max(tolerance, 1e-12)
    return (
        round(point[0] / tolerance),
        round(point[1] / tolerance),
        round(point[2] / tolerance),
    )


def _point(value: Any) -> Point3:
    if hasattr(value, "Coord"):
        x, y, z = value.Coord()
        return float(x), float(y), float(z)
    if hasattr(value, "tolist"):
        value = value.tolist()
    return float(value[0]), float(value[1]), float(value[2])


def _rounded_point(point: Point3) -> Point3:
    return round(point[0], 6), round(point[1], 6), round(point[2], 6)


def _normalize(vector: Point3) -> Point3:
    length = _norm(vector)
    if length <= 1e-12:
        raise ValueError(f"cannot normalize zero-length vector: {vector}")
    return _scale(vector, 1.0 / length)


def _add(a: Point3, b: Point3) -> Point3:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _sub(a: Point3, b: Point3) -> Point3:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _scale(vector: Point3, scalar: float) -> Point3:
    return vector[0] * scalar, vector[1] * scalar, vector[2] * scalar


def _dot(a: Point3, b: Point3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Point3, b: Point3) -> Point3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(vector: Point3) -> float:
    return math.sqrt(_dot(vector, vector))


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))
