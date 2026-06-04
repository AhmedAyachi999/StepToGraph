from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import cadquery as cq
from OCP.BRep import BRep_Tool
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
from OCP.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS


Convexity = Literal["convex", "concave", "neutral"]


class AnalysisCancelled(Exception):
    """Raised when a caller requests cancellation during edge classification."""


class CircleFaceIntersectionError(RuntimeError):
    """Raised when the auxiliary circle does not intersect an adjacent face."""


@dataclass(frozen=True)
class EdgeProbe:
    center: list[float]
    normal_a: list[float]
    normal_b: list[float]
    p1: list[float] | None
    p2: list[float] | None
    midpoint: list[float] | None
    midpoint_inside: bool | None
    chord_length: float | None
    radius: float


@dataclass(frozen=True)
class ClassifiedEdge:
    id: int
    source_face: int
    target_face: int
    curve_type: str
    convexity: Convexity
    samples: list[list[float]]
    probe: EdgeProbe


@dataclass(frozen=True)
class EdgeAnalysis:
    source_file: str
    edge_count: int
    convex_count: int
    concave_count: int
    neutral_count: int
    edges: list[ClassifiedEdge]

    @property
    def visible_edges(self) -> list[ClassifiedEdge]:
        return [edge for edge in self.edges if edge.convexity in {"convex", "concave"}]


def classify_step_edges(
    step_path: str | Path,
    *,
    probe_samples: int | None = None,
    edge_sample_count: int = 24,
    cancel_event: Any | None = None,
) -> EdgeAnalysis:
    """Classify STEP shared edges with an auxiliary-circle probe."""
    _check_cancel(cancel_event)
    step_path = Path(step_path)
    solid = cq.importers.importStep(str(step_path)).val()
    _check_cancel(cancel_event)
    faces = solid.Faces()
    face_ids = _face_ids(step_path, faces)
    classified_edges = _shared_edges(solid, faces, face_ids, edge_sample_count, cancel_event)

    return EdgeAnalysis(
        source_file=str(step_path.resolve()),
        edge_count=len(classified_edges),
        convex_count=sum(edge.convexity == "convex" for edge in classified_edges),
        concave_count=sum(edge.convexity == "concave" for edge in classified_edges),
        neutral_count=sum(edge.convexity == "neutral" for edge in classified_edges),
        edges=classified_edges,
    )


def analysis_to_json(analysis: EdgeAnalysis) -> str:
    return json.dumps(asdict(analysis), indent=2)


def _face_ids(step_path: Path, faces: list[cq.Face]) -> dict[int, int]:
    step_ids = _step_face_ids(step_path)
    return {
        face.hashCode(): step_ids[index] if index < len(step_ids) else index + 1
        for index, face in enumerate(faces)
    }


def _step_face_ids(step_path: Path) -> list[int]:
    text = step_path.read_text(encoding="utf-8", errors="replace")
    return [int(value) for value in re.findall(r"#(\d+)\s*=\s*ADVANCED_FACE\s*\(", text)]


def _shared_edges(
    solid: cq.Solid,
    faces: list[cq.Face],
    face_ids: dict[int, int],
    edge_sample_count: int,
    cancel_event: Any | None,
) -> list[ClassifiedEdge]:
    edge_faces: dict[int, list[tuple[cq.Face, cq.Edge]]] = defaultdict(list)
    for face in faces:
        _check_cancel(cancel_event)
        for edge in face.Edges():
            references = edge_faces[edge.hashCode()]
            already_seen = any(
                edge.isSame(existing_edge) and face.isSame(existing_face)
                for existing_face, existing_edge in references
            )
            if not already_seen:
                references.append((face, edge))

    box = solid.BoundingBox()
    probe_radius = max(max(box.xlen, box.ylen, box.zlen) * 1e-3, 1e-3)
    classified_edges: list[ClassifiedEdge] = []

    for edge_id, references in edge_faces.items():
        _check_cancel(cancel_event)
        references = _unique_face_references(references)
        if len(references) != 2:
            continue

        (face_a, edge), (face_b, _) = references
        probe, convexity = _convexity_probe(
            solid,
            face_a,
            face_b,
            edge,
            probe_radius,
            cancel_event,
        )
        classified_edges.append(
            ClassifiedEdge(
                id=edge_id,
                source_face=face_ids[face_a.hashCode()],
                target_face=face_ids[face_b.hashCode()],
                curve_type=edge.geomType(),
                convexity=convexity,
                samples=_edge_samples(edge, edge_sample_count),
                probe=probe,
            )
        )

    return classified_edges


def _unique_face_references(references: list[tuple[cq.Face, cq.Edge]]) -> list[tuple[cq.Face, cq.Edge]]:
    unique: list[tuple[cq.Face, cq.Edge]] = []
    for face, edge in references:
        if not any(face.isSame(existing_face) for existing_face, _ in unique):
            unique.append((face, edge))
    return unique


def _convexity_probe(
    solid: cq.Solid,
    face_a: cq.Face,
    face_b: cq.Face,
    edge: cq.Edge,
    radius: float,
    cancel_event: Any | None,
) -> tuple[EdgeProbe, Convexity]:
    _check_cancel(cancel_event)
    center = edge.positionAt(0.5)
    normal_a = _probe_normal(face_a, center)
    normal_b = _probe_normal(face_b, center)
    circle = _auxiliary_circle(center, normal_a, normal_b, radius)
    if circle is None:
        return _probe_result(center, normal_a, normal_b, None, None, None, None, None, radius), "neutral"

    p1 = _auxiliary_circle_face_hit(face_a, circle, cancel_event)
    p2 = _auxiliary_circle_face_hit(face_b, circle, cancel_event)

    midpoint = (p1 + p2).multiply(0.5)
    chord_length = _length(p1 - p2)
    if chord_length < radius * 1e-4:
        return _probe_result(center, normal_a, normal_b, p1, p2, midpoint, None, chord_length, radius), "neutral"

    midpoint_inside = solid.isInside(midpoint, max(radius * 1e-3, 1e-7))
    convexity = "convex" if midpoint_inside else "concave"
    return (
        _probe_result(
            center,
            normal_a,
            normal_b,
            p1,
            p2,
            midpoint,
            midpoint_inside,
            chord_length,
            radius,
        ),
        convexity,
    )


def _probe_normal(face: cq.Face, point: cq.Vector) -> cq.Vector:
    return _unit(face.normalAt(point))


def _auxiliary_circle(
    center: cq.Vector,
    normal_a: cq.Vector,
    normal_b: cq.Vector,
    radius: float,
) -> tuple[cq.Vector, cq.Vector, cq.Vector, float] | None:
    u = _unit(normal_a)
    v = normal_b - u.multiply(_dot(normal_b, u))
    if _length(v) < 1e-9:
        return None
    return center, u, _unit(v), radius


def _auxiliary_circle_face_hit(
    face: cq.Face,
    circle: tuple[cq.Vector, cq.Vector, cq.Vector, float],
    cancel_event: Any | None,
) -> cq.Vector:
    _check_cancel(cancel_event)
    center, u, v, radius = circle
    plane_normal = _unit(u.cross(v))
    if _length(plane_normal) < 1e-9:
        raise CircleFaceIntersectionError("Auxiliary circle plane is degenerate.")

    circle_edge = cq.Edge.makeCircle(radius, center, plane_normal)
    section = BRepAlgoAPI_Section(face.wrapped, circle_edge.wrapped, False)
    section.Approximation(True)
    section.ComputePCurveOn1(True)
    section.Build()
    if not section.IsDone():
        raise CircleFaceIntersectionError("OpenCascade section operation failed.")

    points = _section_points(section.Shape())
    if not points:
        raise CircleFaceIntersectionError("Auxiliary circle does not intersect adjacent face.")
    return _nearest_to_circle(points, center, radius)


def _section_points(shape: Any) -> list[cq.Vector]:
    points: list[cq.Vector] = []

    vertex_explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    while vertex_explorer.More():
        vertex = TopoDS.Vertex_s(vertex_explorer.Current())
        point = BRep_Tool.Pnt_s(vertex)
        points.append(cq.Vector(point.X(), point.Y(), point.Z()))
        vertex_explorer.Next()

    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while edge_explorer.More():
        edge = cq.Edge(TopoDS.Edge_s(edge_explorer.Current()))
        points.append(edge.positionAt(0.5))
        edge_explorer.Next()

    return _unique_points(points)


def _nearest_to_circle(points: list[cq.Vector], center: cq.Vector, radius: float) -> cq.Vector:
    return min(points, key=lambda point: abs(_length(point - center) - radius))


def _unique_points(points: list[cq.Vector]) -> list[cq.Vector]:
    unique: list[cq.Vector] = []
    for point in points:
        if not any(_length(point - existing) < 1e-7 for existing in unique):
            unique.append(point)
    return unique


def _check_cancel(cancel_event: Any | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise AnalysisCancelled("Analysis cancelled.")


def _probe_result(
    center: cq.Vector,
    normal_a: cq.Vector,
    normal_b: cq.Vector,
    p1: cq.Vector | None,
    p2: cq.Vector | None,
    midpoint: cq.Vector | None,
    midpoint_inside: bool | None,
    chord_length: float | None,
    radius: float,
) -> EdgeProbe:
    return EdgeProbe(
        center=_tuple(center),
        normal_a=_tuple(normal_a),
        normal_b=_tuple(normal_b),
        p1=_tuple(p1) if p1 else None,
        p2=_tuple(p2) if p2 else None,
        midpoint=_tuple(midpoint) if midpoint else None,
        midpoint_inside=midpoint_inside,
        chord_length=chord_length,
        radius=radius,
    )


def _edge_samples(edge: cq.Edge, count: int) -> list[list[float]]:
    count = max(2, count)
    if edge.Closed():
        parameters = [index / count for index in range(count + 1)]
    else:
        parameters = [index / (count - 1) for index in range(count)]
    return [_tuple(edge.positionAt(parameter)) for parameter in parameters]


def _length(vector: cq.Vector) -> float:
    x, y, z = vector.toTuple()
    return math.sqrt(x * x + y * y + z * z)


def _dot(a: cq.Vector, b: cq.Vector) -> float:
    ax, ay, az = a.toTuple()
    bx, by, bz = b.toTuple()
    return ax * bx + ay * by + az * bz


def _tuple(vector: cq.Vector) -> list[float]:
    return [float(value) for value in vector.toTuple()]


def _unit(vector: cq.Vector) -> cq.Vector:
    length = _length(vector)
    return vector if length == 0 else vector.multiply(1.0 / length)
