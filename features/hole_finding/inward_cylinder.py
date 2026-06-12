from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cylinder
from occwl.compound import Compound


Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class InwardCylinderCandidate:
    id: int
    face_ids: list[int]
    radius: float
    diameter: float
    axial_span: float | None
    axis_origin: Point3
    axis_direction: Point3
    confidence: float
    normal_dot_radial: float
    warnings: list[str]


@dataclass(frozen=True)
class InwardCylinderAnalysis:
    candidates: list[InwardCylinderCandidate]
    face_shapes: dict[int, Any]


@dataclass(frozen=True)
class _CylinderSection:
    face_id: int
    radius: float
    axis_origin: Point3
    axis_direction: Point3
    interval: tuple[float, float]
    normal_dot_radial: float
    confidence: float
    warnings: tuple[str, ...]


@dataclass
class _CylinderGroup:
    sections: list[_CylinderSection]
    axis_origin: Point3
    axis_direction: Point3
    radius: float
    interval: tuple[float, float]


class InwardCylinderHeuristic:
    """Find cylindrical faces whose oriented normals point back toward the axis."""

    def __init__(
        self,
        *,
        inward_dot_threshold: float = -0.15,
        axis_tolerance: float = 0.2,
        radius_tolerance: float = 0.05,
        interval_tolerance: float = 0.05,
        angular_tolerance_degrees: float = 1.0,
    ) -> None:
        self.inward_dot_threshold = inward_dot_threshold
        self.axis_tolerance = axis_tolerance
        self.radius_tolerance = radius_tolerance
        self.interval_tolerance = interval_tolerance
        self.direction_parallel_tolerance = 1.0 - math.cos(math.radians(angular_tolerance_degrees))

    def find_step(self, step_path: str | Path) -> InwardCylinderAnalysis:
        return self.find(Compound.load_from_step(Path(step_path)))

    def find(self, compound: Compound) -> InwardCylinderAnalysis:
        face_shapes: dict[int, Any] = {}
        sections: list[_CylinderSection] = []

        for face_id, face in enumerate(compound.faces(), start=1):
            face_shapes[face_id] = face.topods_shape()
            section = self._extract_inward_cylinder(face_id, face)
            if section is not None:
                sections.append(section)

        groups = self._group_sections(sections)
        candidates = [
            self._make_candidate(index, group)
            for index, group in enumerate(groups, start=1)
        ]
        return InwardCylinderAnalysis(candidates=candidates, face_shapes=face_shapes)

    def _extract_inward_cylinder(self, face_id: int, face: Any) -> _CylinderSection | None:
        surface = BRepAdaptor_Surface(face.topods_shape())
        if surface.GetType() != GeomAbs_Cylinder:
            return None

        cylinder = surface.Cylinder()
        radius = float(cylinder.Radius())
        origin = _point(cylinder.Axis().Location())
        direction = _canonical_direction(_point(cylinder.Axis().Direction()))
        warnings: list[str] = []

        point, normal = self._sample_face(face, surface)
        radial = _radial_direction(origin, direction, point)
        if radial is None:
            return None

        normal_dot_radial = _dot(_normalize(normal), radial)
        if normal_dot_radial >= self.inward_dot_threshold:
            return None

        interval = self._axis_interval(face, surface, origin, direction)
        if interval[1] - interval[0] <= self.interval_tolerance:
            warnings.append(f"face {face_id}: cylindrical axial span is near zero")

        confidence = self._confidence(normal_dot_radial)
        if warnings:
            confidence = min(confidence, 0.78)

        return _CylinderSection(
            face_id=face_id,
            radius=radius,
            axis_origin=origin,
            axis_direction=direction,
            interval=interval,
            normal_dot_radial=normal_dot_radial,
            confidence=confidence,
            warnings=tuple(warnings),
        )

    def _sample_face(self, face: Any, surface: BRepAdaptor_Surface) -> tuple[Point3, Point3]:
        u = _mid_parameter(surface.FirstUParameter(), surface.LastUParameter())
        v = _mid_parameter(surface.FirstVParameter(), surface.LastVParameter())
        uv = [u, v]
        return _point(face.point(uv)), _point(face.normal(uv))

    def _axis_interval(
        self,
        face: Any,
        surface: BRepAdaptor_Surface,
        origin: Point3,
        direction: Point3,
    ) -> tuple[float, float]:
        u = _mid_parameter(surface.FirstUParameter(), surface.LastUParameter())
        points = [
            _point(surface.Value(u, surface.FirstVParameter())),
            _point(surface.Value(u, surface.LastVParameter())),
        ]
        points.extend(_point(vertex.point()) for vertex in face.vertices())
        projections = [_project_parameter(point, origin, direction) for point in points]
        return min(projections), max(projections)

    def _group_sections(self, sections: list[_CylinderSection]) -> list[_CylinderGroup]:
        groups: list[_CylinderGroup] = []
        for section in sorted(sections, key=lambda item: (round(item.radius, 6), item.interval[0], item.face_id)):
            for group in groups:
                if self._belongs_to_group(section, group):
                    group.sections.append(section)
                    group.interval = _merge_bounds(group.interval, self._reproject_interval(section, group))
                    break
            else:
                groups.append(
                    _CylinderGroup(
                        sections=[section],
                        axis_origin=section.axis_origin,
                        axis_direction=section.axis_direction,
                        radius=section.radius,
                        interval=section.interval,
                    )
                )
        return groups

    def _belongs_to_group(self, section: _CylinderSection, group: _CylinderGroup) -> bool:
        if abs(section.radius - group.radius) > self.radius_tolerance:
            return False
        if not _same_axis_line(
            group.axis_origin,
            group.axis_direction,
            section.axis_origin,
            section.axis_direction,
            self.axis_tolerance,
            self.direction_parallel_tolerance,
        ):
            return False
        projected_interval = self._reproject_interval(section, group)
        return _intervals_touch(group.interval, projected_interval, self.interval_tolerance)

    def _reproject_interval(self, section: _CylinderSection, group: _CylinderGroup) -> tuple[float, float]:
        start = _point_at(section.axis_origin, section.axis_direction, section.interval[0])
        end = _point_at(section.axis_origin, section.axis_direction, section.interval[1])
        projections = [
            _project_parameter(start, group.axis_origin, group.axis_direction),
            _project_parameter(end, group.axis_origin, group.axis_direction),
        ]
        return min(projections), max(projections)

    def _make_candidate(self, candidate_id: int, group: _CylinderGroup) -> InwardCylinderCandidate:
        start, end = group.interval
        span = end - start
        normal_dot_radial = sum(section.normal_dot_radial for section in group.sections) / len(group.sections)
        warnings = [warning for section in group.sections for warning in section.warnings]
        confidence = min(section.confidence for section in group.sections)

        return InwardCylinderCandidate(
            id=candidate_id,
            face_ids=sorted(section.face_id for section in group.sections),
            radius=group.radius,
            diameter=2.0 * group.radius,
            axial_span=span if span > self.interval_tolerance else None,
            axis_origin=_point_at(group.axis_origin, group.axis_direction, start),
            axis_direction=group.axis_direction,
            confidence=round(confidence, 3),
            normal_dot_radial=round(normal_dot_radial, 3),
            warnings=warnings,
        )

    def _confidence(self, normal_dot_radial: float) -> float:
        threshold_strength = abs(self.inward_dot_threshold)
        alignment = (-normal_dot_radial - threshold_strength) / max(1.0 - threshold_strength, 1e-9)
        return 0.55 + max(0.0, min(alignment, 1.0)) * 0.4


def _same_axis_line(
    origin_a: Point3,
    direction_a: Point3,
    origin_b: Point3,
    direction_b: Point3,
    axis_tolerance: float,
    direction_parallel_tolerance: float,
) -> bool:
    direction_a = _normalize(direction_a)
    direction_b = _normalize(direction_b)
    if 1.0 - abs(_dot(direction_a, direction_b)) > direction_parallel_tolerance:
        return False
    separation = _norm(_cross(_sub(origin_b, origin_a), direction_a))
    return separation <= axis_tolerance


def _intervals_touch(
    interval_a: tuple[float, float],
    interval_b: tuple[float, float],
    tolerance: float,
) -> bool:
    a0, a1 = sorted(interval_a)
    b0, b1 = sorted(interval_b)
    return max(a0, b0) <= min(a1, b1) + tolerance


def _merge_bounds(interval_a: tuple[float, float], interval_b: tuple[float, float]) -> tuple[float, float]:
    return min(interval_a[0], interval_b[0]), max(interval_a[1], interval_b[1])


def _mid_parameter(first: float, last: float) -> float:
    if math.isfinite(first) and math.isfinite(last):
        return (first + last) * 0.5
    if math.isfinite(first):
        return first
    if math.isfinite(last):
        return last
    return 0.0


def _point(value: Any) -> Point3:
    if hasattr(value, "Coord"):
        x, y, z = value.Coord()
        return float(x), float(y), float(z)
    if hasattr(value, "tolist"):
        value = value.tolist()
    return float(value[0]), float(value[1]), float(value[2])


def _canonical_direction(direction: Point3) -> Point3:
    direction = _normalize(direction)
    for component in direction:
        if abs(component) > 1e-9:
            return _scale(direction, -1.0) if component < 0.0 else direction
    return direction


def _radial_direction(origin: Point3, direction: Point3, point: Point3) -> Point3 | None:
    projected = _point_at(origin, direction, _project_parameter(point, origin, direction))
    radial = _sub(point, projected)
    length = _norm(radial)
    if length <= 1e-9:
        return None
    return _scale(radial, 1.0 / length)


def _project_parameter(point: Point3, origin: Point3, direction: Point3) -> float:
    return _dot(_sub(point, origin), _normalize(direction))


def _point_at(origin: Point3, direction: Point3, parameter: float) -> Point3:
    return _add(origin, _scale(_normalize(direction), parameter))


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
