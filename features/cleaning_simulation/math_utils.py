from __future__ import annotations

import math
from typing import Any, Iterable


Point3 = tuple[float, float, float]


def point(value: Any) -> Point3:
    if hasattr(value, "Coord"):
        x, y, z = value.Coord()
        return float(x), float(y), float(z)
    if hasattr(value, "tolist"):
        value = value.tolist()
    return float(value[0]), float(value[1]), float(value[2])


def rounded_point(value: Point3) -> Point3:
    return round(value[0], 6), round(value[1], 6), round(value[2], 6)


def normalize(vector: Point3) -> Point3:
    length = norm(vector)
    if length <= 1e-12:
        raise ValueError(f"cannot normalize zero-length vector: {vector}")
    return scale(vector, 1.0 / length)


def add(a: Point3, b: Point3) -> Point3:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def sub(a: Point3, b: Point3) -> Point3:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def scale(vector: Point3, scalar: float) -> Point3:
    return vector[0] * scalar, vector[1] * scalar, vector[2] * scalar


def dot(a: Point3, b: Point3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Point3, b: Point3) -> Point3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(vector: Point3) -> float:
    return math.sqrt(dot(vector, vector))


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def normalize_values(values: list[float]) -> list[float]:
    low = min(values, default=0.0)
    high = max(values, default=0.0)
    if high - low <= 1e-12:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def bounds(values: Iterable[float]) -> tuple[float, float]:
    values = list(values)
    if not values:
        return 0.0, 0.0
    return min(values), max(values)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def triangle_normal_and_area(vertices: tuple[Point3, Point3, Point3]) -> tuple[Point3, float]:
    a, b, c = vertices
    area_vector = cross(sub(b, a), sub(c, a))
    length = norm(area_vector)
    if length <= 1e-12:
        return (0.0, 0.0, 1.0), 0.0
    return scale(area_vector, 1.0 / length), 0.5 * length


def centroid(vertices: tuple[Point3, Point3, Point3]) -> Point3:
    return (
        sum(vertex[0] for vertex in vertices) / 3.0,
        sum(vertex[1] for vertex in vertices) / 3.0,
        sum(vertex[2] for vertex in vertices) / 3.0,
    )


def vertex_key(value: Point3, tolerance: float) -> tuple[int, int, int]:
    tolerance = max(tolerance, 1e-12)
    return (
        round(value[0] / tolerance),
        round(value[1] / tolerance),
        round(value[2] / tolerance),
    )
