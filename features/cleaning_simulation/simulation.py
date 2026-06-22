from __future__ import annotations

import math
from pathlib import Path

from .math_utils import (
    Point3,
    clamp,
    dot,
    norm,
    normalize,
    normalize_values,
    rounded_point,
    scale,
    sub,
)
from .mesh import step_to_surface_mesh
from .models import (
    CleaningSample,
    CleaningSimulationParameters,
    CleaningSimulationResult,
    SurfaceMesh,
)


def simulate_cleaning(
    step_path: str | Path,
    *,
    parameters: CleaningSimulationParameters | None = None,
) -> CleaningSimulationResult:
    parameters = parameters or CleaningSimulationParameters()
    mesh = step_to_surface_mesh(step_path, parameters=parameters)
    return simulate_cleaning_on_mesh(mesh, parameters=parameters)


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

    gravity = normalize(parameters.gravity_direction)
    water_directions = [normalize(direction) for direction in parameters.water_directions]
    downstream, slopes = _downstream_neighbors(mesh, gravity)
    exposures = [_spray_exposure(triangle.normal, water_directions) for triangle in mesh.triangles]
    concavities = _concavity_scores(mesh)
    poor_drainage = [
        _poor_drainage_score(downstream_id, slope)
        for downstream_id, slope in zip(downstream, slopes)
    ]
    hiddenness = [
        clamp(0.75 * (1.0 - exposure) + 0.25 * concavity)
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
    water_dose_normalized = normalize_values(water_dose)
    redeposition_normalized = normalize_values(redeposition)

    samples: list[CleaningSample] = []
    for triangle in mesh.triangles:
        index = triangle.id
        low_dose = 1.0 - water_dose_normalized[index]
        geometry_risk = clamp(0.55 * hiddenness[index] + 0.45 * concavities[index])
        drainage_risk = poor_drainage[index]
        hotspot_score = clamp(
            0.43 * low_dose
            + 0.25 * drainage_risk
            + 0.2 * geometry_risk
            + 0.12 * redeposition_normalized[index]
        )
        cleaning_dose = clamp(water_dose_normalized[index] * (1.0 - 0.35 * hiddenness[index]))
        samples.append(
            CleaningSample(
                id=triangle.id,
                face_id=triangle.face_id,
                point=rounded_point(triangle.point),
                normal=rounded_point(triangle.normal),
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

            slow_geometry = clamp(
                0.55 * poor_drainage[index]
                + 0.25 * concavities[index]
                + 0.2 * hiddenness[index]
            )
            deposit_fraction = clamp(
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
    return max(clamp(dot(normal, scale(direction, -1.0))) for direction in water_directions)


def _downstream_neighbors(mesh: SurfaceMesh, gravity: Point3) -> tuple[list[int | None], list[float]]:
    downstream: list[int | None] = []
    slopes: list[float] = []

    for triangle in mesh.triangles:
        best_neighbor: int | None = None
        best_slope = 0.0
        for neighbor_id in mesh.neighbors.get(triangle.id, []):
            neighbor = mesh.triangles[neighbor_id]
            offset = sub(neighbor.point, triangle.point)
            distance = norm(offset)
            if distance <= 1e-12:
                continue
            drop = dot(offset, gravity)
            slope = drop / distance
            if drop > 1e-8 and slope > best_slope:
                best_neighbor = neighbor_id
                best_slope = slope
        downstream.append(best_neighbor)
        slopes.append(clamp(best_slope))

    return downstream, slopes


def _poor_drainage_score(downstream_id: int | None, slope: float) -> float:
    if downstream_id is None:
        return 1.0
    return clamp(1.0 - slope / 0.35)


def _concavity_scores(mesh: SurfaceMesh) -> list[float]:
    scores = [0.0] * len(mesh.triangles)

    for triangle in mesh.triangles:
        for neighbor_id in mesh.neighbors.get(triangle.id, []):
            if neighbor_id <= triangle.id:
                continue
            neighbor = mesh.triangles[neighbor_id]
            angle_score = clamp((1.0 - dot(triangle.normal, neighbor.normal)) / 0.7)
            if angle_score <= 0.0:
                continue

            neighbor_offset = sub(neighbor.point, triangle.point)
            if norm(neighbor_offset) <= 1e-12:
                continue
            toward_neighbor = normalize(neighbor_offset)
            toward_triangle = scale(toward_neighbor, -1.0)
            triangle_inside_turn = max(0.0, dot(triangle.normal, toward_neighbor))
            neighbor_inside_turn = max(0.0, dot(neighbor.normal, toward_triangle))
            scores[triangle.id] = max(scores[triangle.id], clamp(angle_score * triangle_inside_turn))
            scores[neighbor.id] = max(scores[neighbor.id], clamp(angle_score * neighbor_inside_turn))

    return scores
