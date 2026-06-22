from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .math_utils import Point3, mean


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
            "mean_hotspot_score": mean(sample.hotspot_score for sample in self.samples),
            "mean_cleaning_dose": mean(sample.cleaning_dose for sample in self.samples),
            "mean_remaining_dust": mean(sample.remaining_dust for sample in self.samples),
            "mean_redeposition": mean(sample.redeposition for sample in self.samples),
            "remaining_particle_count_0_2": sum(sample.remaining_dust >= 0.2 for sample in self.samples),
            "hotspot_count_0_65": sum(sample.hotspot_score >= 0.65 for sample in self.samples),
        }
