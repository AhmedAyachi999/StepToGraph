import json
from dataclasses import asdict, dataclass
from pathlib import Path

from occwl.compound import Compound
from occwl.edge_data_extractor import EdgeConvexity, EdgeDataExtractor


CONVEXITY = {EdgeConvexity.CONVEX: "convex", EdgeConvexity.CONCAVE: "concave"}


class AnalysisCancelled(Exception):
    pass


@dataclass(frozen=True)
class ClassifiedEdge:
    source_face: int
    target_face: int
    convexity: str
    samples: list[list[float]]


@dataclass(frozen=True)
class EdgeAnalysis:
    source_file: str
    edges: list[ClassifiedEdge]

    @property
    def visible_edges(self) -> list[ClassifiedEdge]:
        return [edge for edge in self.edges if edge.convexity in {"convex", "concave"}]

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def convex_count(self) -> int:
        return self._count("convex")

    @property
    def concave_count(self) -> int:
        return self._count("concave")

    @property
    def neutral_count(self) -> int:
        return self._count("neutral")

    def _count(self, convexity: str) -> int:
        return sum(edge.convexity == convexity for edge in self.edges)


def classify_step_edges(
    step_path: str | Path,
    *,
    edge_sample_count: int = 24,
    cancel_event=None,
) -> EdgeAnalysis:
    _check_cancel(cancel_event)
    step_path = Path(step_path)
    compound = Compound.load_from_step(step_path)
    face_ids = {face: index for index, face in enumerate(compound.faces(), start=1)}
    edges: list[ClassifiedEdge] = []

    edge_sample_count = max(2, edge_sample_count)
    for edge in compound.edges():
        _check_cancel(cancel_event)
        faces = list(compound.faces_from_edge(edge))
        if len(faces) != 2:
            continue

        convexity, samples = "neutral", []
        try:
            edge_data = EdgeDataExtractor(edge, faces, num_samples=edge_sample_count)
            if edge_data.good:
                convexity = CONVEXITY.get(edge_data.edge_convexity(1e-3), "neutral")
                samples = edge_data.points.tolist()
        except Exception:
            convexity, samples = "neutral", []

        edges.append(
            ClassifiedEdge(
                source_face=face_ids[faces[0]],
                target_face=face_ids[faces[1]],
                convexity=convexity,
                samples=samples,
            )
        )

    return EdgeAnalysis(str(step_path.resolve()), edges)


def edge_shape_groups(step_path: str | Path) -> dict[str, list]:
    groups = {"convex": [], "concave": [], "neutral": []}
    compound = Compound.load_from_step(Path(step_path))
    for edge in compound.edges():
        faces = list(compound.faces_from_edge(edge))
        if len(faces) != 2:
            continue

        edge_type = "neutral"
        try:
            edge_data = EdgeDataExtractor(edge, faces)
            if edge_data.good:
                edge_type = CONVEXITY.get(edge_data.edge_convexity(1e-3), "neutral")
        except Exception:
            edge_type = "neutral"
        groups[edge_type].append(edge.topods_shape())
    return groups


def analysis_to_json(analysis: EdgeAnalysis) -> str:
    return json.dumps(
        asdict(analysis)
        | {
            "edge_count": analysis.edge_count,
            "convex_count": analysis.convex_count,
            "concave_count": analysis.concave_count,
            "neutral_count": analysis.neutral_count,
        },
        indent=2,
    )


def _check_cancel(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise AnalysisCancelled("Analysis cancelled.")
