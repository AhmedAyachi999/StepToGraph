from __future__ import annotations

from pathlib import Path

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Core.TopLoc import TopLoc_Location
from occwl.compound import Compound

from .math_utils import centroid, point, scale, triangle_normal_and_area, vertex_key
from .models import CleaningSimulationParameters, SurfaceMesh, SurfaceMeshTriangle


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
            node_index: point(triangulation.Node(node_index).Transformed(transform))
            for node_index in range(1, triangulation.NbNodes() + 1)
        }

        reversed_face = face.topods_shape().Orientation() == TopAbs_REVERSED
        for triangle_index in range(1, triangulation.NbTriangles() + 1):
            node_ids = triangulation.Triangle(triangle_index).Get()
            vertices = tuple(nodes[node_id] for node_id in node_ids)
            normal, area = triangle_normal_and_area(vertices)
            if area <= 1e-12:
                continue
            if reversed_face:
                normal = scale(normal, -1.0)

            triangle_id = len(triangles)
            triangles.append(
                SurfaceMeshTriangle(
                    id=triangle_id,
                    face_id=face_id,
                    vertices=vertices,
                    point=centroid(vertices),
                    normal=normal,
                    area=area,
                )
            )
            triangle_vertex_keys[triangle_id] = tuple(
                vertex_key(vertex, parameters.vertex_merge_tolerance) for vertex in vertices
            )

    return SurfaceMesh(
        source_file=str(step_path.resolve()),
        triangles=triangles,
        neighbors=_build_neighbors(triangle_vertex_keys),
    )


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
