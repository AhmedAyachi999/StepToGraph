from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

import cadquery as cq


PAPER_TITLE = "Manufacturing feature recognition method based on graph and minimum non-intersection feature volume suppression"
PAPER_URL = "https://doi.org/10.1007/s00170-023-11031-x"
PALETTE = ["#93c5fd", "#fca5a5", "#86efac", "#fdba74", "#c4b5fd", "#f9a8d4", "#67e8f9", "#fde68a", "#d8b4fe", "#a7f3d0"]


def build_attributed_adjacency_graph(step_path: str | Path) -> dict:
    # 1. Read the original solid.
    # 2. Build the face adjacency graph from shared edges.
    # 3. Mark convex closed-loop edges as feature boundaries.
    # 4. Remove those boundaries and keep the remaining connected parts as feature candidates.
    step_path = Path(step_path)
    solid = cq.importers.importStep(str(step_path)).val()
    faces = solid.Faces()
    face_ids = _map_faces_to_step_ids(step_path, faces)
    shared_edges = _shared_edges(solid, faces, face_ids)
    boundary_edge_ids = _boundary_edge_ids(shared_edges)
    groups = _connected_groups(face_ids.values(), shared_edges, boundary_edge_ids)
    face_to_group = {face_id: group["id"] for group in groups for face_id in group["faces"]}

    return {
        "source_file": str(step_path.resolve()),
        "method": {
            "name": "Attributed Adjacency Graph (AAG)",
            "reference_title": PAPER_TITLE,
            "reference_url": PAPER_URL,
            "decomposition_note": "Convex shared edges are treated as feature boundaries only if they form a closed loop.",
        },
        "nodes": sorted(
            [_node(face, face_ids[face.hashCode()], face_to_group.get(face_ids[face.hashCode()])) for face in faces],
            key=lambda n: n["id"],
        ),
        "edges": sorted(
            [_edge(info, boundary_edge_ids) for info in shared_edges],
            key=lambda e: (e["source"], e["target"]),
        ),
        "feature_candidates": groups,
        "stats": {
            "face_count": len(faces),
            "adjacency_count": len(shared_edges),
            "boundary_edge_count": len(boundary_edge_ids),
            "feature_candidate_count": len(groups),
        },
    }


def graph_to_json(graph: dict) -> str:
    return json.dumps(graph, indent=2)


def save_graph_plot(graph: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.write_text(_graph_svg(graph), encoding="utf-8")
    return output_path


def export_feature_candidates(graph: dict, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("*"):
        if old_file.is_file():
            old_file.unlink()

    nodes_by_id = {node["id"]: node for node in graph["nodes"]}
    for group in graph["feature_candidates"]:
        face_ids = set(group["faces"])
        sub_nodes = [nodes_by_id[face_id] for face_id in group["faces"]]
        sub_edges = [edge for edge in graph["edges"] if edge["source"] in face_ids and edge["target"] in face_ids]
        name = _feature_name(sub_nodes)
        subgraph = {
            "source_file": graph["source_file"],
            "method": graph["method"],
            "feature_candidate": {"id": group["id"], "name": name},
            "nodes": sub_nodes,
            "edges": sub_edges,
            "stats": {"face_count": len(sub_nodes), "adjacency_count": len(sub_edges)},
        }
        stem = f"{group['id']:02d}_{name}"
        (output_dir / f"{stem}.json").write_text(graph_to_json(subgraph), encoding="utf-8")
        save_graph_plot(subgraph, output_dir / f"{stem}.svg")
    return output_dir


def export_colored_model(step_path: str | Path, graph: dict, step_out: str | Path, gltf_out: str | Path, html_out: str | Path) -> dict:
    step_path = Path(step_path)
    solid = cq.importers.importStep(str(step_path)).val()
    faces = solid.Faces()
    step_face_ids = _step_face_ids(step_path)
    nodes_by_id = {node["id"]: node for node in graph["nodes"]}

    assembly = cq.Assembly(solid, name="part")
    for i, face in enumerate(faces):
        if i >= len(step_face_ids):
            continue
        face_id = step_face_ids[i]
        group_id = nodes_by_id[face_id]["attributes"]["feature_group"]
        assembly.addSubshape(face, name=f"face_{face_id}", color=_cad_color(group_id))

    step_out = Path(step_out)
    gltf_out = Path(gltf_out)
    html_out = Path(html_out)
    assembly.save(str(step_out))
    assembly.save(str(gltf_out))
    html_out.write_text(_viewer_html(gltf_out.name), encoding="utf-8")
    return {"step": step_out, "gltf": gltf_out, "html": html_out}


def _map_faces_to_step_ids(step_path: Path, faces: list[cq.Face]) -> dict[int, int]:
    step_ids = _step_face_ids(step_path)
    return {face.hashCode(): step_ids[i] if i < len(step_ids) else i + 1 for i, face in enumerate(faces)}


def _step_face_ids(step_path: Path) -> list[int]:
    text = step_path.read_text(encoding="utf-8", errors="replace")
    return [int(x) for x in re.findall(r"#(\d+)\s*=\s*ADVANCED_FACE\s*\(", text)]


def _shared_edges(solid: cq.Solid, faces: list[cq.Face], face_ids: dict[int, int]) -> list[dict]:
    # For each shared edge, classify the local material side:
    # inside probe -> convex, outside probe -> concave, near-zero probe -> smooth.
    edge_faces: dict[int, list[tuple[cq.Face, cq.Edge]]] = defaultdict(list)
    for face in faces:
        for edge in face.Edges():
            if not any(edge.isSame(existing_edge) and face.isSame(existing_face) for existing_face, existing_edge in edge_faces[edge.hashCode()]):
                edge_faces[edge.hashCode()].append((face, edge))

    box = solid.BoundingBox()
    probe_step = max(max(box.xlen, box.ylen, box.zlen) * 1e-3, 1e-3)
    result = []

    for edge_id, refs in edge_faces.items():
        refs = _unique_face_refs(refs)
        if len(refs) != 2:
            continue
        (face_a, edge), (face_b, _) = refs
        point = edge.positionAt(0.5)
        material_a = face_a.normalAt(point).cross(_tangent_in_face(face_a, edge))
        material_b = face_b.normalAt(point).cross(_tangent_in_face(face_b, edge))
        probe = material_a + material_b

        if _length(probe) < 1e-6:
            convexity = "smooth"
        else:
            test_point = point + _unit(probe).multiply(probe_step)
            convexity = "convex" if solid.isInside(test_point.toTuple(), 1e-6) else "concave"

        result.append(
            {
                "edge_id": edge_id,
                "edge": edge,
                "source": face_ids[face_a.hashCode()],
                "target": face_ids[face_b.hashCode()],
                "geometry_type": edge.geomType(),
                "convexity": convexity,
            }
        )
    return result


def _unique_face_refs(refs: list[tuple[cq.Face, cq.Edge]]) -> list[tuple[cq.Face, cq.Edge]]:
    unique = []
    for face, edge in refs:
        if not any(face.isSame(other_face) for other_face, _ in unique):
            unique.append((face, edge))
    return unique


def _tangent_in_face(face: cq.Face, target_edge: cq.Edge) -> cq.Vector:
    for wire in face.Wires():
        for edge in wire.Edges():
            if edge.isSame(target_edge):
                tangent = edge.tangentAt(0.5)
                return -tangent if edge.wrapped.Orientation().name == "TopAbs_REVERSED" else tangent
    return target_edge.tangentAt(0.5)


def _boundary_edge_ids(shared_edges: list[dict]) -> set[int]:
    # Paper idea: do not split on every convex edge.
    # Only keep convex edges that survive the closed-loop filter.
    convex_edges = [info for info in shared_edges if info["convexity"] == "convex"]
    closed_loops = {info["edge_id"] for info in convex_edges if info["edge"].Closed() or len(info["edge"].Vertices()) <= 1}

    vertex_links: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for info in convex_edges:
        if info["edge_id"] in closed_loops:
            continue
        vertices = info["edge"].Vertices()
        if len(vertices) != 2:
            continue
        a = vertices[0].hashCode()
        b = vertices[1].hashCode()
        vertex_links[a].add((b, info["edge_id"]))
        vertex_links[b].add((a, info["edge_id"]))

    active = {info["edge_id"] for info in convex_edges if info["edge_id"] not in closed_loops}
    changed = True
    while changed:
        changed = False
        dangling = [v for v in list(vertex_links) if sum(1 for _, e in vertex_links[v] if e in active) < 2]
        if dangling:
            changed = True
            for vertex in dangling:
                for _, edge_id in vertex_links[vertex]:
                    active.discard(edge_id)
                vertex_links.pop(vertex, None)
    return closed_loops | active


def _connected_groups(face_ids, shared_edges: list[dict], boundary_edge_ids: set[int]) -> list[dict]:
    adjacency = {face_id: set() for face_id in face_ids}
    for info in shared_edges:
        if info["edge_id"] in boundary_edge_ids:
            continue
        adjacency[info["source"]].add(info["target"])
        adjacency[info["target"]].add(info["source"])

    groups = []
    seen = set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        faces = []
        while stack:
            current = stack.pop()
            faces.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        groups.append({"id": len(groups) + 1, "faces": sorted(faces)})
    return groups


def _node(face: cq.Face, face_id: int, group_id: int | None) -> dict:
    return {
        "id": face_id,
        "surface_type": face.geomType(),
        "attributes": {
            "bound_count": len(face.Wires()),
            "edge_curve_count": len(face.Edges()),
            "feature_group": group_id,
        },
    }


def _edge(info: dict, boundary_edge_ids: set[int]) -> dict:
    return {
        "source": min(info["source"], info["target"]),
        "target": max(info["source"], info["target"]),
        "shared_edge_curves": [info["edge_id"]],
        "attributes": {
            "shared_edge_curve_count": 1,
            "shared_edge_geometry": [{"edge_curve_id": info["edge_id"], "geometry_type": info["geometry_type"]}],
            "convexity": info["convexity"],
            "is_feature_boundary": info["edge_id"] in boundary_edge_ids,
        },
    }


def _feature_name(nodes: list[dict]) -> str:
    surface_types = [node["surface_type"] for node in nodes]
    face_count = len(nodes)
    cylinders = surface_types.count("CYLINDER")
    planes = surface_types.count("PLANE")
    if face_count == 1 and cylinders == 1:
        return "round_hole_candidate"
    if cylinders >= 1 and face_count <= 3:
        return "cylindrical_feature_candidate"
    if planes == face_count and face_count <= 2:
        return "step_or_slot_candidate"
    if planes == face_count:
        return "planar_feature_candidate"
    return "feature_candidate"


def _graph_svg(graph: dict) -> str:
    width, height = 1400, 1000
    positions = _force_layout(graph["nodes"], graph["edges"], width, height)

    edge_svg = []
    for edge in graph["edges"]:
        x1, y1 = positions[edge["source"]]
        x2, y2 = positions[edge["target"]]
        color = "#dc2626" if edge["attributes"]["is_feature_boundary"] else "#64748b"
        opacity = "0.9" if edge["attributes"]["is_feature_boundary"] else "0.45"
        stroke = 3.0 if edge["attributes"]["is_feature_boundary"] else 1.6
        edge_svg.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-opacity="{opacity}" stroke-width="{stroke:.2f}" />')

    node_svg, label_svg = [], []
    for node in graph["nodes"]:
        x, y = positions[node["id"]]
        r = 16 + min(node["attributes"]["edge_curve_count"], 8)
        fill = _group_color(node["attributes"]["feature_group"])
        title = escape(f'Face #{node["id"]} | surface={node["surface_type"]} | group={node["attributes"]["feature_group"]}')
        node_svg.append(f'<g><title>{title}</title><circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" stroke="#0f172a" stroke-width="1.5" /></g>')
        label_svg.append(f'<text x="{x:.2f}" y="{y + 4:.2f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="white" stroke="white" stroke-width="4" paint-order="stroke">#{node["id"]}</text>')
        label_svg.append(f'<text x="{x:.2f}" y="{y + 4:.2f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#0f172a">#{node["id"]}</text>')

    title = escape(graph["method"]["name"])
    subtitle = escape(Path(graph["source_file"]).name)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8fafc" />
  <text x="60" y="44" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="700" fill="#0f172a">{title}</text>
  <text x="60" y="72" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#475569">{subtitle}</text>
  <text x="60" y="96" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#475569">Colored nodes = candidate feature groups, red edges = removed boundaries</text>
  <g>{''.join(edge_svg)}</g>
  <g>{''.join(node_svg)}</g>
  <g>{''.join(label_svg)}</g>
</svg>
"""


def _force_layout(nodes: list[dict], edges: list[dict], width: int, height: int, iterations: int = 250) -> dict[int, tuple[float, float]]:
    if not nodes:
        return {}
    node_ids = [node["id"] for node in sorted(nodes, key=lambda n: n["id"])]
    positions = {
        node_id: (
            width / 2 + min(width, height) * 0.25 * math.cos(2 * math.pi * i / len(node_ids)),
            height / 2 + min(width, height) * 0.25 * math.sin(2 * math.pi * i / len(node_ids)),
        )
        for i, node_id in enumerate(node_ids)
    }
    links = [(e["source"], e["target"]) for e in edges if not e["attributes"]["is_feature_boundary"]]
    k = math.sqrt(width * height / max(len(node_ids), 1))
    temp = min(width, height) * 0.12

    for _ in range(iterations):
        disp = {node_id: [0.0, 0.0] for node_id in node_ids}
        for i, a in enumerate(node_ids):
            for b in node_ids[i + 1:]:
                dx = positions[a][0] - positions[b][0]
                dy = positions[a][1] - positions[b][1]
                dist = max(math.hypot(dx, dy), 0.01)
                force = (k * k) / dist
                disp[a][0] += dx / dist * force
                disp[a][1] += dy / dist * force
                disp[b][0] -= dx / dist * force
                disp[b][1] -= dy / dist * force
        for a, b in links:
            dx = positions[a][0] - positions[b][0]
            dy = positions[a][1] - positions[b][1]
            dist = max(math.hypot(dx, dy), 0.01)
            force = (dist * dist) / k
            disp[a][0] -= dx / dist * force
            disp[a][1] -= dy / dist * force
            disp[b][0] += dx / dist * force
            disp[b][1] += dy / dist * force
        for node_id in node_ids:
            dx, dy = disp[node_id]
            dist = max(math.hypot(dx, dy), 0.01)
            step = min(dist, temp)
            x = positions[node_id][0] + dx / dist * step
            y = positions[node_id][1] + dy / dist * step
            positions[node_id] = (min(width - 80, max(80, x)), min(height - 80, max(120, y)))
        temp *= 0.96

    xs = [positions[i][0] for i in node_ids]
    ys = [positions[i][1] for i in node_ids]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    return {
        node_id: (
            120 + (positions[node_id][0] - min_x) * (width - 240) / span_x,
            140 + (positions[node_id][1] - min_y) * (height - 260) / span_y,
        )
        for node_id in node_ids
    }


def _group_color(group_id: int | None) -> str:
    return "#d1d5db" if not group_id else PALETTE[(group_id - 1) % len(PALETTE)]


def _cad_color(group_id: int | None) -> cq.Color:
    hex_color = _group_color(group_id).lstrip("#")
    return cq.Color(int(hex_color[0:2], 16) / 255.0, int(hex_color[2:4], 16) / 255.0, int(hex_color[4:6], 16) / 255.0)


def _viewer_html(model_name: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Colored STEP Features</title>
  <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
  <style>
    html, body {{ margin: 0; height: 100%; background: #e2e8f0; font-family: Segoe UI, Arial, sans-serif; }}
    model-viewer {{ width: 100%; height: 100%; --poster-color: #e2e8f0; }}
    .caption {{ position: fixed; top: 16px; left: 16px; padding: 10px 14px; background: rgba(255,255,255,0.9); border-radius: 10px; color: #0f172a; font-size: 14px; }}
  </style>
</head>
<body>
  <div class="caption">Original solid colored by detected feature group</div>
  <model-viewer src="{model_name}" camera-controls auto-rotate exposure="1.0" shadow-intensity="0.8"></model-viewer>
</body>
</html>
"""


def _length(vector: cq.Vector) -> float:
    x, y, z = vector.toTuple()
    return math.sqrt(x * x + y * y + z * z)


def _unit(vector: cq.Vector) -> cq.Vector:
    length = _length(vector)
    return vector if length == 0 else vector.multiply(1.0 / length)
