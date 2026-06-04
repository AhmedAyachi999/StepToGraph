from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

import cadquery as cq

from step_graph import ClassifiedEdge, EdgeAnalysis


def write_edge_visualization(
    step_path: str | Path,
    analysis: EdgeAnalysis,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the only supported visualization: convex and concave shared edges."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gltf_path = output_dir / "convex_concave_edges.gltf"
    html_path = output_dir / "convex_concave_edges.html"
    _write_edge_model(Path(step_path), analysis, gltf_path)
    html_path.write_text(_viewer_html(gltf_path, analysis), encoding="utf-8")
    return {"html": html_path, "gltf": gltf_path}


def _write_edge_model(step_path: Path, analysis: EdgeAnalysis, gltf_path: Path) -> None:
    solid = cq.importers.importStep(str(step_path)).val()
    assembly = cq.Assembly()
    assembly.add(solid, name="part", color=cq.Color(0.72, 0.76, 0.82, 0.28))

    radius = _edge_radius(solid)
    for edge in analysis.visible_edges:
        color = _edge_color(edge)
        for index, (start, end) in enumerate(zip(edge.samples, edge.samples[1:]), start=1):
            _add_segment(assembly, start, end, radius, color, f"edge_{edge.id}_{index}")

    assembly.save(str(gltf_path))


def _edge_radius(solid: cq.Solid) -> float:
    box = solid.BoundingBox()
    return max(max(box.xlen, box.ylen, box.zlen) * 0.003, 0.08)


def _edge_color(edge: ClassifiedEdge) -> cq.Color:
    if edge.convexity == "convex":
        return cq.Color(0.10, 0.65, 0.25)
    return cq.Color(0.05, 0.25, 0.95)


def _add_segment(
    assembly: cq.Assembly,
    start: list[float],
    end: list[float],
    radius: float,
    color: cq.Color,
    name: str,
) -> None:
    start_vector = cq.Vector(*start)
    end_vector = cq.Vector(*end)
    direction = end_vector - start_vector
    length = _length(direction)
    if length < 1e-9:
        return

    cylinder = cq.Workplane("XY").cylinder(length, radius)
    z_axis = cq.Vector(0, 0, 1)
    rotation_axis = z_axis.cross(direction)
    if _length(rotation_axis) > 1e-9:
        angle = math.degrees(z_axis.getAngle(direction))
        cylinder = cylinder.rotate((0, 0, 0), rotation_axis.toTuple(), angle)

    midpoint = (start_vector + end_vector).multiply(0.5)
    assembly.add(cylinder.translate(midpoint.toTuple()), name=name, color=color)


def _length(vector: cq.Vector) -> float:
    x, y, z = vector.toTuple()
    return math.sqrt(x * x + y * y + z * z)


def _viewer_html(gltf_path: Path, analysis: EdgeAnalysis) -> str:
    model_src = _inline_gltf(gltf_path)
    title = escape(Path(analysis.source_file).name)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Convex and Concave Edges</title>
  <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
  <style>
    html, body {{ margin: 0; height: 100%; background: #eef2f7; font-family: Segoe UI, Arial, sans-serif; }}
    model-viewer {{ width: 100%; height: 100%; --poster-color: #eef2f7; }}
    .legend {{
      position: fixed;
      top: 16px;
      left: 16px;
      padding: 12px 14px;
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid #d1d5db;
      color: #111827;
      font-size: 13px;
      line-height: 1.45;
    }}
    .title {{ font-weight: 700; margin-bottom: 8px; }}
    .item {{ display: flex; align-items: center; gap: 8px; }}
    .swatch {{ width: 12px; height: 12px; display: inline-block; }}
    .convex {{ background: #1aa63f; }}
    .concave {{ background: #0d40f2; }}
  </style>
</head>
<body>
  <div class="legend">
    <div class="title">{title}</div>
    <div class="item"><span class="swatch convex"></span>Convex edges: {analysis.convex_count}</div>
    <div class="item"><span class="swatch concave"></span>Concave edges: {analysis.concave_count}</div>
  </div>
  <model-viewer src="{model_src}" camera-controls auto-rotate shadow-intensity="0.8"></model-viewer>
</body>
</html>
"""


def _inline_gltf(gltf_path: Path) -> str:
    document = json.loads(gltf_path.read_text(encoding="utf-8"))
    for buffer in document.get("buffers", []):
        uri = buffer.get("uri")
        if uri and not uri.startswith("data:"):
            payload = (gltf_path.parent / uri).read_bytes()
            encoded_payload = base64.b64encode(payload).decode("ascii")
            buffer["uri"] = f"data:application/octet-stream;base64,{encoded_payload}"

    document_json = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded = base64.b64encode(document_json).decode("ascii")
    return f"data:model/gltf+json;base64,{encoded}"
