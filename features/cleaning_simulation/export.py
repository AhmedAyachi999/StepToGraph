from __future__ import annotations

import html
import json
from dataclasses import asdict
from pathlib import Path

from .math_utils import bounds, clamp
from .models import CleaningSample, CleaningSimulationResult


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


def _heatmap_html(result: CleaningSimulationResult, metric: str) -> str:
    if not result.samples:
        body = "<p>No mesh samples were generated.</p>"
        return _html_document("Cleaning heatmap", body)

    if not hasattr(result.samples[0], metric):
        raise ValueError(f"Unknown heatmap metric: {metric}")

    axis_a, axis_b = _projection_axes(result.samples)
    bounds_a = bounds(sample.point[axis_a] for sample in result.samples)
    bounds_b = bounds(sample.point[axis_b] for sample in result.samples)
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
        low, high = bounds(sample.point[axis] for sample in samples)
        ranges.append((high - low, axis))
    return tuple(axis for _, axis in sorted(ranges, reverse=True)[:2])  # type: ignore[return-value]


def _scale_to_canvas(value: float, value_bounds: tuple[float, float], low: float, high: float) -> float:
    start, end = value_bounds
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
    value = clamp(value)
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
