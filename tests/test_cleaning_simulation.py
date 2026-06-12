from __future__ import annotations

import json
import unittest
from pathlib import Path

from features.cleaning_simulation import (
    CleaningSimulationParameters,
    SurfaceMesh,
    SurfaceMeshTriangle,
    export_heatmap_html,
    simulate_cleaning_on_mesh,
    simulation_to_json,
)


ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = ROOT / "tmp_cleaning_simulation_fixture"


def triangle(
    triangle_id: int,
    *,
    point: tuple[float, float, float],
    normal: tuple[float, float, float],
    area: float = 1.0,
    face_id: int | None = None,
) -> SurfaceMeshTriangle:
    return SurfaceMeshTriangle(
        id=triangle_id,
        face_id=face_id or triangle_id + 1,
        vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        point=point,
        normal=normal,
        area=area,
    )


class CleaningSimulationTests(unittest.TestCase):
    def test_hidden_low_dose_surface_scores_hotter_than_exposed_surface(self) -> None:
        mesh = SurfaceMesh(
            source_file="synthetic.step",
            triangles=[
                triangle(0, point=(0.0, 0.0, 1.0), normal=(0.0, 0.0, 1.0)),
                triangle(1, point=(1.0, 0.0, 1.0), normal=(1.0, 0.0, 0.0)),
            ],
            neighbors={0: [], 1: []},
        )

        result = simulate_cleaning_on_mesh(
            mesh,
            parameters=CleaningSimulationParameters(flow_steps=4),
        )

        exposed = result.samples[0]
        hidden = result.samples[1]
        self.assertGreater(exposed.cleaning_dose, hidden.cleaning_dose)
        self.assertGreater(hidden.hiddenness, exposed.hiddenness)
        self.assertGreater(hidden.hotspot_score, exposed.hotspot_score)

    def test_downhill_flow_moves_water_and_redeposition_to_lower_neighbor(self) -> None:
        mesh = SurfaceMesh(
            source_file="synthetic.step",
            triangles=[
                triangle(0, point=(0.0, 0.0, 2.0), normal=(0.0, 0.0, 1.0)),
                triangle(1, point=(0.5, 0.0, 0.0), normal=(1.0, 0.0, 0.0)),
            ],
            neighbors={0: [1], 1: [0]},
        )

        result = simulate_cleaning_on_mesh(
            mesh,
            parameters=CleaningSimulationParameters(flow_steps=6, flow_retention=0.9),
        )

        upper = result.samples[0]
        lower = result.samples[1]
        self.assertEqual(1, upper.downstream_id)
        self.assertIsNone(lower.downstream_id)
        self.assertGreater(lower.redeposition, upper.redeposition)

    def test_default_spray_reaches_above_and_below_facing_surfaces(self) -> None:
        mesh = SurfaceMesh(
            source_file="synthetic.step",
            triangles=[
                triangle(0, point=(0.0, 0.0, 1.0), normal=(0.0, 0.0, 1.0)),
                triangle(1, point=(1.0, 0.0, 1.0), normal=(0.0, 0.0, -1.0)),
                triangle(2, point=(2.0, 0.0, 1.0), normal=(1.0, 0.0, 0.0)),
            ],
            neighbors={0: [], 1: [], 2: []},
        )

        result = simulate_cleaning_on_mesh(mesh, parameters=CleaningSimulationParameters(flow_steps=3))

        upward, downward, vertical = result.samples
        self.assertGreater(upward.cleaning_dose, vertical.cleaning_dose)
        self.assertGreater(downward.cleaning_dose, vertical.cleaning_dose)
        self.assertLess(upward.hotspot_score, vertical.hotspot_score)
        self.assertLess(downward.hotspot_score, vertical.hotspot_score)
        self.assertAlmostEqual(1.0, upward.exposure)
        self.assertAlmostEqual(1.0, downward.exposure)
        self.assertAlmostEqual(0.0, vertical.exposure)

    def test_higher_water_force_leaves_less_remaining_dust(self) -> None:
        mesh = SurfaceMesh(
            source_file="synthetic.step",
            triangles=[
                triangle(0, point=(0.0, 0.0, 1.0), normal=(0.0, 0.0, 1.0)),
            ],
            neighbors={0: []},
        )

        weak = simulate_cleaning_on_mesh(
            mesh,
            parameters=CleaningSimulationParameters(flow_steps=3, water_force=0.4),
        ).samples[0]
        strong = simulate_cleaning_on_mesh(
            mesh,
            parameters=CleaningSimulationParameters(flow_steps=3, water_force=1.8),
        ).samples[0]

        self.assertLess(strong.remaining_dust, weak.remaining_dust)

    def test_json_and_html_exports_include_hotspot_data(self) -> None:
        GENERATED_DIR.mkdir(exist_ok=True)
        mesh = SurfaceMesh(
            source_file="synthetic.step",
            triangles=[
                triangle(0, point=(0.0, 0.0, 1.0), normal=(0.0, 0.0, 1.0)),
                triangle(1, point=(1.0, 0.0, 1.0), normal=(1.0, 0.0, 0.0)),
            ],
            neighbors={0: [], 1: []},
        )
        result = simulate_cleaning_on_mesh(mesh, parameters=CleaningSimulationParameters(flow_steps=2))

        payload = json.loads(simulation_to_json(result))
        self.assertIn("summary", payload)
        self.assertIn("top_hotspots", payload)
        self.assertEqual(2, payload["summary"]["sample_count"])

        html_path = GENERATED_DIR / "heatmap.html"
        export_heatmap_html(result, html_path)
        self.assertTrue(html_path.is_file())
        self.assertIn("Cleaning heatmap", html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
