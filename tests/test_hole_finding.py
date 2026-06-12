from __future__ import annotations

import unittest
from pathlib import Path

import cadquery as cq
from cadquery import exporters

from features.hole_finding import FACE_FORM_TYPES, FaceFormFinder, InwardCylinderHeuristic
from features.hole_finding.inward_cylinder import (
    _CylinderSection,
    _intervals_touch,
    _same_axis_line,
)


ROOT = Path(__file__).resolve().parents[1]
GENERATED_STEP_DIR = ROOT / "tmp_hole_finding_fixture"


def section(
    face_id: int,
    *,
    radius: float = 2.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    interval: tuple[float, float] = (0.0, 1.0),
) -> _CylinderSection:
    return _CylinderSection(
        face_id=face_id,
        radius=radius,
        axis_origin=origin,
        axis_direction=direction,
        interval=interval,
        normal_dot_radial=-1.0,
        confidence=0.95,
        warnings=(),
    )


def export_step(part, filename: str) -> Path:
    GENERATED_STEP_DIR.mkdir(exist_ok=True)
    step_path = GENERATED_STEP_DIR / filename
    exporters.export(part, str(step_path), exportType="STEP")
    if not step_path.is_file():
        raise AssertionError(f"CadQuery did not create STEP fixture: {step_path}")
    return step_path


class InwardCylinderHeuristicTests(unittest.TestCase):
    def test_axis_collinearity_rejects_parallel_offset_cylinders(self) -> None:
        self.assertTrue(
            _same_axis_line(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, 5.0),
                (0.0, 0.0, -1.0),
                axis_tolerance=0.01,
                direction_parallel_tolerance=1e-4,
            )
        )
        self.assertFalse(
            _same_axis_line(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.2, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                axis_tolerance=0.01,
                direction_parallel_tolerance=1e-4,
            )
        )

    def test_intervals_touch_only_when_overlapping_or_within_tolerance(self) -> None:
        self.assertTrue(_intervals_touch((0.0, 1.0), (1.04, 2.0), 0.05))
        self.assertFalse(_intervals_touch((0.0, 1.0), (1.2, 2.0), 0.05))

    def test_grouping_requires_same_axis_radius_and_touching_intervals(self) -> None:
        heuristic = InwardCylinderHeuristic(axis_tolerance=0.01, radius_tolerance=0.01, interval_tolerance=0.05)
        groups = heuristic._group_sections(
            [
                section(1, interval=(0.0, 1.0)),
                section(2, interval=(1.02, 2.0)),
                section(3, interval=(3.0, 4.0)),
                section(4, origin=(0.2, 0.0, 0.0), interval=(0.0, 1.0)),
                section(5, radius=2.2, interval=(0.0, 1.0)),
            ]
        )

        merged = [group for group in groups if {item.face_id for item in group.sections} == {1, 2}]
        self.assertEqual(1, len(merged))
        self.assertAlmostEqual(0.0, merged[0].interval[0])
        self.assertAlmostEqual(2.0, merged[0].interval[1])
        self.assertEqual(4, len(groups))

    def test_external_cylinder_fixture_is_not_inward(self) -> None:
        fixture = ROOT / "step_datasets" / "Cylinder1x1.step"
        if fixture.is_file():
            analysis = InwardCylinderHeuristic().find_step(fixture)
            self.assertEqual([], analysis.candidates)

    def test_generated_holes_report_inward_cylinders_without_hole_classification(self) -> None:
        part = (
            cq.Workplane("XY")
            .box(30, 18, 10)
            .faces(">Z")
            .workplane()
            .pushPoints([(-7, 0)])
            .hole(4)
            .pushPoints([(7, 0)])
            .hole(5, depth=4)
        )

        step_path = export_step(part, "holes.step")
        candidates = InwardCylinderHeuristic().find_step(step_path).candidates

        diameters = sorted(round(candidate.diameter, 3) for candidate in candidates)
        self.assertEqual([4.0, 5.0], diameters)
        self.assertTrue(all(candidate.normal_dot_radial < -0.15 for candidate in candidates))

    def test_generated_cylindrical_boss_is_not_inward(self) -> None:
        part = (
            cq.Workplane("XY")
            .box(20, 20, 4)
            .faces(">Z")
            .workplane()
            .circle(2)
            .extrude(5)
        )

        step_path = export_step(part, "boss.step")
        candidates = InwardCylinderHeuristic().find_step(step_path).candidates

        self.assertEqual([], candidates)


class FaceFormFinderTests(unittest.TestCase):
    def test_box_reports_only_plane_faces(self) -> None:
        step_path = export_step(cq.Workplane("XY").box(10, 8, 4), "box.step")
        analysis = FaceFormFinder().find_step(step_path)

        self.assertEqual(6, analysis.count("plane"))
        for form_type in FACE_FORM_TYPES:
            if form_type != "plane":
                self.assertEqual(0, analysis.count(form_type))

    def test_cylinder_reports_cylinder_and_plane_faces(self) -> None:
        step_path = export_step(cq.Workplane("XY").circle(3).extrude(5), "cylinder.step")
        analysis = FaceFormFinder().find_step(step_path)

        self.assertGreaterEqual(analysis.count("plane"), 2)
        self.assertGreaterEqual(analysis.count("cylinder"), 1)
        self.assertEqual(analysis.count("plane"), len(analysis.shape_groups()["plane"]))
        self.assertEqual(analysis.count("cylinder"), len(analysis.shape_groups()["cylinder"]))


if __name__ == "__main__":
    unittest.main()
