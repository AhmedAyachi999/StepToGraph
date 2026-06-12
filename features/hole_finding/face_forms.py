from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import (
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
)
from occwl.compound import Compound


FACE_FORM_TYPES = ("plane", "cylinder", "cone", "sphere", "torus", "other")
FACE_FORM_LABELS = {
    "plane": "planes",
    "cylinder": "cylinders",
    "cone": "cones",
    "sphere": "spheres",
    "torus": "tori",
    "other": "other forms",
}

SURFACE_FORM_TYPES = {
    GeomAbs_Plane: "plane",
    GeomAbs_Cylinder: "cylinder",
    GeomAbs_Cone: "cone",
    GeomAbs_Sphere: "sphere",
    GeomAbs_Torus: "torus",
}


@dataclass(frozen=True)
class FaceForm:
    id: int
    face_id: int
    form_type: str


@dataclass(frozen=True)
class FaceFormAnalysis:
    forms: list[FaceForm]
    face_shapes: dict[int, Any]

    def count(self, form_type: str) -> int:
        return sum(form.form_type == form_type for form in self.forms)

    def shape_groups(self) -> dict[str, list[Any]]:
        groups = {form_type: [] for form_type in FACE_FORM_TYPES}
        for form in self.forms:
            shape = self.face_shapes.get(form.face_id)
            if shape is not None:
                groups.setdefault(form.form_type, []).append(shape)
        return groups


class FaceFormFinder:
    """Classify STEP faces by their analytic surface form."""

    def find_step(self, step_path: str | Path) -> FaceFormAnalysis:
        return self.find(Compound.load_from_step(Path(step_path)))

    def find(self, compound: Compound) -> FaceFormAnalysis:
        forms: list[FaceForm] = []
        face_shapes: dict[int, Any] = {}

        for face_id, face in enumerate(compound.faces(), start=1):
            face_shapes[face_id] = face.topods_shape()
            surface_type = BRepAdaptor_Surface(face.topods_shape()).GetType()
            form_type = SURFACE_FORM_TYPES.get(surface_type, "other")
            forms.append(FaceForm(id=face_id, face_id=face_id, form_type=form_type))

        return FaceFormAnalysis(forms=forms, face_shapes=face_shapes)
