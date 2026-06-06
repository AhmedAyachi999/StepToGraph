import sys
import tkinter as tk
from pathlib import Path

from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Display.tkDisplay import tkViewer3d
from OCC.Extend.DataExchange import read_step_file
from occwl.compound import Compound
from occwl.edge_data_extractor import EdgeConvexity, EdgeDataExtractor


DEFAULT_STEP_FILE = Path("step_datasets/perfect_L_no_holes.step")
BASE_COLOR = Quantity_Color(0.72, 0.75, 0.80, Quantity_TOC_RGB)
EDGE_COLORS = {
    "convex": Quantity_Color(0.0, 0.65, 0.18, Quantity_TOC_RGB),
    "concave": Quantity_Color(0.02, 0.18, 0.95, Quantity_TOC_RGB),
    "neutral": Quantity_Color(0.55, 0.58, 0.62, Quantity_TOC_RGB),
}
CONVEXITY = {EdgeConvexity.CONVEX: "convex", EdgeConvexity.CONCAVE: "concave"}


def view_step_file(filename: str | Path) -> None:
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"STEP file not found: {path}")

    shape = read_step_file(str(path))
    edge_groups = _edge_groups(path)

    root = tk.Tk()
    root.title(f"STEP Viewer - {path.name}")
    root.geometry("1180x820")

    toolbar = tk.Frame(root, padx=8, pady=6)
    toolbar.pack(side=tk.TOP, fill=tk.X)
    status = tk.StringVar(value=_status_text(path, edge_groups, "solid only"))

    viewer = tkViewer3d(root)
    viewer.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    viewer.wait_visibility()
    display = viewer._display

    def show_colored_edges() -> None:
        display.EraseAll()
        display.DisplayShape(shape, color=BASE_COLOR, transparency=0.55, update=False)
        for edge_type, color in EDGE_COLORS.items():
            for ais in display.DisplayShape(edge_groups[edge_type], color=color, update=False):
                display.Context.SetWidth(ais, 4.0, False)
        status.set(_status_text(path, edge_groups, "colored edges"))
        display.Repaint()

    tk.Button(toolbar, text="Color edges", command=show_colored_edges).pack(side=tk.LEFT, padx=(0, 12))
    tk.Label(toolbar, textvariable=status, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

    display.DisplayShape(shape, color=BASE_COLOR, transparency=0.15, update=True)
    root.mainloop()


def _edge_groups(path: Path) -> dict[str, list]:
    groups = {"convex": [], "concave": [], "neutral": []}
    compound = Compound.load_from_step(path)
    for edge in compound.edges():
        faces = list(compound.faces_from_edge(edge))
        if len(faces) != 2:
            continue
        edge_data = EdgeDataExtractor(edge, faces)
        edge_type = "neutral"
        if edge_data.good:
            edge_type = CONVEXITY.get(edge_data.edge_convexity(1e-3), "neutral")
        groups[edge_type].append(edge.topods_shape())
    return groups


def _status_text(path: Path, edge_groups: dict[str, list], mode: str) -> str:
    counts = "   ".join(f"{name}: {len(edges)}" for name, edges in edge_groups.items())
    return f"{path.name}   {mode}   {counts}"


def main() -> int:
    view_step_file(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_STEP_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
