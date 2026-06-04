# STEP Convex/Concave Edge Viewer

This project does one thing: classify shared STEP edges as convex or concave and visualize those edges on the original solid.

The implementation keeps the current auxiliary-circle edge classifier and removes hole detection, feature grouping, dataset reports, galleries, and generated examples.

## Run

```powershell
.\.venv311\Scripts\python.exe main.py
```

By default, the CLI uses the first `step_datasets/Par*.STEP` file.

Analyze a specific file from the kept dataset:

```powershell
$step = Get-ChildItem step_datasets -Filter "Par*.STEP" | Select-Object -First 1
.\.venv311\Scripts\python.exe main.py $step.FullName
```

Write JSON as well as the viewer:

```powershell
.\.venv311\Scripts\python.exe main.py --json-output edges.json
```

## Desktop UI

Run the desktop interface:

```powershell
.\.venv311\Scripts\python.exe desktop_ui.py
```

The app lists the `Par*.STEP` datasets without analyzing them at startup. Select
a dataset and click `Open Selected`, or click `Choose File` to load another
`.STEP` or `.STP` file. The analysis runs in the background with a progress bar
and estimated remaining time, then displays the solid with convex edges in green
and concave edges in blue. Use `Stop Loading` to cancel a long run; cancellation
takes effect when the current geometry operation yields back to the classifier.

The convex/concave rule follows the paper-style auxiliary-circle test:

1. Take the midpoint of the shared edge.
2. Build a small auxiliary circle from the two adjacent face normals.
3. Intersect that circle with each adjacent face to get one point per face.
4. Take the midpoint between those two circle points.
5. If that midpoint is inside the solid, classify the edge as convex; otherwise
   classify it as concave.

After a file is loaded, use `Solid View` for the 3D edge overlay or `Graph View`
for the face-adjacency graph. Graph nodes are STEP faces; graph edges are shared
face edges colored by classification.

Desktop UI run metadata and JSON outputs are written to `visualization/output/desktop/`.

## Files

- `main.py`: command-line entry point.
- `desktop_ui.py`: desktop interface for loading one STEP file at a time.
- `step_graph.py`: STEP loading and convex/concave edge classification.
- `visualization/viewer.py`: 3D visualization for convex and concave edges.
- `step_datasets/`: `Par*.STEP`, `Cylinder1x1.step`, and `L-bracket.STEP`.

Generated viewer files are written to `visualization/output/`.
