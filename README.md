# STEP Convex/Concave Edge Viewer

This project classifies shared STEP edges as convex or concave and opens the STEP file in the desktop OpenCascade viewer.

The edge classifier uses `occwl`'s `Compound`, `EdgeDataExtractor`, and `EdgeConvexity` APIs rather than a local geometric convexity test.

## Run

Use a Python environment with `occwl` and `pythonocc-core` available.

```powershell
.\occwl-env\python.exe main.py
```

By default, the CLI uses the first `step_datasets/Par*.STEP` file.

Analyze a specific file from the kept dataset:

```powershell
$step = Get-ChildItem step_datasets -Filter "Par*.STEP" | Select-Object -First 1
.\occwl-env\python.exe main.py $step.FullName
```

Write JSON:

```powershell
.\occwl-env\python.exe main.py --json-output edges.json
```

## Desktop UI

Open the native OpenCascade viewer directly:

```powershell
.\occwl-env\python.exe desktop_ui.py step_datasets\perfect_L_no_holes.step
```

If no argument is provided, `desktop_ui.py` opens `step_datasets/perfect_L_no_holes.step`.

The viewer has one button: `Color edges`. It overlays all classified edges at
once: convex edges are green, concave edges are blue, and neutral edges are gray.

Convexity is computed from `occwl.edge_data_extractor.EdgeDataExtractor`,
which samples the oriented shared edge, reads the left/right adjacent face
normals, and maps `EdgeConvexity.CONVEX` and `EdgeConvexity.CONCAVE` to the
reported counts. Smooth or unreadable edge data is counted as neutral.

## Files

- `main.py`: command-line entry point.
- `desktop_ui.py`: desktop OpenCascade viewer.
- `step_graph.py`: STEP loading and convex/concave edge classification.
- `step_datasets/`: `Par*.STEP`, `Cylinder1x1.step`, and `L-bracket.STEP`.
