# STEP Edge and Inward Cylinder Viewer

This project classifies shared STEP edges as convex or concave, finds inward-facing cylindrical surface candidates with a hole-finding heuristic, and opens the STEP file in the desktop OpenCascade viewer.
It also includes an experimental cleaning simulation that meshes a STEP part, samples surface triangles, starts with dust everywhere, sprays from above and below, runs gravity-based water flow over the surface, and exports hot-spot heatmaps.

The edge classifier uses `occwl`'s `Compound`, `EdgeDataExtractor`, and `EdgeConvexity` APIs rather than a local geometric convexity test.
The features are organized independently under `features/`: edge classification lives in `features/edge_classification/`, and hole finding lives in `features/hole_finding/`.

The hole-finding feature also includes a face-form finder. It classifies faces by analytic surface form, currently planes, cylinders, cones, spheres, tori, and other/freeform surfaces.

The inward-cylinder heuristic does not classify holes as through or blind. It scans cylindrical faces and keeps the ones whose oriented face normal points back toward the cylinder axis, which is the inward/reversed normal pattern expected on an internal cylindrical wall.

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

Run the cleaning simulation and export a standalone heatmap:

```powershell
.\occwl-env\python.exe main.py step_datasets\L-bracket.STEP --simulate-cleaning --cleaning-json-output cleaning.json --heatmap-output cleaning_heatmap.html
```

The cleaning model is intentionally lightweight. It converts the STEP shape to a triangle mesh, uses triangle centroids as sampled surface points, initializes dust to `1` everywhere, sends direct water to upward- and downward-facing surfaces, propagates runoff to downhill mesh neighbors, and scores hot spots from low water dose, poor drainage, redeposition, concavity, and hiddenness.

## Desktop UI

Open the native OpenCascade viewer directly:

```powershell
.\occwl-env\python.exe desktop_ui.py step_datasets\perfect_L_no_holes.step
```

If no argument is provided, `desktop_ui.py` opens `step_datasets/perfect_L_no_holes.step`.

The viewer includes these buttons:

- `Color edges`: overlays all classified edges at once. Convex edges are green, concave edges are blue, and neutral edges are gray.
- `Color inward cylinders`: overlays inward cylindrical candidates in orange.
- `Introduce particles`: meshes the STEP surface and overlays dust particles on sampled surface points.
- `Clean / remaining`: runs the all-direction cleaning simulation with the selected water force, highlights high-risk hot-spot faces, and shows the particles that remain after cleaning.
- `Color planes`, `Color cylinders`, `Color cones`, `Color spheres`, `Color tori`, and `Color other forms`: overlay faces by surface form.

Type a `Water force` value from `0.000` to `1.000` to scale the amount of direct water in the simulation. After pressing `Clean / remaining` once, changing the value refreshes the cleaning view automatically. The UI always sprays from top, bottom, `+X`, `-X`, `+Y`, and `-Y`.

The status text reports both edge counts and the inward cylinder heuristic summary.

Convexity is computed from `occwl.edge_data_extractor.EdgeDataExtractor`,
which samples the oriented shared edge, reads the left/right adjacent face
normals, and maps `EdgeConvexity.CONVEX` and `EdgeConvexity.CONCAVE` to the
reported counts. Smooth or unreadable edge data is counted as neutral.

## Files

- `main.py`: command-line entry point.
- `desktop_ui.py`: desktop OpenCascade viewer.
- `features/edge_classification/`: convex, concave, and neutral edge classification.
- `features/cleaning_simulation/`: sampled surface wash, redeposition, and heatmap export.
- `features/hole_finding/`: inward-cylinder hole-finding heuristic.
- `step_graph.py`: compatibility wrapper for older edge-classification imports.
- `step_datasets/`: `Par*.STEP`, `Cylinder1x1.step`, and `L-bracket.STEP`.
