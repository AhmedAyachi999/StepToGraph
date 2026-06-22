# STEP Cleaning Hotspot Predictor

This project opens STEP/STP CAD files, runs a lightweight cleaning simulation, and uses a trained LightGBM ranking model to predict which faces are most likely to stay dirty.

In the desktop UI:

- Pink faces are the model prediction.
- Simulation colors show where particles stayed after cleaning.
- The trained model is stored at `cache/lightgbm_hotspot_model.txt`.

## Download

Clone the project:

```powershell
git clone https://github.com/AhmedAyachi999/StepToGraph.git
cd StepToGraph
```

Or download it from GitHub as a ZIP file and extract it.

## Install Dependencies

The project has two dependency groups:

- `occwl-env`: STEP/OpenCascade UI environment for reading and displaying CAD files.
- `.venv311`: ML environment for LightGBM training/evaluation.

Run the dependency installer from PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\install_dependencies.ps1
```

The script installs the ML packages from `requirements-ml.txt`. If `Library\bin\micromamba.exe` is available, it can also create the OpenCascade environment with `pythonocc-core` and `occwl`.

Manual install, if needed:

```powershell
.\Library\bin\micromamba.exe create -y -p .\occwl-env -c conda-forge python=3.11 pythonocc-core occwl pip
.\occwl-env\python.exe -m pip install -r requirements-ml.txt

py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install -r requirements-ml.txt
```

## Run The UI

Open the desktop viewer:

```powershell
.\occwl-env\python.exe desktop_ui.py
```

Open a specific STEP file:

```powershell
.\occwl-env\python.exe desktop_ui.py step_datasets\perfect_L_no_holes.step
```

The simplified UI controls are:

| Control | What it does |
| --- | --- |
| `Open STEP File` | Selects a STEP/STP file from disk. |
| `Predict Hotspots` | Runs the trained model and colors predicted hotspot faces in pink. |
| `Run Cleaning Simulation` | Runs the cleaning simulation and colors faces where particles stayed. |
| `Water Force` | Sets cleaning force from `0.000` to `1.000`; default is `0.500`. |
| `Spray Direction` | Chooses all-axis spray or one side: `+X`, `-X`, `+Y`, `-Y`, `+Z`, `-Z`. |

The 3D viewer shows a built-in axis triedron in the lower-right corner.

## Model

The current model is a LightGBM LambdaRank model:

- model class: `LGBMRanker`
- objective: `lambdarank`
- target: `retained_particle_marker_count`
- model file: `cache/lightgbm_hotspot_model.txt`
- inference code: `features/hotspot_prediction.py`

It ranks faces inside each STEP object. The highest-ranked faces are the model's predicted dirty hotspots.

The model uses more than surface type. It uses surface form, face size, cleaning-simulation values, concavity/convexity, boundary shape, and neighbor-face context.

## Optional Training

Generate cleaning-retention CSV files:

```powershell
.\occwl-env\python.exe analyze_cleaning_particle_retention.py
```

Train the ranking model:

```powershell
.\.venv311\Scripts\python.exe train_hotspot_ranker.py
```

The trained model is written to:

```text
cache\lightgbm_hotspot_model.txt
```

## Important Files

- `desktop_ui.py`: simplified desktop UI.
- `install_dependencies.ps1`: dependency installer.
- `requirements-ml.txt`: ML Python packages.
- `features/hotspot_prediction.py`: model inference for new STEP files.
- `features/cleaning_simulation/`: cleaning simulation and surface sampling.
- `features/edge_classification/`: convex/concave edge detection.
- `features/hole_finding/`: surface-form and inward-cylinder detection.
- `train_hotspot_ranker.py`: LightGBM LambdaRank training.
- `analyze_cleaning_particle_retention.py`: CSV generation from cleaning simulations.
