# STEP Cleaning Hotspot Predictor

This project opens STEP/STP CAD files, runs a lightweight cleaning simulation, and uses a trained LightGBM classifier to predict which faces are likely to stay dirty.

In the desktop UI:

- Pink faces are the model prediction.
- Simulation colors show where particles stayed after cleaning.
- The trained model is stored at `cache/lightgbm_hotspot_model.txt`.

## Project Layout

The project is organized around the shipped path:

```text
stepclean/
  app/          desktop UI
  ml/           training data builder and classifier trainer
  prediction/   model inference for new STEP files
data/training/  CSV files used to train the current model
features/       CAD, cleaning-simulation, edge, and form-detection code
tools/          dataset generation and diagnostics
```

Root files such as `desktop_ui.py` and `train_hotspot_classifier.py` are only launchers. The real code lives in the folders above.

## Fresh Clone Quick Start

These instructions are for Windows PowerShell.

Install these first:

- Git, for cloning the repository and installing the pinned `occwl` source commit
- Internet access for dependency downloads
- Python 3.11 available as `py -3.11`, only if you also want the separate ML training environment

Do not copy or commit `occwl-env`. It is a generated local runtime folder.
OpenCascade must come from `conda-forge`; a plain `pip install` cannot install
this CAD stack correctly.

Clone the project:

```powershell
git clone https://github.com/AhmedAyachi999/StepToGraph.git
cd StepToGraph
```

Create or repair the CAD/UI environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_occwl_env.ps1
```

The script downloads project-local Micromamba if needed, creates or updates
`occwl-env` from `environment-occwl.yml`, installs `occwl` 3.0.0 from a pinned GitHub commit without letting pip change the conda-managed CAD packages, and runs an
import test. A successful run ends with:

```text
occwl-env ready
```

If an old `occwl-env` exists and still fails with an `OCC`, `occwl.compound`, or
`compound_ext`-style import error, rebuild the generated folder:

```powershell
if (Test-Path .\occwl-env) { Remove-Item .\occwl-env -Recurse -Force }
.\setup_occwl_env.ps1
```

Run the UI:

```powershell
.\occwl-env\python.exe desktop_ui.py
```

Open a sample STEP file directly:

```powershell
.\occwl-env\python.exe desktop_ui.py step_datasets\perfect_L_no_holes.step
```

To create both the CAD/UI environment and the separate ML training environment,
run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\install_dependencies.ps1
```

The generated local folders are ignored by Git:

- `occwl-env`: CAD/OpenCascade environment used to open STEP files and run the UI.
- `.venv311`: machine-learning environment used for training and diagnostics.
- `Library`: project-local Micromamba executable.

Verify the environments:

```powershell
Test-Path .\occwl-env\python.exe
Test-Path .\.venv311\Scripts\python.exe
```

See `docs/OCCWL_ENV_SETUP.md` for the same setup in smaller troubleshooting
steps.

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
| `Predict Dirty Faces` | Runs the trained classifier and colors predicted dirty faces in pink. |
| `Run Cleaning Simulation` | Runs the cleaning simulation and colors faces where particles stayed. |
| `Compare Model vs Simulation` | Shows classifier-predicted dirty faces in pink together with simulated retained-particle faces. |
| `Water Force` | Sets cleaning force from `0.000` to `1.000`; default is `0.500`. |
| `Dirty Threshold` | Colors model-predicted dirty faces whose probability is at least this value; default is `0.550`. |
| `Spray Direction` | Chooses all-axis spray or one side: `+X`, `-X`, `+Y`, `-Y`, `+Z`, `-Z`. |

The 3D viewer shows a built-in axis triedron in the lower-right corner.

## Model

The current model is a LightGBM binary classifier:

- model class: `LGBMClassifier`
- objective: `binary`
- target: `stayed_dirty = retained_particle_marker_count > 0`
- features: 10 SHAP-selected cleaning and neighbor-context features
- training data: `data/training/cleaning_retention_wf05/`
- model file: `cache/lightgbm_hotspot_model.txt`
- inference code: `stepclean/prediction/hotspots.py`

It estimates a stayed-dirty probability for every face. The desktop UI colors faces pink when the probability is at least `0.55`.

The compact model uses the strongest cleaning-simulation and neighbor-context factors found during SHAP analysis.

## Optional Training

Generate cleaning-retention CSV files:

```powershell
.\occwl-env\python.exe analyze_cleaning_particle_retention.py
```

Train the stayed-dirty classifier:

```powershell
.\.venv311\Scripts\python.exe train_hotspot_classifier.py
```

The trained model is written to:

```text
cache\lightgbm_hotspot_model.txt
```

## Important Files

- `desktop_ui.py`: launcher for the desktop UI.
- `stepclean/app/desktop.py`: simplified desktop UI.
- `stepclean/prediction/hotspots.py`: model inference for new STEP files.
- `stepclean/ml/classifier.py`: LightGBM stayed-dirty classifier training.
- `stepclean/ml/training_data.py`: shared feature table builder used by training.
- `data/training/cleaning_retention_wf05/`: current training CSVs.
- `install_dependencies.ps1`: dependency installer.
- `environment-occwl.yml`: conda-forge CAD/UI runtime packages.
- `requirements-ml.txt`: bounded ML Python packages for `.venv311`.
- `features/cleaning_simulation/`: cleaning simulation and surface sampling.
- `features/edge_classification/`: convex/concave edge detection.
- `features/hole_finding/`: surface-form and inward-cylinder detection.
- `tools/data/cleaning_retention.py`: CSV generation from cleaning simulations.
- `tools/diagnostics/hotspot_features.py`: PCA, feature-importance, and ablation diagnostics.
