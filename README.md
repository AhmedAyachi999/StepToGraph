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

- Git
- Python 3.11, available as `py -3.11`
- Internet access for dependency downloads

The CAD packages come from `conda-forge`, so plain `pip install` is not enough
for the CAD environment. The steps below download a project-local Micromamba
executable so you do not need `conda` installed globally.

Clone the project:

```powershell
git clone https://github.com/AhmedAyachi999/StepToGraph.git
cd StepToGraph
```

Check that the required commands are available:

```powershell
py -3.11 --version
```

Download Micromamba into the clone. This creates
`Library\bin\micromamba.exe`, which is the local executable used by the
installer:

```powershell
Invoke-WebRequest -Uri https://micro.mamba.pm/api/micromamba/win-64/latest -OutFile micromamba.tar.bz2
tar xf micromamba.tar.bz2
.\Library\bin\micromamba.exe --version
```

Create the local environments:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\install_dependencies.ps1
```

The installer creates two local folders:

- `occwl-env`: CAD/OpenCascade environment used to open STEP files and run the UI.
- `.venv311`: machine-learning environment used for training and diagnostics.

Verify that both environments were created:

```powershell
Test-Path .\occwl-env\python.exe
Test-Path .\.venv311\Scripts\python.exe
```

Both commands should print `True`.

Run the UI:

```powershell
.\occwl-env\python.exe desktop_ui.py
```

Open a sample STEP file directly:

```powershell
.\occwl-env\python.exe desktop_ui.py step_datasets\perfect_L_no_holes.step
```

If `occwl-env` was not created, create it manually with the environment manager
you installed. With the project-local Micromamba from above, run:

```powershell
.\Library\bin\micromamba.exe create -y -p .\occwl-env -c conda-forge python=3.11 pythonocc-core=7.8.1.1 pip
.\occwl-env\python.exe -m pip install git+https://github.com/AutodeskAILab/occwl.git@v3.0.0
.\occwl-env\python.exe -m pip install -r requirements-ml.txt
```

You can also use `conda`, `mamba`, or a globally installed `micromamba` for the
same create command.

The `occwl-env` and `.venv311` folders are generated locally and are ignored by
Git. Do not push them to GitHub.

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
- `requirements-ml.txt`: ML Python packages.
- `features/cleaning_simulation/`: cleaning simulation and surface sampling.
- `features/edge_classification/`: convex/concave edge detection.
- `features/hole_finding/`: surface-form and inward-cylinder detection.
- `tools/data/cleaning_retention.py`: CSV generation from cleaning simulations.
- `tools/diagnostics/hotspot_features.py`: PCA, feature-importance, and ablation diagnostics.
