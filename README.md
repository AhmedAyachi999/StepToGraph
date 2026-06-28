# STEP Cleaning Hotspot Viewer

This project opens STEP/STP files, runs a cleaning simulation, and uses a trained LightGBM model to predict faces that may stay dirty.

The trained model is:

```text
cache/lightgbm_hotspot_model.txt
```

The desktop launcher is:

```text
desktop_ui.py
```

## Create `occwl-env`

From PowerShell in the project folder:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_occwl_env.ps1
```

The script creates `occwl-env` from `environment-occwl.yml`, installs the pinned `occwl` package, and tests the imports.

A successful setup ends with:

```text
occwl-env ready
```

## If Setup Says Non-Conda Folder Exists

This means `occwl-env` already exists but is not a valid conda environment.

Check it:

```powershell
Test-Path .\occwl-env
Test-Path .\occwl-env\conda-meta
Test-Path .\occwl-env\python.exe
```

If it is broken, remove or rename it, then run setup again:

```powershell
Remove-Item -LiteralPath .\occwl-env -Recurse -Force
.\setup_occwl_env.ps1
```

## Run The UI

Always run the UI with the Python inside `occwl-env`:

```powershell
.\occwl-env\python.exe desktop_ui.py
```

Open a sample directly:

```powershell
.\occwl-env\python.exe desktop_ui.py step_datasets\perfect_L_no_holes.step
```

Do not run the UI with plain `python` or `.venv311`; those environments may not have `occwl`.

## If Setup Reports `occwl` Is Missing

If setup prints a Python traceback such as:

```text
importlib.metadata.PackageNotFoundError: No package metadata was found for occwl
```

force reinstall the pinned `occwl` package:

```powershell
.\setup_occwl_env.ps1 -ForceOccwlReinstall
```

Then test:

```powershell
.\occwl-env\python.exe -c "import occwl; print('occwl ok')"
```

## If The LightGBM Model Is Corrupted

Test the model file:

```powershell
Get-Content .\cache\lightgbm_hotspot_model.txt -TotalCount 5
```

A valid model starts like:

```text
tree
version=...
num_class=1
```

If the file contains numeric rows instead, replace it with the valid `cache/lightgbm_hotspot_model.txt` from this branch.

Then test:

```powershell
.\occwl-env\python.exe -c "import lightgbm as lgb; lgb.Booster(model_file='cache/lightgbm_hotspot_model.txt'); print('model ok')"
```
