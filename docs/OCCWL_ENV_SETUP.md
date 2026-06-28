# Recreate `occwl-env` After Cloning

`occwl-env` is not committed to GitHub. It is a generated local runtime folder.
If another computer has `OCC`, `occwl.compound`, or `compound_ext` import errors,
recreate this folder instead of trying to upload it.

Do not run:

```powershell
pip install openCascade
```

This project needs `pythonocc-core` and related OpenCascade binaries from
`conda-forge`, plus the `occwl` Python wrapper from a pinned working commit.

## One-command setup

From the project root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_occwl_env.ps1
```

A successful run ends with:

```text
occwl-env ready
```

Then run the UI:

```powershell
.\occwl-env\python.exe desktop_ui.py step_datasets\perfect_L_no_holes.step
```

## Repair a broken local environment

If `occwl-env` already exists but imports still fail, delete only the generated
environment folder and rebuild it:

```powershell
if (Test-Path .\occwl-env) { Remove-Item .\occwl-env -Recurse -Force }
.\setup_occwl_env.ps1
```

You can also force-refresh only the `occwl` Python wrapper:

```powershell
.\setup_occwl_env.ps1 -ForceOccwlReinstall
```

## Manual setup

The script above is preferred. These commands show what it does.

Download Micromamba if `Library\bin\micromamba.exe` is missing:

```powershell
Invoke-WebRequest -Uri https://micro.mamba.pm/api/micromamba/win-64/latest -OutFile micromamba.tar.bz2
tar -xf micromamba.tar.bz2
.\Library\bin\micromamba.exe --version
```

Create or update `occwl-env` from the conda environment file:

```powershell
.\Library\bin\micromamba.exe create -y -p .\occwl-env -f environment-occwl.yml
```

If the environment already exists, update it instead:

```powershell
.\Library\bin\micromamba.exe install -y -p .\occwl-env -f environment-occwl.yml
```

Install the pinned `occwl` wrapper without allowing pip to change the
conda-managed CAD packages:

```powershell
.\occwl-env\python.exe -m pip install --upgrade --force-reinstall --no-deps git+https://github.com/AutodeskAILab/occwl.git@8b536ea8b3cf977dbafc1cf0a89eaa28fa996bba
```

Test the environment:

```powershell
.\occwl-env\python.exe -c "from OCC.Core.gp import gp_Pnt; from occwl.compound import Compound; import cadquery, lightgbm, pandas, sklearn; print('occwl-env ready')"
```

Run the UI:

```powershell
.\occwl-env\python.exe desktop_ui.py step_datasets\perfect_L_no_holes.step
```
