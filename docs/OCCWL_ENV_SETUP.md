# Recreate `occwl-env` After Cloning

`occwl-env` is not committed to GitHub. It is a generated local runtime folder.

Do not run:

```powershell
pip install openCascade
```

`openCascade` is not the package used here. This project needs `pythonocc-core`
from `conda-forge`, plus `occwl`.

## One-command setup

From the project root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

```powershell
.\setup_occwl_env.ps1
```

Then run the UI:

```powershell
.\occwl-env\python.exe desktop_ui.py step_datasets\perfect_L_no_holes.step
```

## Manual setup

Clone the project:

```powershell
git clone https://github.com/AhmedAyachi999/StepToGraph.git
```

```powershell
cd StepToGraph
```

Download Micromamba:

```powershell
Invoke-WebRequest -Uri https://micro.mamba.pm/api/micromamba/win-64/latest -OutFile micromamba.tar.bz2
```

```powershell
tar -xf micromamba.tar.bz2
```

```powershell
.\Library\bin\micromamba.exe --version
```

Create `occwl-env`:

```powershell
.\Library\bin\micromamba.exe create -y -p .\occwl-env -f environment-occwl.yml
```

Install or refresh Python packages:

```powershell
.\occwl-env\python.exe -m pip install git+https://github.com/AutodeskAILab/occwl.git@v3.0.0
```

```powershell
.\occwl-env\python.exe -m pip install -r requirements-ml.txt
```

Test the environment:

```powershell
.\occwl-env\python.exe -c "from OCC.Core.gp import gp_Pnt; from occwl.compound import Compound; import lightgbm, pandas, sklearn; print('occwl-env ready')"
```

Run the UI:

```powershell
.\occwl-env\python.exe desktop_ui.py step_datasets\perfect_L_no_holes.step
```
