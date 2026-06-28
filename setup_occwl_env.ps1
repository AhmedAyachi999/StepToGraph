[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$MicromambaArchiveUrl = "https://micro.mamba.pm/api/micromamba/win-64/latest"
$MicromambaArchive = Join-Path $ProjectRoot "micromamba.tar.bz2"
$Micromamba = Join-Path $ProjectRoot "Library\bin\micromamba.exe"
$OccEnvPath = Join-Path $ProjectRoot "occwl-env"
$OccPython = Join-Path $OccEnvPath "python.exe"
$RequirementsFile = Join-Path $ProjectRoot "requirements-ml.txt"
$EnvironmentFile = Join-Path $ProjectRoot "environment-occwl.yml"
$OccwlPackage = "git+https://github.com/AutodeskAILab/occwl.git@v3.0.0"

if (-not (Test-Path -LiteralPath $EnvironmentFile)) {
    throw "Missing environment file: ${EnvironmentFile}"
}

if (-not (Test-Path -LiteralPath $RequirementsFile)) {
    throw "Missing requirements file: ${RequirementsFile}"
}

if (-not (Test-Path -LiteralPath $Micromamba)) {
    Write-Host "Downloading project-local Micromamba..."
    Invoke-WebRequest -Uri $MicromambaArchiveUrl -OutFile $MicromambaArchive
    tar -xf $MicromambaArchive
}

if (-not (Test-Path -LiteralPath $OccPython)) {
    Write-Host "Creating occwl-env from environment-occwl.yml..."
    & $Micromamba create -y -p $OccEnvPath -f $EnvironmentFile
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create occwl-env."
    }
}
else {
    Write-Host "Using existing occwl-env."
}

Write-Host "Ensuring occwl and ML packages are installed..."
& $OccPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip in occwl-env."
}

& $OccPython -m pip install $OccwlPackage
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install occwl."
}

& $OccPython -m pip install -r $RequirementsFile
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install requirements-ml.txt."
}

Write-Host "Testing imports..."
& $OccPython -c "from OCC.Core.gp import gp_Pnt; from occwl.compound import Compound; import lightgbm, pandas, sklearn; print('occwl-env ready')"
if ($LASTEXITCODE -ne 0) {
    throw "occwl-env import test failed."
}

Write-Host ""
Write-Host "Run the UI with:"
Write-Host ".\occwl-env\python.exe desktop_ui.py step_datasets\perfect_L_no_holes.step"
