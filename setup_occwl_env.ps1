[CmdletBinding()]
param(
    [switch]$ForceOccwlReinstall
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$MicromambaArchiveUrl = "https://micro.mamba.pm/api/micromamba/win-64/latest"
$MicromambaArchive = Join-Path $ProjectRoot "micromamba.tar.bz2"
$Micromamba = Join-Path $ProjectRoot "Library\bin\micromamba.exe"
$OccEnvPath = Join-Path $ProjectRoot "occwl-env"
$OccPython = Join-Path $OccEnvPath "python.exe"
$EnvironmentFile = Join-Path $ProjectRoot "environment-occwl.yml"
$OccwlVersion = "3.0.0"
$OccwlCommit = "8b536ea8b3cf977dbafc1cf0a89eaa28fa996bba"
$OccwlPackage = "git+https://github.com/AutodeskAILab/occwl.git@${OccwlCommit}"

function Test-OccwlPackage {
    param(
        [Parameter(Mandatory = $true)][string]$PythonExe,
        [Parameter(Mandatory = $true)][string]$ExpectedVersion
    )

    if (-not (Test-Path -LiteralPath $PythonExe)) {
        return $false
    }

    & $PythonExe -c "import importlib.metadata as metadata; from occwl.compound import Compound; raise SystemExit(0 if metadata.version('occwl') == '${ExpectedVersion}' else 1)" *> $null
    return $LASTEXITCODE -eq 0
}

if (-not (Test-Path -LiteralPath $EnvironmentFile)) {
    throw "Missing environment file: ${EnvironmentFile}"
}

if (-not (Test-Path -LiteralPath $Micromamba)) {
    Write-Host "Downloading project-local Micromamba..."
    Invoke-WebRequest -Uri $MicromambaArchiveUrl -OutFile $MicromambaArchive
    tar -xf $MicromambaArchive

    if (-not (Test-Path -LiteralPath $Micromamba)) {
        throw "Micromamba download/extract did not create ${Micromamba}."
    }
}

if (-not (Test-Path -LiteralPath $OccPython)) {
    Write-Host "Creating occwl-env from environment-occwl.yml..."
    & $Micromamba create -y -p $OccEnvPath -f $EnvironmentFile
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create occwl-env."
    }
}
else {
    Write-Host "Updating existing occwl-env from environment-occwl.yml..."
    & $Micromamba install -y -p $OccEnvPath -f $EnvironmentFile
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to update occwl-env."
    }
}

if ($ForceOccwlReinstall -or -not (Test-OccwlPackage -PythonExe $OccPython -ExpectedVersion $OccwlVersion)) {
    Write-Host "Installing occwl ${OccwlVersion} from pinned commit without changing conda-managed CAD packages..."
    & $OccPython -m pip install --upgrade --force-reinstall --no-deps $OccwlPackage
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install occwl ${OccwlVersion}."
    }
}
else {
    Write-Host "occwl ${OccwlVersion} is already installed."
}

Write-Host "Testing imports..."
& $OccPython -c "from OCC.Core.gp import gp_Pnt; from occwl.compound import Compound; import cadquery, lightgbm, pandas, sklearn; print('occwl-env ready')"
if ($LASTEXITCODE -ne 0) {
    throw "occwl-env import test failed."
}

Write-Host ""
Write-Host "Run the UI with:"
Write-Host ".\occwl-env\python.exe desktop_ui.py step_datasets\perfect_L_no_holes.step"
