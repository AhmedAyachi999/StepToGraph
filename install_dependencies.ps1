[CmdletBinding()]
param(
    [switch]$SkipOccEnv,
    [switch]$SkipMlEnv
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$RequirementsFile = Join-Path $ProjectRoot "requirements-ml.txt"
$OccEnvPath = Join-Path $ProjectRoot "occwl-env"
$OccPython = Join-Path $OccEnvPath "python.exe"
$MlEnvPath = Join-Path $ProjectRoot ".venv311"
$MlPython = Join-Path $MlEnvPath "Scripts\python.exe"
$Micromamba = Join-Path $ProjectRoot "Library\bin\micromamba.exe"

function Install-PipRequirements {
    param(
        [Parameter(Mandatory = $true)][string]$PythonExe,
        [Parameter(Mandatory = $true)][string]$EnvironmentName
    )

    if (-not (Test-Path -LiteralPath $PythonExe)) {
        throw "Python executable not found for ${EnvironmentName}: ${PythonExe}"
    }

    Write-Host "Installing Python packages in ${EnvironmentName}..."
    & $PythonExe -m pip install --upgrade pip
    & $PythonExe -m pip install -r $RequirementsFile
}

if (-not (Test-Path -LiteralPath $RequirementsFile)) {
    throw "Missing requirements file: ${RequirementsFile}"
}

if (-not $SkipOccEnv) {
    if (-not (Test-Path -LiteralPath $OccPython)) {
        if (Test-Path -LiteralPath $Micromamba) {
            Write-Host "Creating OpenCascade UI environment at ${OccEnvPath}..."
            & $Micromamba create -y -p $OccEnvPath -c conda-forge python=3.11 pythonocc-core occwl pip
        }
        else {
            Write-Warning "OpenCascade environment not found and micromamba is not available at ${Micromamba}."
            Write-Warning "Install pythonocc-core and occwl manually, or add micromamba and rerun this script."
        }
    }

    if (Test-Path -LiteralPath $OccPython) {
        Install-PipRequirements -PythonExe $OccPython -EnvironmentName "OpenCascade UI environment"
    }
}

if (-not $SkipMlEnv) {
    if (-not (Test-Path -LiteralPath $MlPython)) {
        Write-Host "Creating ML environment at ${MlEnvPath}..."
        $PythonLauncher = Get-Command py -ErrorAction SilentlyContinue
        if ($PythonLauncher) {
            & py -3.11 -m venv $MlEnvPath
        }
        else {
            & python -m venv $MlEnvPath
        }
    }

    Install-PipRequirements -PythonExe $MlPython -EnvironmentName "ML training environment"
}

Write-Host "Dependency installation finished."
