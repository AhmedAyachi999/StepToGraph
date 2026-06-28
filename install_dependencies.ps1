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
$SetupOccwlScript = Join-Path $ProjectRoot "setup_occwl_env.ps1"
$MlEnvPath = Join-Path $ProjectRoot ".venv311"
$MlPython = Join-Path $MlEnvPath "Scripts\python.exe"

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
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upgrade pip in ${EnvironmentName}."
    }

    & $PythonExe -m pip install -r $RequirementsFile
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install ${RequirementsFile} in ${EnvironmentName}."
    }
}

if (-not (Test-Path -LiteralPath $RequirementsFile)) {
    throw "Missing requirements file: ${RequirementsFile}"
}

if (-not $SkipOccEnv) {
    if (-not (Test-Path -LiteralPath $SetupOccwlScript)) {
        throw "Missing setup script: ${SetupOccwlScript}"
    }

    & $SetupOccwlScript
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install OpenCascade UI environment."
    }
}

if (-not $SkipMlEnv) {
    if (-not (Test-Path -LiteralPath $MlPython)) {
        Write-Host "Creating ML environment at ${MlEnvPath}..."
        $PythonLauncher = Get-Command py -ErrorAction SilentlyContinue
        if ($PythonLauncher) {
            & py -3.11 -m venv $MlEnvPath
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to create ML environment with py -3.11."
            }
        }
        else {
            & python -m venv $MlEnvPath
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to create ML environment with python."
            }
        }
    }

    Install-PipRequirements -PythonExe $MlPython -EnvironmentName "ML training environment"
}

Write-Host "Dependency installation finished."
