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
$OccwlPackage = "git+https://github.com/AutodeskAILab/occwl.git@v3.0.0"

function Find-CondaEnvironmentManager {
    if (Test-Path -LiteralPath $Micromamba) {
        return $Micromamba
    }

    foreach ($CommandName in @("micromamba", "mamba", "conda")) {
        $Command = Get-Command $CommandName -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($Command) {
            if ($Command.Path) {
                return $Command.Path
            }
            if ($Command.Source) {
                return $Command.Source
            }
            return $Command.Name
        }
    }

    return $null
}

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

function Install-OccwlPackage {
    param(
        [Parameter(Mandatory = $true)][string]$PythonExe
    )

    $EnvironmentRoot = Split-Path -Parent $PythonExe
    $OccwlImportPath = Join-Path $EnvironmentRoot "Lib\site-packages\occwl"
    if (Test-Path -LiteralPath $OccwlImportPath) {
        return
    }

    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "Git is required to install occwl from ${OccwlPackage}."
    }

    Write-Host "Installing occwl from ${OccwlPackage}..."
    & $PythonExe -m pip install $OccwlPackage
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install occwl from ${OccwlPackage}."
    }
}

if (-not (Test-Path -LiteralPath $RequirementsFile)) {
    throw "Missing requirements file: ${RequirementsFile}"
}

if (-not $SkipOccEnv) {
    if (-not (Test-Path -LiteralPath $OccPython)) {
        $EnvironmentManager = Find-CondaEnvironmentManager
        if ($EnvironmentManager) {
            Write-Host "Creating OpenCascade UI environment at ${OccEnvPath} with ${EnvironmentManager}..."
            & $EnvironmentManager create -y -p $OccEnvPath -c conda-forge python=3.11 pythonocc-core=7.8.1.1 pip
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to create OpenCascade UI environment at ${OccEnvPath}."
            }
        }
        else {
            throw "OpenCascade environment not found. Download Micromamba into Library\bin or install micromamba, mamba, or conda, then rerun this script."
        }
    }

    if (Test-Path -LiteralPath $OccPython) {
        Install-OccwlPackage -PythonExe $OccPython
        Install-PipRequirements -PythonExe $OccPython -EnvironmentName "OpenCascade UI environment"
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
