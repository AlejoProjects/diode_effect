param(
    [string]$PythonCommand = "python",
    [string]$EnvironmentDirectory = ".venv"
)

$ErrorActionPreference = "Stop"
$ProjectDirectory = $PSScriptRoot
$EnvironmentPath = Join-Path $ProjectDirectory $EnvironmentDirectory
$EnvironmentPython = Join-Path $EnvironmentPath "Scripts\python.exe"
$RequirementsPath = Join-Path $ProjectDirectory "requirements.txt"

Write-Host "Creating virtual environment at $EnvironmentPath"
& $PythonCommand -m venv $EnvironmentPath

Write-Host "Updating packaging tools"
& $EnvironmentPython -m pip install --upgrade pip setuptools wheel

Write-Host "Installing project dependencies"
& $EnvironmentPython -m pip install --requirement $RequirementsPath

Write-Host "Registering the Jupyter kernel"
& $EnvironmentPython -m ipykernel install --user --name diode-effect --display-name "Python (diode-effect)"

Write-Host "Environment ready. Activate it with:"
Write-Host "$EnvironmentPath\Scripts\Activate.ps1"
Write-Host "Then start JupyterLab with: python -m jupyterlab"
