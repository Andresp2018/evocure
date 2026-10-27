# EVOCURE - Local Virtual Environment Setup Script (Windows PowerShell)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  EVOCURE - Isolated Local Environment Setup (.venv)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$UvCmd = Get-Command uv -ErrorAction SilentlyContinue

if ($UvCmd) {
    Write-Host "[1/4] Detected 'uv' package manager: $($UvCmd.Source)" -ForegroundColor Green
    if (-not (Test-Path $VenvPython)) {
        Write-Host "[2/4] Creating isolated .venv with Python 3.12 via uv ..." -ForegroundColor Yellow
        & uv venv --python 3.12 --seed $VenvDir
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Fallback attempt with Python 3.11..." -ForegroundColor Yellow
            & uv venv --python 3.11 --seed $VenvDir
        }
    } else {
        Write-Host "[2/4] Virtual environment .venv already exists." -ForegroundColor Green
    }
    Write-Host "[3/4] Verifying .venv interpreter ..." -ForegroundColor Green
    & $VenvPython --version
    Write-Host "[4/4] Installing local scientific dependencies strictly into .venv ..." -ForegroundColor Yellow
    & uv pip install --python $VenvPython -r (Join-Path $ProjectRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "Failed to install packages into .venv." }
} else {
    $Candidates = @(
        "$env:APPDATA\uv\python\cpython-3.12-windows-x86_64-none\python.exe",
        "$env:APPDATA\uv\python\cpython-3.11-windows-x86_64-none\python.exe"
    )
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try { $p = (& py -3.12 -c "import sys, encodings, venv; print(sys.executable)" 2>$null); if ($LASTEXITCODE -eq 0 -and $p) { $Candidates += $p.Trim() } } catch {}
        try { $p = (& py -3.11 -c "import sys, encodings, venv; print(sys.executable)" 2>$null); if ($LASTEXITCODE -eq 0 -and $p) { $Candidates += $p.Trim() } } catch {}
    }
    $PythonExe = $null
    foreach ($c in $Candidates) {
        if ($c -and (Test-Path $c)) {
            & $c -c "import encodings, venv" 2>$null
            if ($LASTEXITCODE -eq 0) { $PythonExe = $c; break }
        }
    }
    if (-not $PythonExe) {
        throw "No working Python 3.11 or 3.12 interpreter found on this system."
    }
    Write-Host "[1/4] Validated Python: $PythonExe" -ForegroundColor Green
    & $PythonExe --version
    if (-not (Test-Path $VenvPython)) {
        Write-Host "[2/4] Creating isolated .venv ..." -ForegroundColor Yellow
        & $PythonExe -m venv $VenvDir
    }
    Write-Host "[3/4] Upgrading pip/wheel inside .venv ..." -ForegroundColor Yellow
    & $VenvPython -m pip install --upgrade pip wheel setuptools --disable-pip-version-check
    Write-Host "[4/4] Installing local scientific dependencies inside .venv ..." -ForegroundColor Yellow
    & $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt") --disable-pip-version-check
}

if (-not (Test-Path (Join-Path $ProjectRoot ".env"))) {
    Copy-Item (Join-Path $ProjectRoot ".env.example") (Join-Path $ProjectRoot ".env")
    Write-Host "Created .env from .env.example." -ForegroundColor Green
}

Write-Host ""
Write-Host "Setup complete! To start EVOCURE:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\python.exe run.py" -ForegroundColor White
Write-Host "Or double-click start_evocure.bat" -ForegroundColor White
