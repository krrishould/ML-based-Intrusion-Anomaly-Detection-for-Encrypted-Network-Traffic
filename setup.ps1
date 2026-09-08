# ---------------------------------------------------------------------------
# One-shot setup for the encrypted-traffic IDS project (Windows / PowerShell).
#
#   .\setup.ps1              # environment + dependencies + CTU-13 + train
#   .\setup.ps1 -NoData      # environment + dependencies only
#   .\setup.ps1 -NoTrain     # everything except training
#
# Run from the project root. Does not need administrator rights (Npcap does,
# and is only needed for live capture - see the note at the end).
# ---------------------------------------------------------------------------
param(
    [switch]$NoData,
    [switch]$NoTrain
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

function Step($text) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor DarkCyan
    Write-Host "  $text" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor DarkCyan
}

# --- Python ----------------------------------------------------------------
Step "Checking Python"
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { throw "Python not found on PATH. Install Python 3.10-3.14 first." }

$version = (& python -c "import sys; print('%d.%d' % sys.version_info[:2])")
Write-Host "Found Python $version"
if ([version]$version -lt [version]"3.10") {
    throw "Python 3.10 or newer is required (found $version)."
}

# --- virtual environment ---------------------------------------------------
Step "Creating the virtual environment"
if (Test-Path ".venv") {
    Write-Host ".venv already exists - reusing it"
} else {
    & python -m venv .venv
    Write-Host "Created .venv"
}
$venvPython = Join-Path $root ".venv\Scripts\python.exe"

# --- dependencies ----------------------------------------------------------
Step "Installing dependencies (this takes a few minutes)"
& $venvPython -m pip install --upgrade pip setuptools wheel --quiet
& $venvPython -m pip install -r requirements.txt
& $venvPython -m pip install -e . --quiet
Write-Host "Dependencies installed" -ForegroundColor Green

# --- sanity check ----------------------------------------------------------
Step "Verifying the installation"
& $venvPython -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Test suite failed - stopping." }
Write-Host "All tests passed" -ForegroundColor Green

# --- data ------------------------------------------------------------------
if (-not $NoData) {
    Step "Downloading CTU-13 (approx. 290 MB)"
    & $venvPython scripts\download_datasets.py --dataset ctu13

    Step "Building the feature table"
    & $venvPython scripts\prepare_data.py --force
}

# --- training --------------------------------------------------------------
if (-not $NoTrain) {
    Step "Training the two-stage detector"
    & $venvPython scripts\train_model.py --source synthetic
}

# --- done ------------------------------------------------------------------
Step "Setup complete"
Write-Host @"
Activate the environment in a new shell with:

    .\.venv\Scripts\Activate.ps1

Then:

    python scripts\evaluate_model.py --source synthetic   # full evaluation
    python scripts\run_dashboard.py                       # live dashboard
    python scripts\live_monitor.py --source synthetic --duration 30

LIVE PACKET CAPTURE (optional)
  Sniffing a real interface needs Npcap and an Administrator terminal.
  The installer is bundled at tools\npcap-1.88.exe - right-click, Run as
  administrator, and tick "Install Npcap in WinPcap API-compatible Mode".
  Without it, pcap replay and the synthetic source still work fully.
"@ -ForegroundColor Green
