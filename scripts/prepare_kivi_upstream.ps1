param(
    [string]$Destination = "artifacts/third_party/KIVI",
    [string]$Repository = "https://github.com/jy-yuan/KIVI.git"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is required to prepare the upstream KIVI source checkout."
}

$parent = Split-Path -Parent $Destination
if ($parent) {
    New-Item -ItemType Directory -Force $parent | Out-Null
}

if (Test-Path (Join-Path $Destination ".git")) {
    git -C $Destination fetch --depth 1 origin main
    git -C $Destination checkout FETCH_HEAD
} else {
    git clone --depth 1 $Repository $Destination
}

$commit = git -C $Destination rev-parse HEAD
Write-Host "Prepared upstream KIVI at $Destination"
Write-Host "Commit: $commit"
Write-Host "Install for real inference:"
Write-Host "  pip install -e $Destination"
Write-Host "  Push-Location $Destination\quant; pip install -e .; Pop-Location"
