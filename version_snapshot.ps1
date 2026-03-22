param(
    [Parameter(Mandatory = $true)]
    [string]$Version,
    [string]$Note = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$safeVersion = ($Version -replace "[^a-zA-Z0-9\.\-_]", "_")
$tag = "backup/$safeVersion-$stamp"
$snapshotRoot = Join-Path $repoRoot "backups"
$snapshotDir = Join-Path $snapshotRoot "$safeVersion-$stamp"
$zipPath = Join-Path $snapshotRoot "$safeVersion-$stamp.zip"

if (-not (Test-Path $snapshotRoot)) {
    New-Item -ItemType Directory -Path $snapshotRoot | Out-Null
}

$status = git status --porcelain
if ($LASTEXITCODE -ne 0) {
    throw "git status failed."
}

if ($status) {
    Write-Host "[INFO] Working tree has changes. Auto-commit snapshot commit..."
    git add -A
    git commit -m "snapshot: $safeVersion $stamp $Note"
}

git tag -a $tag -m "Snapshot $safeVersion at $stamp. $Note"
if ($LASTEXITCODE -ne 0) {
    throw "git tag failed."
}

New-Item -ItemType Directory -Path $snapshotDir | Out-Null

$include = @(
    "streamlit_app.py",
    "notes_store.py",
    "rss_ingest.py",
    "ai_event_theme.py",
    "data_loader.py",
    "github_sync.py",
    "requirements.txt",
    "README.md",
    "monthly_update.ps1"
)

foreach ($f in $include) {
    $src = Join-Path $repoRoot $f
    if (Test-Path $src) {
        Copy-Item $src -Destination (Join-Path $snapshotDir $f) -Force
    }
}

$outDir = Join-Path $repoRoot "output"
if (Test-Path $outDir) {
    $files = @(
        "us_stocks_investable_themes_zh_tw_fast.xlsx",
        "notes_export.json"
    )
    foreach ($f in $files) {
        $src = Join-Path $outDir $f
        if (Test-Path $src) {
            Copy-Item $src -Destination (Join-Path $snapshotDir $f) -Force
        }
    }
}

if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}
Compress-Archive -Path (Join-Path $snapshotDir "*") -DestinationPath $zipPath -CompressionLevel Optimal

@"
Snapshot created successfully.

Version:    $safeVersion
Tag:        $tag
Folder:     $snapshotDir
Zip:        $zipPath

Restore command:
git checkout $tag -- streamlit_app.py notes_store.py rss_ingest.py ai_event_theme.py data_loader.py github_sync.py requirements.txt README.md monthly_update.ps1

Push tag to GitHub:
git push origin $tag
"@ | Write-Host

