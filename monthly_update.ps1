param(
    [string]$PythonExe = "E:\VScode\my_python_project\.venv\Scripts\python.exe",
    [string]$WorkDir = "E:\VScode\my_python_project\stock_fetcher",
    [string]$UserAgent = "stock-universe-builder/1.0 your_email@example.com",
    [int]$AiMaxRows = 1200,
    [double]$AiSleepSec = 0.9,
    [double]$RequestSleepSec = 0.6,
    [int]$MaxRetries = 6,
    [string]$AiApiType = "anthropic-messages",
    [string]$AiBaseUrl = "https://api.kimi.com/coding/v1",
    [string]$AiModel = "kimi-for-coding",
    [switch]$SkipAI,
    [switch]$DryRun,
    [switch]$PushToGit
)

$ErrorActionPreference = "Stop"

function Step($msg) {
    Write-Host ""
    Write-Host "==== $msg ====" -ForegroundColor Cyan
}

function Run-Py([string[]]$argsList) {
    Write-Host "$PythonExe $($argsList -join ' ')" -ForegroundColor DarkGray
    if ($DryRun) {
        return
    }
    & $PythonExe @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $PythonExe)) {
    throw "Python not found: $PythonExe"
}
if (-not (Test-Path $WorkDir)) {
    throw "Work directory not found: $WorkDir"
}

Set-Location $WorkDir
Step "Monthly update start"
Write-Host "WorkDir: $WorkDir"
Write-Host "Python : $PythonExe"
Write-Host "SkipAI : $SkipAI"

$out1 = "E:/VScode/my_python_project/stock_fetcher/output/us_stocks_subsectors.xlsx"
$out2 = "E:/VScode/my_python_project/stock_fetcher/output/us_stocks_subsectors_ai_zh_tw.xlsx"
$out3 = "E:/VScode/my_python_project/stock_fetcher/output/us_stocks_investable_themes.xlsx"
$out4 = "E:/VScode/my_python_project/stock_fetcher/output/us_stocks_investable_themes_zh_tw_fast.xlsx"

Step "1/4 Build US stock universe"
Run-Py @(
    ".\build_us_stock_universe.py",
    "--user-agent", "$UserAgent",
    "--request-sleep-sec", "$RequestSleepSec",
    "--max-retries", "$MaxRetries"
)

if (-not $SkipAI) {
    if ([string]::IsNullOrWhiteSpace($env:KIMI_API_KEY)) {
        throw "KIMI_API_KEY is empty. Run: `$env:KIMI_API_KEY='sk-xxxx'  (or use -SkipAI)."
    }
    Step "2/4 AI enrich + zh-tw"
    Run-Py @(
        ".\enrich_ai_translate_excel.py",
        "--in", "$out1",
        "--out", "$out2",
        "--ai-api-type", "$AiApiType",
        "--ai-base-url", "$AiBaseUrl",
        "--ai-model", "$AiModel",
        "--ai-api-key", "$env:KIMI_API_KEY",
        "--ai-only-missing",
        "--ai-max-rows", "$AiMaxRows",
        "--ai-sleep-sec", "$AiSleepSec"
    )
}
else {
    Step "2/4 Skip AI enrich"
    if (-not (Test-Path $out2)) {
        throw "SkipAI is set but $out2 not found. You need an existing AI output file."
    }
}

Step "3/4 Build investable + themes"
Run-Py @(
    ".\postprocess_investable_and_themes.py",
    "--in", "$out2",
    "--out", "$out3",
    "--min-theme-count", "12",
    "--max-theme-sheets", "200"
)

Step "4/4 Fast zh-tw export"
Run-Py @(
    ".\quick_zh_tw_export.py",
    "--in", "$out3",
    "--out", "$out4"
)

Step "Done"
Write-Host "Generated files:" -ForegroundColor Green
Write-Host " - $out1"
Write-Host " - $out2"
Write-Host " - $out3"
Write-Host " - $out4"

if ($PushToGit) {
    Step "Git commit + push"
    git add output/us_stocks_subsectors.xlsx `
            output/us_stocks_subsectors_ai_zh_tw.xlsx `
            output/us_stocks_investable_themes.xlsx `
            output/us_stocks_investable_themes_zh_tw_fast.xlsx
    git commit -m "monthly data refresh $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-Host
    git push | Out-Host
    Write-Host "Git push finished." -ForegroundColor Green
}
