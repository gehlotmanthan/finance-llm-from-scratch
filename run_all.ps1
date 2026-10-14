<#
Runs the whole project on Windows, chapter by chapter, then rebuilds the results report.

  powershell -ExecutionPolicy Bypass -File .\run_all.ps1            # everything (~1.5-2.5 h on an 8 GB GPU)
  powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Smoke     # 3-minute pipeline check, synthetic data
  powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Only ch06,ch07
  powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -From ch05dapt

Uses the virtual environment one folder up (..\.venv) if it exists, else `python` on PATH.
Each step writes results\<step>.json; a failed step stops the run so you can fix and resume with -From.
#>
param(
    [switch]$Smoke,
    [string[]]$Only,
    [string]$From,
    [switch]$SkipInstall,
    [switch]$SkipTests
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"

$Py = Join-Path (Split-Path -Parent $Root) ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }
Write-Host "Python: $Py" -ForegroundColor Cyan

if (-not $SkipInstall) {
    & $Py -m pip install -q -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
}
& $Py -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

if (-not $SkipTests) {
    & $Py -m pytest
    if ($LASTEXITCODE -ne 0) { throw "unit tests failed" }
}

$steps = [ordered]@{
    "check"       = "scripts\check_setup.py"
    "ch02"        = "scripts\ch02_tokenization.py"
    "ch03"        = "scripts\ch03_attention.py"
    "ch04"        = "scripts\ch04_model.py"
    "ch05scratch" = "scripts\ch05_pretrain_from_scratch.py"
    "ch05gpt2"    = "scripts\ch05_gpt2_domain.py"
    "ch05dapt"    = "scripts\ch05_dapt.py"
    "ch06"        = "scripts\ch06_sentiment.py"
    "ch07"        = "scripts\ch07_instruct.py"
    "score"       = "scripts\capstone_score_news.py"
    "backtest"    = "scripts\capstone_backtest.py"
}

$started = -not $From
foreach ($key in $steps.Keys) {
    if (-not $started -and $key -eq $From) { $started = $true }
    if (-not $started) { continue }
    if ($Only -and -not ($Only -contains $key)) { continue }
    if ($Smoke -and $key -eq "check") { continue }   # the setup check needs the network
    $extra = @()
    if ($Smoke) { $extra += "--smoke" }
    Write-Host "`n==== $key : $($steps[$key]) $extra ====" -ForegroundColor Green
    $t0 = Get-Date
    & $Py $steps[$key] @extra
    if ($LASTEXITCODE -ne 0) { throw "step $key failed (resume with -From $key)" }
    Write-Host ("---- {0} done in {1:N1} min" -f $key, ((Get-Date) - $t0).TotalMinutes) -ForegroundColor DarkGray
}

$reportArgs = @()
if ($Smoke) { $reportArgs += "--smoke" }
& $Py scripts\make_report.py @reportArgs
Write-Host "`nDone. Results: reports\RESULTS.md (smoke runs: results\smoke\RESULTS.md)" -ForegroundColor Cyan
