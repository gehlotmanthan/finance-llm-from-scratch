<#
Runs the whole pipeline on Windows, step by step, then rebuilds the results report.

  powershell -ExecutionPolicy Bypass -File .\run_all.ps1            # everything (~1.5-2.5 h on an 8 GB GPU)
  powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Smoke     # 3-minute pipeline check, synthetic data
  powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Only sentiment,instruct
  powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -From adapt

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
    "tokenizer"   = "scripts\tokenizer_study.py"
    "attention"   = "scripts\attention_benchmark.py"
    "memory"      = "scripts\memory_profile.py"
    "pretrain"    = "scripts\pretrain_small_gpt.py"
    "domaingap"   = "scripts\domain_gap.py"
    "adapt"       = "scripts\domain_adaptation.py"
    "sentiment"   = "scripts\sentiment_classifier.py"
    "instruct"    = "scripts\instruction_tuning.py"
    "score"       = "scripts\score_headlines.py"
    "backtest"    = "scripts\backtest.py"
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
& $Py scripts\build_report.py @reportArgs
Write-Host "`nDone. Results: reports\RESULTS.md (smoke runs: results\smoke\RESULTS.md)" -ForegroundColor Cyan
