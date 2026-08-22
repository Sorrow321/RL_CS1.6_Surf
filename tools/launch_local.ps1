# launch_local.ps1 - THE canonical local launcher. No hand-typed flag lines.
#
#   powershell -File tools\launch_local.ps1 scratch_chunk
#   powershell -File tools\launch_local.ps1 resume <ckpt> <run-name> [extra flags]
#
# Every preset carries the COMPLETE argument set. The failure this file
# exists to prevent: resumed runs silently restore respawn/intrinsic/etc
# from the checkpoint, so a hand-typed line that "worked" all session is
# missing half its flags the moment there is no checkpoint behind it
# (that is exactly how the first scratch chunk run shipped without
# --respawn-frac and --int-coef and flatlined at -time_pen).
#
# The launcher: prints the full command, starts it detached with the log
# at runs\<run>_launch.txt (UTF-16, read with -Encoding Unicode), waits,
# then PROVES liveness: trainer pid + the log tail. If the process is not
# alive with output after the wait, it exits 1 loudly.
param(
    [Parameter(Mandatory = $true)][string]$Preset,
    [string]$Arg1,
    [string]$Arg2,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Extra
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$map = "C:\RL_Surf\maps\surf_src_cannonball.bsp"

# the exploration infrastructure EVERY run gets unless a preset says why
# not: mid-run respawns + count-based novelty (view+speed keyed cells)
$EXPLORE = @("--respawn-frac", "0.9", "--respawn-margin", "10",
             "--int-coef", "0.25", "--int-view", "8", "--int-speed", "3")

switch ($Preset) {
    "scratch_chunk" {
        $run = if ($Arg1) { $Arg1 } else { "xCHUNK" }
        $args_ = @("--map", $map, "--run", $run, "--reward", "race",
                   "--envs", "2048", "--lidar-w", "64", "--lidar-h", "32",
                   "--steps", "20e9", "--ckpt-every", "1e9",
                   "--record-every", "250e6", "--eval-eps", "9",
                   "--chunk", "10", "--codes", "64",
                   "--n-steps", "16", "--epochs", "8", "--minibatches", "4"
                   ) + $EXPLORE + $Extra
    }
    "scratch_flat" {
        $run = if ($Arg1) { $Arg1 } else { "xFLAT" }
        $args_ = @("--map", $map, "--run", $run, "--reward", "race",
                   "--envs", "2048", "--lidar-w", "64", "--lidar-h", "32",
                   "--steps", "20e9", "--ckpt-every", "1e9",
                   "--record-every", "250e6", "--eval-eps", "9"
                   ) + $EXPLORE + $Extra
    }
    "resume" {
        if (-not $Arg1 -or -not $Arg2) {
            throw "usage: launch_local.ps1 resume <ckpt> <run-name> [extra]"
        }
        # without this the log lands in runs\_launch.txt for every resumed
        # run and the liveness proof tails the WRONG file
        $run = $Arg2
        # resumes restore their config from the ckpt; extras override
        $args_ = @("--map", $map, "--ckpt", $Arg1, "--run", $Arg2,
                   "--steps", "20e9", "--ckpt-every", "1e9",
                   "--record-every", "250e6") + $Extra
    }
    default { throw "unknown preset '$Preset' (scratch_chunk, scratch_flat, resume)" }
}

$log = Join-Path $root "runs\$run`_launch.txt"
Write-Host "== python -u python\train_fast.py $($args_ -join ' ')"
Write-Host "== log: $log"

$before = @(Get-Process python -ErrorAction SilentlyContinue | ForEach-Object Id)
$inner = "Set-Location '$root'; python -u python\train_fast.py " +
         ($args_ -join ' ') + " *> '$log'"
Start-Process powershell -ArgumentList '-NoProfile', '-WindowStyle', 'Hidden',
    '-Command', $inner -WindowStyle Hidden | Out-Null

Start-Sleep -Seconds 45
$py = Get-Process python -ErrorAction SilentlyContinue |
    Where-Object { $before -notcontains $_.Id } |
    Sort-Object StartTime -Descending | Select-Object -First 1
if (-not $py) {
    Write-Host "!! LAUNCH FAILED: no new python process after 45s. Log tail:"
    if (Test-Path $log) { Get-Content $log -Encoding Unicode -Tail 10 }
    exit 1
}
Write-Host ("== ALIVE: trainer pid {0}, started {1}" -f $py.Id, $py.StartTime)
Get-Content $log -Encoding Unicode -Tail 5
Write-Host "== pid $($py.Id)"
