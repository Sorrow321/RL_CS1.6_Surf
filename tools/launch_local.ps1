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
    "maskmm" {
        # The three-part experiment (user, 2026-08-23):
        #   1) NO --obs-reward   2) joint training on 2 maps
        #   3) --surf-mask 2  = the surfability MASK ALONE, no depth channel
        # at 32x16. The march still computes depth to find the hit, so
        # --surf-mask 2 costs exactly what depth-only costs; in_ch stays 1.
        #
        # --respawn-speed 1.0 2.5 is BASELINE here, not a treatment: 1.5 sits
        # just below petrus's ~1,550 u/s ramp gate, and with it the run stalls
        # at 20% for reasons that have nothing to do with perception - the
        # experiment could not answer its own question.
        $run = if ($Arg1) { $Arg1 } else { "xMASKMM" }
        $args_ = @("--maps", "$root\maps\surf_src_cannonball.bsp,$root\maps\surf_petrus_lite.bsp",
                   "--run", $run, "--reward", "race", "--envs", "2048",
                   "--spawn", "platform",
                   "--surf-mask", "2",
                   "--lidar-w", "32", "--lidar-h", "16", "--lidar-cell", "32",
                   "--lidar-range", "11500", "--lidar-near", "2000",
                   "--emb", "512", "--hidden", "448",
                   "--act-every", "3", "--pitch-rate", "1.33", "--teleport-fail",
                   "--lr", "3e-4", "--gamma", "0.9995", "--gae", "0.95",
                   "--clip", "0.2", "--vf", "0.5", "--ent", "0.005",
                   "--epochs", "4",
                   "--ep-ticks", "12000", "--time-pen", "0.005",
                   "--success-bonus", "50", "--finish-k", "0", "--stall-secs", "15",
                   "--race-dist", "geodesic", "--maxvel", "4000",
                   "--train-stride", "1", "--yaw-adaptive",
                   "--respawn-frac", "0.9", "--respawn-margin", "2",
                   "--respawn-reservoir", "100000",
                   "--respawn-speed", "1.0", "2.5",
                   "--int-coef", "0.25", "--int-view", "8", "--int-speed", "3",
                   "--eval-eps", "9", "--eval-greedy-only",
                   "--steps", "6e9", "--ckpt-every", "1e9",
                   "--record-every", "75e6") + $Extra
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
    default { throw "unknown preset '$Preset' (scratch_chunk, scratch_flat, maskmm, resume)" }
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
