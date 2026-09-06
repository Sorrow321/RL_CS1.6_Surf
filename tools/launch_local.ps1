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
#
# VIEW=abs|delta|bins (environment variable; default abs, user-set
# 2026-09-06, CLAUDE.md section 2): the ACTION SPACE of every SCRATCH
# preset. abs = --view-continuous --view-absolute velocity (absolute
# continuous view targets in the velocity frame, docs/contyaw.md: the 97k
# gate in 0.75-1.5B steps on three seeds / three cards against ~3B for the
# bins and >7.7B for per-tick deltas); delta = --view-continuous; bins =
# nothing. scratch_chunk is the one exception: --chunk codes decode into
# BIN distributions and --view-continuous refuses them, so that preset
# stays on the bins whatever VIEW says. The resume preset passes no view
# flag: the trainer restores the mode its checkpoint carries.
#   $env:VIEW = "bins"; powershell -File tools\launch_local.ps1 scratch_ablate xCTL
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

# the default action space of every scratch preset (see the header)
$VIEW = if ($env:VIEW) { $env:VIEW } else { "abs" }
switch ($VIEW) {
    "abs"   { $VIEWARGS = @("--view-continuous", "--view-absolute", "velocity") }
    "delta" { $VIEWARGS = @("--view-continuous") }
    "bins"  { $VIEWARGS = @() }
    default { throw "VIEW must be abs, delta or bins (got '$VIEW')" }
}

switch ($Preset) {
    "scratch_chunk" {
        # BINS ONLY: --chunk codes decode into bin distributions and
        # --view-continuous refuses --chunk, so $VIEWARGS is not applied
        $run = if ($Arg1) { $Arg1 } else { "xCHUNK" }
        Write-Host "== scratch_chunk stays on the discrete bins (--chunk excludes --view-continuous)"
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
                   ) + $EXPLORE + $VIEWARGS + $Extra
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
                   "--record-every", "75e6") + $VIEWARGS + $Extra
    }
    "scratch_ablate" {
        # The user's from-scratch ablation baseline (2026-08-23), identical
        # to run_arm.sh's SCRATCH=1 branch so a local arm and a rented one
        # are the same experiment: cannonball, 64x32 depth, NO --obs-reward,
        # from zero. A scratch run restores nothing from a checkpoint, so the
        # COMPLETE argument set is here - that is the Round 17 failure.
        #
        # NOTE the card: this box is a 5090 and the rented arms are 3090s.
        # CLAUDE.md forbids comparing across card types (the lidar march is
        # not bit-identical across architectures), so a local arm needs a
        # LOCAL control - never borrow a rented one.
        $run = if ($Arg1) { $Arg1 } else { "xABL" }
        # $map, NOT $root\maps: from a worktree the repo-relative bsp is a
        # COPY with a fresh mtime, the cache signature (size + mtime_ns)
        # misses, and the trainer silently re-bakes for ~30 minutes
        # (CLAUDE.md, the worktree trap). The main checkout's bsp is the one
        # every cache next to it was signed against.
        $args_ = @("--map", $map,
                   "--run", $run, "--reward", "race", "--envs", "2048",
                   "--spawn", "platform",
                   "--lidar-w", "64", "--lidar-h", "32", "--lidar-cell", "32",
                   "--lidar-range", "11500", "--lidar-near", "2000",
                   "--emb", "512", "--hidden", "448",
                   "--act-every", "4", "--pitch-rate", "1.33", "--teleport-fail",
                   "--lr", "3e-4", "--gamma", "0.9995", "--gae", "0.95",
                   "--clip", "0.2", "--vf", "0.5", "--ent", "0.005",
                   "--n-steps", "128", "--epochs", "4", "--minibatches", "16",
                   "--ep-ticks", "12000", "--time-pen", "0.005",
                   "--success-bonus", "50", "--finish-k", "0", "--stall-secs", "15",
                   "--race-dist", "geodesic", "--maxvel", "4000",
                   "--train-stride", "1", "--yaw-adaptive",
                   "--respawn-frac", "0.9", "--respawn-margin", "10",
                   "--respawn-reservoir", "100000",
                   "--int-coef", "0.25", "--int-view", "8", "--int-speed", "3",
                   "--steps", "3e9", "--ckpt-every", "1e9",
                   "--record-every", "75e6",
                   "--eval-eps", "9", "--eval-greedy-only") + $VIEWARGS + $Extra
    }
    "resume" {
        if (-not $Arg1 -or -not $Arg2) {
            throw "usage: launch_local.ps1 resume <ckpt> <run-name> [extra]"
        }
        # without this the log lands in runs\_launch.txt for every resumed
        # run and the liveness proof tails the WRONG file
        $run = $Arg2
        # resumes restore their config from the ckpt; extras override. No
        # view flag here on purpose: the checkpoint's own mode is restored
        $args_ = @("--map", $map, "--ckpt", $Arg1, "--run", $Arg2,
                   "--steps", "20e9", "--ckpt-every", "1e9",
                   "--record-every", "250e6") + $Extra
    }
    default { throw "unknown preset '$Preset' (scratch_chunk, scratch_flat, maskmm, scratch_ablate, resume)" }
}

$log = Join-Path $root "runs\$run`_launch.txt"
if ($Preset -ne "resume" -and $Preset -ne "scratch_chunk") {
    Write-Host "== view: $VIEW ($($VIEWARGS -join ' '))"
}
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
