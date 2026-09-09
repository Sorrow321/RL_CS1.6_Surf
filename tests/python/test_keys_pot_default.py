"""The FROM-SCRATCH defaults ``--keys-hold`` + ``--obs-potential norm``
(+ ``--obs-potential-curtain``), user-set 2026-09-09.

Both flags change TENSOR SHAPES - ``--keys-hold`` widens the action head
(NVEC 15,7,3,3,2,2 -> 15,7,4,4,2,3) and ``--obs-potential`` takes the conv
trunk's ``in_ch`` from 1 to 2 - so a default is only ever safe on a launch
that starts from nothing.  What must hold, in the order it would hurt:

  * **the SCRATCH line carries all three flags** and the trainer therefore
    writes ``keys_hold: true`` / ``obs_potential: "norm"`` /
    ``obs_potential_curtain: 1`` into run.json without an arm typing them;
  * **every WARM path carries NONE of them.**  ``train_fast`` refuses a
    ``--keys-hold`` or ``--obs-potential`` that disagrees with the
    checkpoint (it cannot widen a head or a first layer), so a flag leaking
    into the resume branch would break the resume of *every* checkpoint
    trained before this change - sOBSR2, cyABSV, cySPINEW, the exitABS
    round - loudly and for good;
  * **the opt-outs remove exactly their own flags.**  ``KEYS=off`` and
    ``POT=off`` restore the pre-change line byte for byte, because a control
    arm now has to be spelled that way.  ``POT=off`` must drop the CURTAIN
    too: ``--obs-potential-curtain`` without a channel is a hard error.

The live half runs the launcher with a fake ``python3`` on PATH that records
its argv, so these are the arguments the trainer would actually be given,
not a reading of the script.

    python -m pytest tests/python/test_keys_pot_default.py -q
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

RUN_ARM = (ROOT / "tools" / "run_arm.sh").read_text(encoding="utf-8")
LOCAL = (ROOT / "tools" / "launch_local.ps1").read_text(encoding="utf-8")
TRAIN_SRC = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")

# run_arm.sh's three regions. The SCRATCH one is the only one that may name
# the flags; MULTIMAP trains maps this was never measured on and the warm
# branch must keep restoring the checkpoint's own config.
_I_MM = RUN_ARM.index('if [ -n "${MULTIMAP:-}" ]; then')
_I_SCRATCH = RUN_ARM.index('if [ "${SCRATCH:-0}" = "1" ]; then')
_I_WARM = RUN_ARM.index("# ARM_RESUME=1:")
MULTIMAP_BRANCH = RUN_ARM[_I_MM:_I_SCRATCH]
SCRATCH_BRANCH = RUN_ARM[_I_SCRATCH:_I_WARM]
WARM_BRANCH = RUN_ARM[_I_WARM:]
# the warm branch's own argument array - the only thing that reaches the
# trainer there. It may SAY what a resume does with KEYS/POT; it may not
# pass either.
_I_WARG = WARM_BRANCH.index('ARGS=(--ckpt "$CKPT"')
WARM_ARGS = WARM_BRANCH[_I_WARG:WARM_BRANCH.index('"$@")', _I_WARG)]

FLAG_NAMES = ("keys-hold", "keys_hold", "obs-potential", "obs_potential")


# --------------------------------------------------------------- the text --

def test_run_arm_scratch_defaults_to_hold_and_norm():
    assert 'KEYS="${KEYS:-hold}"' in RUN_ARM
    assert 'POT="${POT:-norm}"' in RUN_ARM
    assert "KEYS_ARGS=(--keys-hold)" in RUN_ARM
    assert 'POT_ARGS=(--obs-potential "$POT" --obs-potential-curtain)' in RUN_ARM
    # and the scratch ARGS array actually splices them
    assert '${KEYS_ARGS[@]+"${KEYS_ARGS[@]}"}' in SCRATCH_BRANCH
    assert '${POT_ARGS[@]+"${POT_ARGS[@]}"}' in SCRATCH_BRANCH


def test_run_arm_off_is_an_empty_array_not_a_different_flag():
    """KEYS=off / POT=off must subtract, never substitute."""
    for frag in ("  off)  KEYS_ARGS=()", "  off)  POT_ARGS=()"):
        assert frag in RUN_ARM, frag


def test_run_arm_warm_and_multimap_branches_pass_none_of_them():
    for name in FLAG_NAMES:
        assert name not in WARM_ARGS, f"{name} leaked into the resume ARGS"
        assert name not in MULTIMAP_BRANCH, f"{name} leaked into MULTIMAP"
    assert "KEYS_ARGS" not in WARM_BRANCH and "POT_ARGS" not in WARM_BRANCH
    assert "KEYS_ARGS" not in MULTIMAP_BRANCH
    assert "POT_ARGS" not in MULTIMAP_BRANCH
    # the resume branch says out loud that it ignores them, which is the
    # only mention it is allowed
    assert "KEYS and POT are ignored on a resume" in WARM_BRANCH


def test_launch_local_scratch_ablate_only():
    assert '$KEYS = if ($env:KEYS) { $env:KEYS } else { "hold" }' in LOCAL
    assert '$POT = if ($env:POT) { $env:POT } else { "norm" }' in LOCAL
    assert '$KEYSARGS = @("--keys-hold")' in LOCAL
    assert '$POTARGS = @("--obs-potential", $POT, "--obs-potential-curtain")' \
        in LOCAL
    # spliced into scratch_ablate's argument set, and into no other preset
    assert ") + $VIEWARGS + $KEYSARGS + $POTARGS + $Extra" in LOCAL
    assert LOCAL.count("+ $KEYSARGS") == 1
    assert LOCAL.count("+ $POTARGS") == 1
    i_ab = LOCAL.index('    "scratch_ablate" {')
    i_res = LOCAL.index('    "resume" {')
    assert i_ab < i_res
    i_end = LOCAL.index("    default { throw \"unknown preset")
    for preset, body in (("scratch_chunk",
                          LOCAL[LOCAL.index('    "scratch_chunk" {'):
                                LOCAL.index('    "scratch_flat" {')]),
                         ("scratch_flat",
                          LOCAL[LOCAL.index('    "scratch_flat" {'):
                                LOCAL.index('    "maskmm" {')]),
                         ("maskmm", LOCAL[LOCAL.index('    "maskmm" {'):i_ab]),
                         ("resume", LOCAL[i_res:i_end])):
        assert "KEYSARGS" not in body, preset
        assert "POTARGS" not in body, preset
        for name in FLAG_NAMES:
            assert name not in body, f"{name} in {preset}"
    # and the echo of the two is guarded to scratch_ablate
    assert 'if ($Preset -eq "scratch_ablate") {' in LOCAL
    i_guard = LOCAL.index('if ($Preset -eq "scratch_ablate") {')
    assert LOCAL.index('Write-Host "== keys: $KEYS') > i_guard
    assert LOCAL.index('Write-Host "== pot:  $POT') > i_guard


def test_the_trainer_still_restores_rather_than_takes_a_default():
    """The scoping only holds because the FLAGS default to None in argparse.

    If either ever gained a non-None argparse default, the warm branch would
    start passing it implicitly and every pre-change checkpoint would be
    refused. Pinned here so that change cannot be quiet.
    """
    assert ('ap.add_argument("--keys-hold", action="store_const", const=1, '
            "default=None,") in TRAIN_SRC
    assert 'ap.add_argument("--obs-potential", default=None,' in TRAIN_SRC
    assert ('ap.add_argument("--obs-potential-curtain", action="store_const", '
            'const=1,\n                    default=None,') in TRAIN_SRC
    # and the restore-or-refuse contract both flags ride on
    assert 'if args.keys_hold is None and ck_cfg.get("keys_hold"):' in TRAIN_SRC
    assert ('if args.obs_potential is None and ck_cfg.get("obs_potential"):'
            in TRAIN_SRC)
    assert "--keys-hold changes the action head's SIZE" in TRAIN_SRC
    assert "--obs-potential changes the conv trunk's input channels" in TRAIN_SRC
    # the curtain alone is not a configuration
    assert "--obs-potential-curtain modifies the " in TRAIN_SRC


# --------------------------------------------------------- the actual argv --

BASH = shutil.which("bash")


def _scratch_argv(tmp_path, env_over):
    """Run run_arm.sh's SCRATCH branch with a fake python3 and return argv."""
    shim = tmp_path / "shim"
    shim.mkdir(parents=True)
    argv_out = tmp_path / "argv.txt"
    (shim / "python3").write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$@" > "{argv_out.as_posix()}"\n'
        'echo "step 1 fake"\n'
        "sleep 20\n", encoding="utf-8", newline="\n")
    os.chmod(shim / "python3", 0o755)
    run = "kdARGV"
    env = dict(os.environ)
    env["PATH"] = f"{shim.as_posix()}:{env.get('PATH', '')}"
    env["SCRATCH"] = "1"
    env.update(env_over)
    pid_file = ROOT / "runs" / f"{run}.pid"
    log = ROOT / "runs" / f"{run}_launch.txt"
    try:
        subprocess.run([BASH, "tools/run_arm.sh", run], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=180)
        for _ in range(40):
            if argv_out.exists():
                break
            time.sleep(0.25)
        assert argv_out.exists(), "the launcher never invoked python3"
        return argv_out.read_text(encoding="utf-8").split("\n")
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text().strip()), 9)
            except OSError:
                pass
            pid_file.unlink()
        if log.exists():
            log.unlink()


@pytest.mark.skipif(BASH is None, reason="no bash")
def test_live_scratch_argv_carries_all_three_by_default(tmp_path):
    argv = _scratch_argv(tmp_path, {})
    assert "--keys-hold" in argv
    i = argv.index("--obs-potential")
    assert argv[i + 1] == "norm"
    assert "--obs-potential-curtain" in argv


@pytest.mark.skipif(BASH is None, reason="no bash")
def test_live_keys_off_removes_only_keys(tmp_path):
    argv = _scratch_argv(tmp_path, {"KEYS": "off"})
    assert "--keys-hold" not in argv
    assert "--obs-potential" in argv and "--obs-potential-curtain" in argv


@pytest.mark.skipif(BASH is None, reason="no bash")
def test_live_pot_off_removes_the_channel_and_the_curtain(tmp_path):
    argv = _scratch_argv(tmp_path, {"POT": "off"})
    assert "--obs-potential" not in argv
    assert "--obs-potential-curtain" not in argv
    assert "--keys-hold" in argv


@pytest.mark.skipif(BASH is None, reason="no bash")
def test_live_both_off_is_the_pre_change_line(tmp_path):
    """KEYS=off POT=off must be the launcher exactly as it was."""
    base = _scratch_argv(tmp_path / "a", {"KEYS": "off", "POT": "off"})
    full = _scratch_argv(tmp_path / "b", {})
    added = [a for a in full if a not in base]
    assert set(added) == {"--keys-hold", "--obs-potential", "norm",
                          "--obs-potential-curtain"}, added
    # and nothing was dropped or reordered ahead of the additions
    assert [a for a in full if a in base] == base


@pytest.mark.skipif(BASH is None, reason="no bash")
def test_live_bad_values_are_refused(tmp_path):
    for over in ({"KEYS": "yes"}, {"POT": "on"}):
        env = dict(os.environ, SCRATCH="1", **over)
        r = subprocess.run([BASH, "tools/run_arm.sh", "kdBAD"], cwd=ROOT,
                           env=env, capture_output=True, text=True, timeout=60)
        assert r.returncode == 1, (over, r.stdout, r.stderr)
        assert "must be" in r.stdout
