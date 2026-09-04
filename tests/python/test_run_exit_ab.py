"""tools/wave/run_exit_ab.sh against a FAKE box.

The launcher's whole job is to get one string onto a rented machine intact,
and the reuse scripts before it broke on exactly that twice. So this runs
the real script with `SSH_CMD` pointed at a local shell and `BOXROOT` at a
temp directory, with stub `scp` / `python3` / `expert_loop.py` /
`dashboard.py` / `box_watchdog.sh`, and asserts what actually arrived:

  1. the checkpoint lands at <boxroot>/RL_Surf/runs_seed.pt and its MD5 is
     verified ON THE BOX (CLAUDE.md: scp can truncate a 150 MB file and
     still exit 0), and a CORRUPTED transfer is caught and refused;
  2. the trainer flags reach expert_loop's argv VERBATIM, as separate
     tokens, LAST on the line and immediately after --train-extra (it is
     argparse.REMAINDER: anything after it belongs to the trainer);
  3. a flag with a SPACE in it survives as one token - the quoting bug
     class this file exists for;
  4. the driver's pid is in runs/<name>.pid and the on-box watchdog is
     started with <instance> <name> <deadline> 40;
  5. the harvest spec registered is the expert-loop one
     (--harvest-only-extra, the pid file, the summary and the newest
     round's artifacts), at a deadline UNDER the on-box one.

    python -m pytest tests/python/test_run_exit_ab.py -q
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "wave" / "run_exit_ab.sh"
BASH = next((p for p in (r"C:\Program Files\Git\bin\bash.exe",
                         r"C:\Program Files (x86)\Git\bin\bash.exe",
                         shutil.which("bash") or "")
             if p and Path(p).exists()), None)
needs_bash = pytest.mark.skipif(BASH is None, reason="needs bash")


def _msys(p: Path) -> str:
    """C:\\x -> /c/x, the path form git-bash's own tools take."""
    s = str(Path(p).resolve()).replace("\\", "/")
    if len(s) > 1 and s[1] == ":":
        s = "/" + s[0].lower() + s[2:]
    return s


def _fake_box(tmp: Path, seed_bytes=b"seed-payload" * 4096):
    """A directory that behaves enough like /root on a deployed box."""
    box = tmp / "box"
    (box / "RL_Surf" / "tools").mkdir(parents=True)
    (box / "RL_Surf" / "runs").mkdir()
    (box / "RL_Surf" / "maps").mkdir()
    # git log --oneline -1 runs in the remote script; give it a repo
    subprocess.run(["git", "init", "-q"], cwd=str(box / "RL_Surf"), check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-q", "--allow-empty", "-m", "fake"],
                   cwd=str(box / "RL_Surf"), check=True)

    # the stub expert_loop: record argv, then sit alive so the pid check and
    # the watchdog have something real to look at
    (box / "RL_Surf" / "tools" / "expert_loop.py").write_text(
        "import json, sys, time\n"
        "open('runs/argv.json', 'w').write(json.dumps(sys.argv))\n"
        "print('stub expert_loop up', flush=True)\n"
        "time.sleep(120)\n", encoding="utf-8")
    (box / "RL_Surf" / "tools" / "dashboard.py").write_text(
        "import time; print('stub dashboard'); time.sleep(120)\n",
        encoding="utf-8")
    (box / "RL_Surf" / "tools" / "restamp_maps.py").write_text(
        "print('0 maps restamped')\n", encoding="utf-8")
    # the local seed the launcher ships
    ck = tmp / "ckpt_1.pt"
    ck.write_bytes(seed_bytes)
    return box, ck


def _stubs(tmp: Path, box: Path, corrupt: bool = False):
    """scp that copies into the fake box (and a box_watchdog recorder in
    place of the real self-destruct loop), plus a fleet_watchdog recorder."""
    binm = tmp / "bin"
    binm.mkdir()
    # A fake scp: `<src> root@<host>:<dst>` (the -q/-P pair lives in the
    # SCP_CMD prefix, so what reaches here is exactly the two operands).
    # box_watchdog.sh is replaced by a RECORDER: the real one is an infinite
    # self-destruct loop and a test must never start one.
    truncate = 'printf x >> "$dst"   # a TRUNCATED transfer' if corrupt \
        else 'true'
    (binm / "scp").write_text(f"""#!/bin/bash
src="$1"; dst="${{2#*:}}"
echo "$src -> $dst" >> '{_msys(tmp / "scp.log")}'
if [ "$(basename "$src")" = box_watchdog.sh ]; then
  cat > "$dst" <<'WD'
#!/bin/bash
echo "$@" > "$(dirname "$0")/watchdog_args.txt"
echo "stub watchdog up" >> "$(dirname "$0")/box_watchdog.log"
WD
else
  cp "$src" "$dst"
  {truncate}
fi
""", encoding="utf-8", newline="\n")
    # the registry: record the register lines instead of touching the fleet
    main = tmp / "main"
    (main / "tools").mkdir(parents=True)
    (main / "tools" / "fleet_watchdog.py").write_text(
        "import sys\n"
        "open(r'%s', 'a', encoding='utf-8').write(' '.join(sys.argv[1:]) "
        "+ '\\n')\n"
        "print('registered')\n" % str(tmp / "register.log"),
        encoding="utf-8")
    key = tmp / "fake_api_key"
    key.write_text("not-a-real-key\n", encoding="utf-8")
    for f in (binm / "scp",):
        os.chmod(f, 0o755)
    return binm, main, key


def _run(tmp, box, ck, binm, main, key, flags, check=True):
    env = dict(os.environ)
    env.update(
        BOXROOT=_msys(box), MAIN=_msys(main), VAST_KEY=_msys(key),
        # the two overrides that make the whole script runnable locally
        SSH_CMD="bash -c", SCP_CMD="bash " + _msys(binm / "scp"),
        PY3=sys.executable.replace("\\", "/"),
        PYTHONIOENCODING="utf-8",
        # the stub tools are plain python; the remote script calls python3
        THREADS="4")
    return subprocess.run(
        [BASH, _msys(SCRIPT), "41234", "fake.vast.ai", "48512345",
         "exitTPT", "0.02", _msys(ck)] + flags,
        capture_output=True, text=True, env=env, cwd=str(ROOT),
        timeout=300, check=False)


@needs_bash
def test_the_remote_script_is_well_formed_and_puts_train_extra_last():
    """PRINT_ONLY: the exact string that would reach the box. Cheap enough
    to run without a fake box, and it catches an unbalanced quote."""
    env = dict(os.environ, PRINT_ONLY="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run([BASH, _msys(SCRIPT), "41234", "h", "48512345",
                        "exitTPT", "6", "/dev/null", "--bc-target", "dist",
                        "--bc-value-coef", "0.25"],
                       capture_output=True, text=True, env=env,
                       cwd=str(ROOT), check=True)
    remote = r.stdout
    # bash itself must be able to parse it
    chk = subprocess.run([BASH, "-n", "-c", remote], capture_output=True,
                         text=True)
    assert chk.returncode == 0, chk.stderr
    # --train-extra is argparse.REMAINDER: everything after it is the
    # trainer's, so nothing of expert_loop's own may follow it
    i = remote.index("--train-extra")
    tail = remote[i:remote.index(" > runs/", i)]
    assert tail.split() == ["--train-extra", "--bc-target", "dist",
                            "--bc-value-coef", "0.25"], tail
    # the pid the watchdogs poll is the DRIVER's, written by the box
    assert "echo $! > runs/exitTPT.pid" in remote
    assert "/root/box_watchdog.sh 48512345 exitTPT " in remote
    # the last thing the box does is show the watchdog's own first line -
    # tolerantly, because a missing log must not fail the launch, and loudly,
    # because a box with no self-destruct is the one thing worse
    assert "tail -1 /root/box_watchdog.log 2>/dev/null" in remote
    assert "no self-destruct" in remote


@needs_bash
def test_a_fake_box_receives_the_seed_and_the_flags_verbatim(tmp_path):
    box, ck = _fake_box(tmp_path)
    binm, main, key = _stubs(tmp_path, box)
    flags = ["--bc-target", "dist", "--bc-value-coef", "0.25",
             "--demo-note", "a b"]          # a flag with a SPACE in it
    r = _run(tmp_path, box, ck, binm, main, key, flags)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]

    # 1. the seed arrived and was md5-verified ON THE BOX
    seed = box / "RL_Surf" / "runs_seed.pt"
    assert seed.exists()
    assert hashlib.md5(seed.read_bytes()).hexdigest() == \
        hashlib.md5(ck.read_bytes()).hexdigest()
    assert "md5 OK on the box" in r.stdout

    # 2. + 3. the flags reached argv verbatim, space included
    import json
    argv = json.loads((box / "RL_Surf" / "runs" / "argv.json")
                      .read_text(encoding="utf-8"))
    assert argv[1] == str(box / "RL_Surf" / "runs_seed.pt").replace("\\", "/")
    j = argv.index("--train-extra")
    assert argv[j + 1:] == flags, argv[j:]
    assert "--name" in argv and argv[argv.index("--name") + 1] == "exitTPT"
    assert argv[argv.index("--rounds") + 1] == "2"
    assert argv[argv.index("--train-steps") + 1] == "3e8"
    assert argv[argv.index("--plan-budget") + 1] == "600"

    # 4. the pid file and the watchdog's arguments
    pid = int((box / "RL_Surf" / "runs" / "exitTPT.pid")
              .read_text(encoding="utf-8").strip())
    assert pid > 0
    assert "driver alive pid" in r.stdout
    wd = (box / "watchdog_args.txt").read_text(encoding="utf-8").split()
    assert wd[0] == "48512345" and wd[1] == "exitTPT" and wd[3] == "40"
    deadline = int(wd[2])

    # 5. the harvest spec: two register calls, the second with the expert
    # loop's shape, at a deadline UNDER the on-box one
    reg = [l for l in (tmp_path / "register.log").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    assert len(reg) == 2, reg
    assert "--harvest" not in reg[0]          # deadline first, before launch
    assert "--harvest-only-extra" in reg[1]
    assert "--pid-file runs/exitTPT.pid" in reg[1]
    assert "runs/exitTPT/expert_summary.jsonl" in reg[1]
    assert "runs/exitTPT_driver.txt" in reg[1]
    assert "runs/exitTPT/round_*/train/ckpt_final.pt" in reg[1]
    mins = int(reg[1].split("--minutes")[1].split()[0].strip('"'))
    import time
    assert mins * 60 <= deadline - time.time() + 60, (mins, deadline)

    for p in (box / "RL_Surf" / "runs" / "exitTPT.pid",):
        try:
            os.kill(pid, 9)
        except OSError:
            pass


@needs_bash
def test_a_truncated_checkpoint_is_refused_and_nothing_is_launched(tmp_path):
    """CLAUDE.md: an scp that truncates a checkpoint still exits 0, and only
    the md5 on the box catches it. Two tries, then refuse - never launch an
    arm on a seed that did not arrive."""
    box, ck = _fake_box(tmp_path)
    binm, main, key = _stubs(tmp_path, box, corrupt=True)
    r = _run(tmp_path, box, ck, binm, main, key,
             ["--bc-target", "argmax", "--bc-value-coef", "0"])
    assert r.returncode == 1, r.stdout[-2000:]
    assert "md5 MISMATCH" in r.stdout
    assert "did not arrive intact" in (r.stdout + r.stderr)
    assert not (box / "RL_Surf" / "runs" / "argv.json").exists()
    assert not (box / "RL_Surf" / "runs" / "exitTPT.pid").exists()
    # and the box is still REGISTERED with a deadline: a rented box is never
    # left out of the registry because a transfer failed
    assert not (tmp_path / "register.log").exists() or \
        "--harvest" not in (tmp_path / "register.log").read_text(
            encoding="utf-8")


@needs_bash
def test_no_trainer_flags_is_refused():
    """The two arms differ ONLY in the trainer flags, so an empty tail is
    always a mistake - and silently launching two identical arms would look
    exactly like a null result."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([BASH, _msys(SCRIPT), "1", "h", "1", "x", "1",
                        "/dev/null"], capture_output=True, text=True,
                       env=env, cwd=str(ROOT))
    assert r.returncode == 2
    assert "differ ONLY in them" in r.stderr
