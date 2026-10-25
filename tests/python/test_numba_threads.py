"""surfgym's NUMBA_NUM_THREADS default (pbr4b3, 2026-10-25).

numba sizes its pool to every visible core and ignores a container's CFS quota; on a
64-thread host under a 30.7-CPU quota the goal field's prange team throttled the trainer
~35x. ``import surfgym`` now defaults the pool to OMP_NUM_THREADS (or half the usable CPUs),
before goalfield imports numba."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def _numba_threads_after_import(env_extra):
    env = {k: v for k, v in os.environ.items()
           if k not in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS")}
    env.update(env_extra)
    env["PYTHONPATH"] = str(ROOT / "python")
    out = subprocess.run(
        [sys.executable, "-c",
         "import surfgym, numba, os; "
         "print(numba.config.NUMBA_NUM_THREADS, os.environ['NUMBA_NUM_THREADS'])"],
        env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    a, b = out.stdout.split()
    return int(a), b


def test_default_follows_omp_threads():
    from surfgym import default_numba_threads
    assert default_numba_threads({"OMP_NUM_THREADS": "15"}) == "15"
    assert default_numba_threads({"OMP_NUM_THREADS": " 7 "}) == "7"


def test_default_without_omp_is_half_the_usable_cpus():
    from surfgym import _usable_cpus, default_numba_threads
    want = str(max(4, min(32, _usable_cpus() // 2)))
    assert default_numba_threads({}) == want
    assert default_numba_threads({"OMP_NUM_THREADS": "0"}) == want
    assert default_numba_threads({"OMP_NUM_THREADS": "x"}) == want


def test_import_sets_the_pool_before_numba_reads_it():
    n, env = _numba_threads_after_import({"OMP_NUM_THREADS": "3"})
    assert (n, env) == (3, "3")


def test_an_explicit_numba_setting_is_kept():
    n, env = _numba_threads_after_import({"OMP_NUM_THREADS": "3", "NUMBA_NUM_THREADS": "2"})
    assert (n, env) == (2, "2")


def test_goal_field_sample_is_thread_count_independent():
    """The fused sampler is element-wise: 1 thread and the default give the same bits."""
    import numpy as np
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_NUM_THREADS"}
    env["PYTHONPATH"] = str(ROOT / "python")
    code = (
        "import numpy as np, sys\n"
        "from surfgym.goalfield import GoalField\n"
        "rng = np.random.default_rng(0)\n"
        "g = rng.uniform(0, 5000, (20, 24, 28)).astype(np.float32)\n"
        "g[3, 4, 5] = 1e9\n"
        "f = GoalField(g, np.array([-100.0, -200.0, 50.0]), 32.0, 6000.0)\n"
        "p = rng.uniform([-150, -250, 0], [900, 600, 700], (5000, 3))\n"
        "sys.stdout.buffer.write(f.sample(p).astype(np.float32).tobytes())\n")
    outs = []
    for nt in ("1", "4"):
        e = dict(env, NUMBA_NUM_THREADS=nt)
        r = subprocess.run([sys.executable, "-c", code], env=e, capture_output=True,
                           timeout=300)
        assert r.returncode == 0, r.stderr.decode(errors="replace")
        outs.append(np.frombuffer(r.stdout, np.float32))
    assert len(outs[0]) == 5000
    assert np.array_equal(outs[0], outs[1])
