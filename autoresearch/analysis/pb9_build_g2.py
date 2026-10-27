"""Join two Go-Explore gains into one curriculum spine (pbr9b2, 2026-10-27).

    python autoresearch/analysis/pb9_build_g2.py LINE.npy LINE.keys.npy ARCHIVE.npz CHAIN.npy K \\
        BASE.npy BASE.keys.npy OUT.npy [--cut BASE_ROW]

LINE = the policy's own greedy line the chain is rooted on (record_ckpt --dump-states), CHAIN =
ge_replay's chain from ARCHIVE (rooted at a LINE state), K = the chain tick to hand over at, BASE =
a spine whose first rows are LINE's (pb9_g1 = LINE rows 0-7479 + GE A + the policy's beam).

OUT = LINE up to the chain's root + the chain's first K ticks (tools/stitch_spine.py, keys from the
archive) + BASE from the first row at which LINE's running-best progress reaches the chain's
progress at tick K (a SEAM at matched progress: every row is a real state, so a backward
curriculum spawns from either side; the rows the chain saved drop out of the spine's clock, which
--demo-success-time reads). --cut writes OUT's cut at BASE's row BASE_ROW as <OUT stem>e.npy.
Nothing comes from a demo: the line is the policy's, the chains the run's own reward-free search."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))
CFG = json.loads((REPO / "autoresearch" / "config.json").read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    for k in ("line", "line_keys", "archive", "chain"):
        ap.add_argument(k)
    ap.add_argument("k", type=int)
    ap.add_argument("base"); ap.add_argument("base_keys"); ap.add_argument("out")
    ap.add_argument("--cut", type=int, default=None)
    a = ap.parse_args()
    from surfgym.core import STATE_DTYPE
    from surfgym.goalfield import GoalField
    import stitch_spine
    z = np.load(REPO / CFG["field"], allow_pickle=False)
    f = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"], float(z["cell"]),
                  float(z["reach_max"]))
    d0 = float(CFG["start_d"])

    def prog(o):
        d = f.sample(np.asarray(o, np.float64).reshape(-1, 3)).astype(np.float64)
        return 100 * (d0 - np.where(d < f.reach_max - 16, d, 1e9)) / d0

    L = np.load(a.line); LK = np.load(a.line_keys)
    C = np.load(a.chain)
    r = int(np.argmin(np.linalg.norm(L["origin"].astype(np.float64)
                                     - C["origin"][0].astype(np.float64), axis=1)))
    with tempfile.TemporaryDirectory() as td:
        lp, kp = Path(td) / "line.npy", Path(td) / "line.keys.npy"
        np.save(lp, L[:r + 1]); np.save(kp, LK[:r + 1])
        p1 = Path(td) / "p1.npy"
        rc = stitch_spine.main(["--out", str(p1), "--field", str(REPO / CFG["field"]), "--no-cut",
                                f"spine:{lp}:{kp}", f"ge:{a.archive}:{a.chain}:{a.k}"])
        assert rc == 0
        P1 = np.load(p1); P1K = np.load(p1.with_suffix(".keys.npy"))
    B = np.load(a.base); BK = np.load(a.base_keys)
    pk = float(prog(P1["origin"][-1:])[0])
    bp = np.maximum.accumulate(prog(B["origin"]))
    j = int(np.flatnonzero(bp >= pk)[0])
    out = np.concatenate([P1, B[j:]]).astype(STATE_DTYPE)
    keys = np.concatenate([P1K, BK[j:]]).astype(np.int32)
    out["tick"] = 0
    out["stuck_ticks"] = 0
    np.save(a.out, out)
    np.save(Path(a.out).with_suffix(".keys.npy"), keys)
    seam = float(np.linalg.norm(P1["origin"][-1].astype(np.float64)
                                - B["origin"][j].astype(np.float64)))
    info = {"line": a.line, "chain": a.chain, "archive": a.archive, "root_row": r, "k": a.k,
            "handover_progress": pk, "base": a.base, "base_join_row": j, "seam_u": seam,
            "n": int(len(out)), "rows_saved": int((r + a.k) - j)}
    print(json.dumps(info))
    Path(a.out).with_suffix(".json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    if a.cut is not None:
        c = len(P1) + (int(a.cut) - j)
        e = Path(a.out).with_name(Path(a.out).stem + "e.npy")
        np.save(e, out[:c + 1]); np.save(e.with_suffix(".keys.npy"), keys[:c + 1])
        print(f"cut at OUT row {c} (BASE row {a.cut}) -> {e}")


if __name__ == "__main__":
    main()
