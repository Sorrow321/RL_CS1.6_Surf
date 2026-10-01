"""--obs-no-depth (the user, 2026-10-02): the policy does not see the depth channel - one camera,
the face normal and the potential only. The renderer is untouched (the march needs the depth to
find the hit; the recorder and the POV draw it); the conv reads channels 1.. of the image.

(a) the policy: conv.0 has in_ch - 1 inputs, the depth channel cannot move the features, the
    other channels do;
(b) a trainer smoke (CPU): conv.0 (16, 4, 5, 5) beside --obs-normal + --obs-potential, the config
    key, record_ckpt mirrors it, the refusals (no other channel; a mismatched resume).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_obs_potential import ABS, CANNONBALL, RECORD, _run, _train  # noqa: E402
from test_view_continuous import needs_run                           # noqa: E402

FACEID = CANNONBALL.parent / "surf_src_cannonball.faceid_32.npz"


def test_the_conv_never_reads_the_depth():
    from train_fast import N_SCALAR, Policy
    W, H, C = 16, 8, 5
    torch.manual_seed(0)
    pol = Policy(N_SCALAR + W * H * C, W, H, emb=32, hidden=32, in_ch=C, drop_depth=True)
    assert tuple(pol.conv[0].weight.shape) == (16, C - 1, 5, 5)
    scal = torch.zeros(3, N_SCALAR)
    img = torch.rand(3, H, W, C)
    f0 = pol.features(scal, img.reshape(3, -1))
    other = img.clone()
    other[..., 0] = torch.rand(3, H, W)              # a different depth image
    assert torch.equal(pol.features(scal, other.reshape(3, -1)), f0)
    other = img.clone()
    other[..., 1] += 1.0                             # a different potential channel
    assert not torch.equal(pol.features(scal, other.reshape(3, -1)), f0)
    with pytest.raises(SystemExit):
        Policy(N_SCALAR + W * H, W, H, emb=32, hidden=32, in_ch=1, drop_depth=True)


@needs_run
@pytest.mark.skipif(not FACEID.exists(), reason="needs the baked cannonball face grid (cell 32)")
def test_no_depth_trainer_smoke():
    run = "cya_nodepth"
    r = _train(run, ABS + ["--obs-potential", "norm", "--obs-normal", "1", "--obs-no-depth", "1"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "--obs-no-depth" in r.stdout and "conv in_ch 4" in r.stdout
    d = ROOT / "runs" / run
    assert json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]["obs_no_depth"] == 1
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert tuple(ck["policy"]["conv.0.weight"].shape) == (16, 4, 5, 5)
    rec = _run([sys.executable, "-u", str(RECORD), str(d / "ckpt_final.pt"),
                "--map", str(CANNONBALL), "--episodes", "1", "--out", str(d / "rec.jsonl")])
    assert rec.returncode == 0, rec.stdout[-4000:] + rec.stderr[-4000:]
    bad = _train("cya_nodepth_bad", ABS + ["--obs-no-depth", "0"], ckpt=d / "ckpt_final.pt")
    assert bad.returncode != 0 and "--obs-no-depth changes" in bad.stdout + bad.stderr
    alone = _train("cya_nodepth_alone", ABS + ["--obs-no-depth", "1"])
    assert alone.returncode != 0 and "needs another channel" in alone.stdout + alone.stderr
    for n in (run, "cya_nodepth_bad", "cya_nodepth_alone"):
        shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)
