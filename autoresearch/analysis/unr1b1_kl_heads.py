"""analysis only (unr1b1, 2026-11-09): which policy head carries PPO's approx_kl in unr1b1_v2?
Records one greedy episode of a checkpoint (tools/record_ckpt.py, CPU), keeps every observation
the policy saw, then moves the weights by k steps of the checkpoint's OWN stored Adam direction
(-lr * m_hat / (sqrt(v_hat) + eps), the moments the trainer saved) and measures, on those
observations, the KL of every head the PPO ratio multiplies (under --view-continuous: the four
key heads fwd/side/jump/duck + the yaw and pitch Gaussians), and the joint log-ratio of actions
sampled from the old policy (approx_kl = mean(logp_old - logp_new); clip share = |ratio - 1| > 0.2).
A proxy for one PPO update (4 epochs x 16 minibatches = 64 optimizer steps), not the update itself.

usage: kl_heads.py <ckpt> <out traj.jsonl>
"""
import copy
import sys
import time

import numpy as np
import torch

W = "/home/steam/ar_uni/wt/b1"
sys.path.insert(0, W + "/python")
sys.path.insert(0, W + "/tools")
import train_fast  # noqa: E402
from train_fast import Policy  # noqa: E402
import record_ckpt  # noqa: E402

CK, OUT = sys.argv[1], sys.argv[2]
MAP = W + "/maps/surf_src_sidistic.bsp"
CAP, SELF = [], []
_orig = Policy.forward


def _fwd(self, obs, h=None, priv=None):
    CAP.append(obs.detach().float().cpu().clone())
    if not SELF:
        SELF.append(self)
    return _orig(self, obs, h, priv)


Policy.forward = _fwd
t0 = time.time()
record_ckpt.main([CK, "--map", MAP, "--episodes", "1", "--out", OUT])
Policy.forward = _orig
pol = SELF[0].cpu().float().eval()
X = torch.cat(CAP)
print(f"\n{CK.split('/')[-1]}: captured {tuple(X.shape)} observations in {time.time() - t0:.0f}s")

ck = torch.load(CK, map_location="cpu", weights_only=False)
opt = ck["optimizer"]
pg = opt["param_groups"][0]
st = opt["state"]
params = list(pol.parameters())
ids = pg["params"]
assert len(params) == len(ids), (len(params), len(ids))
for p, i in zip(params, ids):
    assert tuple(st[i]["exp_avg"].shape) == tuple(p.shape), (tuple(p.shape), tuple(st[i]["exp_avg"].shape))
b1, b2 = pg["betas"]
eps, lr = pg["eps"], pg["lr"]
D = []
for p, i in zip(params, ids):
    s = st[i]
    t = float(s["step"])
    m = s["exp_avg"].float() / (1 - b1 ** t)
    v = s["exp_avg_sq"].float() / (1 - b2 ** t)
    D.append(-lr * m / (v.sqrt() + eps))
print(f"adam: lr {lr:g} betas {b1:g},{b2:g} eps {eps:g} step {float(st[ids[0]]['step']):.0f}; "
      f"{len(params)} tensors")

NVEC = (15, 7, 4, 4, 2, 3)
NV = int(train_fast.N_VIEW)
offs = np.concatenate([[0], np.cumsum(NVEC)])
A = int(offs[-1])
NAMES = ["fwd", "side", "jump", "duck"]


@torch.no_grad()
def dists(model):
    out, _ = _orig(model, X)
    out = out.float()
    assert out.shape[1] == A + 2, out.shape
    cats = [torch.log_softmax(out[:, offs[j]:offs[j + 1]], dim=1) for j in range(NV, len(NVEC))]
    mu = out[:, A:A + 2]
    ls = model.log_std().float().reshape(-1)[:2]
    return cats, mu, ls


c0, mu0, ls0 = dists(pol)
g = torch.Generator().manual_seed(1109)
S = 16                                   # sampled actions per observation
samp_c = [torch.multinomial(lp.exp(), S, replacement=True, generator=g) for lp in c0]
z = mu0[:, None, :] + ls0.exp()[None, None, :] * torch.randn(X.shape[0], S, 2, generator=g)


def glogp(zz, mu, ls):
    return -0.5 * ((zz - mu[:, None, :]) / ls.exp()) ** 2 - ls - 0.5 * np.log(2 * np.pi)


print(f"sigma yaw {ls0[0].exp():.4f} pitch {ls0[1].exp():.4f}; greedy key-head entropy "
      + " ".join(f"{n} {-(lp.exp() * lp).sum(1).mean():.3f}" for n, lp in zip(NAMES, c0)))
for k in (1, 8, 64):
    m1 = copy.deepcopy(pol)
    with torch.no_grad():
        for p, d in zip(m1.parameters(), D):
            p.add_(k * d)
    c1, mu1, ls1 = dists(m1)
    kl = {}
    lr_joint = torch.zeros(X.shape[0], S)
    for n, a, b, s in zip(NAMES, c0, c1, samp_c):
        kl[n] = float((a.exp() * (a - b)).sum(1).mean())
        lr_joint += b.gather(1, s) - a.gather(1, s)
    for j, n in enumerate(("yaw", "pitch")):
        s0, s1 = ls0[j].exp(), ls1[j].exp()
        kl[n] = float((torch.log(s1 / s0) + (s0 ** 2 + (mu0[:, j] - mu1[:, j]) ** 2) / (2 * s1 ** 2) - 0.5).mean())
        lr_joint += (glogp(z[:, :, j:j + 1], mu1[:, j:j + 1], ls1[j:j + 1])
                     - glogp(z[:, :, j:j + 1], mu0[:, j:j + 1], ls0[j:j + 1]))[:, :, 0]
    tot = sum(kl.values())
    akl = float((-lr_joint).mean())
    clip = float(((lr_joint.exp() - 1).abs() > 0.2).float().mean())
    dmu = (mu1 - mu0).abs().mean(0)
    print(f"k {k:2d} steps: sum of head KLs {tot:9.4f} | approx_kl (sampled) {akl:9.4f} | clip share {clip:.3f} | "
          + " ".join(f"{n} {v:.4f} ({100 * v / max(tot, 1e-12):.0f}%)" for n, v in kl.items())
          + f" | mean |d mu| yaw {dmu[0]:.4f} pitch {dmu[1]:.4f}")
    sys.stdout.flush()
