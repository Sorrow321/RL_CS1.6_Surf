"""rampfast.py - the ramp-window task's geometry compiled (numba): the target search and the whole
window line of surfgym.goalramps, step for step the Python path's arithmetic.

Why: at the training shape a 2,048-env iteration (1M steps) drew ~3,150 windows - ~1,800 for new
episodes, ~700 at captures, ~650 at shifts - at ~1 ms each in Python, 3.5 s of a 9.2 s iteration
(the 2026-09-28 timing probe; the user: "FPS is below 100k while normally ... around 350k").

What is ported, from goalramps.py:
  * nearest_pair / candidates - RampWindows' search (brute-force minima over the same subsampled
    contact origins the per-target KD trees hold);
  * ride_face - RampWindows._ride_face;
  * next_target - RampWindows._next_fn: the closest eligible target (or the finish box) to the arc
    launched off the previous ride, beyond the previous piece and within the reach cap;
  * window - window_line for one or two targets, the second fixed or chosen on the way, with the
    riding-first ride, the Hermite arrivals, the rides, the lookahead and the resampling.
The goal field is sampled with goalfield's own float32 trilinear arithmetic (one point at a time,
single-threaded: the fused sampler's prange dispatch cost ~30 us per single-point call).

Equivalence is asserted against the Python path (tests/python/test_goalramps.py; on utopia by the
1,500-spawn replay in the session's verify script): the same targets, the same lines to float
noise (BLAS dot products and libm pow may round the last bit differently).

Ids: kernels work in target INDICES j (0..F-1, RampWindows._ids order) and compact piece indices;
FIN = -2 and NONE = -1 as in goalramps; CHOOSE = -3 asks window() to choose the second target.
"""
from __future__ import annotations

import os

import numpy as np

FIN = -2
NONE = -1
CHOOSE = -3


try:
    from numba import njit as _njit
except Exception:
    _njit = None

if _njit is not None and os.environ.get("SURFGYM_NO_NUMBA") != "1":

    @_njit(cache=True, nogil=True)
    def nearest_pair(path, O, s, c):
        """(distance, path index, origin index - s) of the closest (path point, O[s:s+c]) pair:
        per path point its nearest origin, then the first path point with the least"""
        best = np.inf
        bj = 0
        bi = s
        for j in range(path.shape[0]):
            px = path[j, 0]
            py = path[j, 1]
            pz = path[j, 2]
            dj = np.inf
            ij = s
            for i in range(s, s + c):
                dx = O[i, 0] - px
                dy = O[i, 1] - py
                dz = O[i, 2] - pz
                d2 = dx * dx + dy * dy + dz * dz
                if d2 < dj:
                    dj = d2
                    ij = i
            if dj < best:
                best = dj
                bj = j
                bi = ij
        return np.sqrt(best), bj, bi - s

    @_njit(cache=True, nogil=True)
    def candidates(path, cen, rad, O, ostart, ocount, pj, n_p, dsurf, use_d, d_min, d_max,
                   excl, need):
        """best-first over bounding-sphere bounds with exact distances, stopping once no later
        target can beat the need-th best PIECE -> (per piece its best distance, inf = none; per
        target its exact distance, -1 = not computed)"""
        F = cen.shape[0]
        lb = np.empty(F)
        for f in range(F):
            m = np.inf
            for t in range(path.shape[0]):
                dx = path[t, 0] - cen[f, 0]
                dy = path[t, 1] - cen[f, 1]
                dz = path[t, 2] - cen[f, 2]
                d = np.sqrt(dx * dx + dy * dy + dz * dz)
                if d < m:
                    m = d
            lb[f] = max(0.0, m - rad[f])
        order = np.argsort(lb)
        pbest = np.full(n_p, np.inf)
        fdist = np.full(F, -1.0)
        kth = np.inf
        nfound = 0
        for jj in range(F):
            f = order[jj]
            if lb[f] > kth:
                break
            p = pj[f]
            if excl[p]:
                continue
            if use_d:
                d = dsurf[f]
                if not (d_min <= d and d < d_max):
                    continue
            dm, _j, _i = nearest_pair(path, O, ostart[f], ocount[f])
            fdist[f] = dm
            if dm < pbest[p]:
                if pbest[p] == np.inf:
                    nfound += 1
                pbest[p] = dm
            if nfound >= need:
                kth = np.sort(pbest)[need - 1]
        return pbest, fdist

    @_njit(cache=True, nogil=True)
    def ride_face(p, pf_start, pf_count, pf_j, path, axx, axy, O, ostart, ocount, fdist):
        """RampWindows._ride_face for compact piece p: its surface with the longest run along
        the axis (axx, axy) less the arc's distance to it (first maximum) -> index j"""
        best = -1
        sc = -np.inf
        for t in range(pf_start[p], pf_start[p] + pf_count[p]):
            j = pf_j[t]
            ds = fdist[j]
            if ds < 0.0:
                ds, _a, _b = nearest_pair(path, O, ostart[j], ocount[j])
            lo = np.inf
            hi = -np.inf
            for i in range(ostart[j], ostart[j] + ocount[j]):
                pr = O[i, 0] * axx + O[i, 1] * axy
                if pr < lo:
                    lo = pr
                if pr > hi:
                    hi = pr
            x = (hi - lo) - ds
            if x > sc:
                best = j
                sc = x
        return best

    @_njit(cache=True, nogil=True)
    def gf_sample1(x, y, z, grid, mins, cell, valid_max, sentinel):
        """goalfield's fused trilinear sampler for ONE point - its float32 arithmetic, its order"""
        nz, ny, nx = grid.shape
        gx = (x - mins[0]) / cell - 0.5
        gy = (y - mins[1]) / cell - 0.5
        gz = (z - mins[2]) / cell - 0.5
        ix0 = int(np.floor(gx))
        iy0 = int(np.floor(gy))
        iz0 = int(np.floor(gz))
        fx = np.float32(gx - ix0)
        fy = np.float32(gy - iy0)
        fz = np.float32(gz - iz0)
        num = np.float32(0.0)
        den = np.float32(0.0)
        for dz in range(2):
            wz = fz if dz == 1 else np.float32(1.0) - fz
            kz = iz0 + dz
            kz = 0 if kz < 0 else (nz - 1 if kz > nz - 1 else kz)
            for dy in range(2):
                wy = fy if dy == 1 else np.float32(1.0) - fy
                ky = iy0 + dy
                ky = 0 if ky < 0 else (ny - 1 if ky > ny - 1 else ky)
                for dx in range(2):
                    wx = fx if dx == 1 else np.float32(1.0) - fx
                    kx = ix0 + dx
                    kx = 0 if kx < 0 else (nx - 1 if kx > nx - 1 else kx)
                    v = grid[kz, ky, kx]
                    honest = np.float32(1.0) if v < valid_max else np.float32(0.0)
                    w = ((wx * wy) * wz) * honest
                    num += w * v
                    den += w
        if den > np.float32(1e-6):
            dd = den if den > np.float32(1e-6) else np.float32(1e-6)
            return num / dd
        return sentinel

    @_njit(cache=True, nogil=True)
    def norm3(x, y, z):
        return np.sqrt(x * x + y * y + z * z)

    @_njit(cache=True, nogil=True)
    def next_target(cpx, cpy, cpz, cvx, cvy, cvz, q, excl, cen, rad, O, ostart, ocount, pj,
                    n_p, dsurf, use_d, plow, pf_start, pf_count, pf_j, pax, pay, phas, need,
                    grid, mins, cell, valid_max, sentinel, reach_max, fin_lo, fin_hi, gravity,
                    horizon, progress_delta, speed_margin):
        """RampWindows._next_fn: the target closest to the arc from (cp, cv) - beyond compact
        piece q (-1 = none) and the launch point, within the reach cap - or the finish box when
        the reach cap reaches it and it is at least as close -> index j or FIN (FIN also when
        nothing is eligible)"""
        d_max = np.inf
        d_min = -np.inf
        if use_d:
            dq = np.inf
            if q >= 0:
                dq = plow[q]
            d_at = np.float64(gf_sample1(cpx, cpy, cpz, grid, mins, cell, valid_max, sentinel))
            if not (d_at < reach_max):
                d_at = np.inf
            d_ref = dq if dq < d_at else d_at
            if np.isfinite(d_ref):
                d_max = d_ref - progress_delta
                sp = norm3(cvx, cvy, cvz)
                d_min = d_ref - (sp + gravity * horizon + speed_margin) * horizon
        ex = excl.copy()
        if q >= 0:
            ex[q] = True
        ts = np.arange(0.0, horizon, 0.1)
        path = np.empty((ts.shape[0], 3))
        for t in range(ts.shape[0]):
            tt = ts[t]
            path[t, 0] = cpx + cvx * tt + 0.5 * 0.0 * tt ** 2
            path[t, 1] = cpy + cvy * tt + 0.5 * 0.0 * tt ** 2
            path[t, 2] = cpz + cvz * tt + 0.5 * (-gravity) * tt ** 2
        pbest, fdist = candidates(path, cen, rad, O, ostart, ocount, pj, n_p, dsurf, use_d,
                                  d_min, d_max, ex, need)
        bp = -1
        bd = np.inf
        for p in range(n_p):
            if pbest[p] < bd:
                bd = pbest[p]
                bp = p
        # the finish box: the arc's closest approach to it (0 inside)
        fd = np.inf
        for t in range(path.shape[0]):
            qx = min(max(path[t, 0], fin_lo[0]), fin_hi[0])
            qy = min(max(path[t, 1], fin_lo[1]), fin_hi[1])
            qz = min(max(path[t, 2], fin_lo[2]), fin_hi[2])
            d = norm3(path[t, 0] - qx, path[t, 1] - qy, path[t, 2] - qz)
            if d < fd:
                fd = d
        fin_ok = d_min <= 0.0
        if bp < 0 or (fin_ok and fd <= bd):
            return FIN
        if phas[bp]:
            axx = pax[bp]
            axy = pay[bp]
        else:
            sh = norm3(cvx, cvy, 0.0)
            sh = sh if sh > 1e-9 else 1e-9
            axx = cvx / sh
            axy = cvy / sh
        return ride_face(bp, pf_start, pf_count, pf_j, path, axx, axy, O, ostart, ocount, fdist)

    @_njit(cache=True, nogil=True)
    def ride_dir(nbx, nby, nbz, tx, ty, tz):
        """goalramps.ride_dir: the level line (cross(nb, z)), or `toward` projected into the
        plane on a floor, unit, signed along `toward`"""
        lx = nby * 1.0 - nbz * 0.0
        ly = nbz * 0.0 - nbx * 1.0
        lz = nbx * 0.0 - nby * 0.0
        if norm3(lx, ly, lz) < 1e-3:
            d = tx * nbx + ty * nby + tz * nbz
            lx = tx - d * nbx
            ly = ty - d * nby
            lz = tz - d * nbz
        n = norm3(lx, ly, lz)
        n = n if n > 1e-9 else 1e-9
        lx = lx / n
        ly = ly / n
        lz = lz / n
        if lx * tx + ly * ty + lz * tz >= 0.0:
            return lx, ly, lz
        return -lx, -ly, -lz

    @_njit(cache=True, nogil=True)
    def resample(buf, n, spacing):
        """route.resample_polyline: drop repeated points, then constant-arc-length samples"""
        keep = np.empty(n, np.int64)
        m = 0
        for i in range(n):
            if i == 0:
                keep[m] = 0
                m += 1
                continue
            dx = buf[i, 0] - buf[i - 1, 0]
            dy = buf[i, 1] - buf[i - 1, 1]
            dz = buf[i, 2] - buf[i - 1, 2]
            if norm3(dx, dy, dz) > 1e-6:
                keep[m] = i
                m += 1
        if m < 2:
            raise ValueError("route polyline has fewer than 2 distinct points")
        s = np.zeros(m)
        for i in range(1, m):
            a = keep[i - 1]
            b = keep[i]
            s[i] = s[i - 1] + norm3(buf[b, 0] - buf[a, 0], buf[b, 1] - buf[a, 1],
                                    buf[b, 2] - buf[a, 2])
        total = s[m - 1]
        cnt = max(2, int(round(total / spacing)) + 1)
        q = np.linspace(0.0, total, cnt)
        out = np.empty((cnt, 3), np.float32)
        for k in range(3):
            col = np.empty(m)
            for i in range(m):
                col[i] = buf[keep[i], k]
            out[:, k] = np.interp(q, s, col).astype(np.float32)
        return out

    @_njit(cache=True, nogil=True)
    def window(ox, oy, oz, vx, vy, vz, k0, k1, riding_first, excl, cen, rad, O, NRM, ostart,
               ocount, pj, n_p, dsurf, use_d, plow, pf_start, pf_count, pf_j, pax, pay, phas,
               need, grid, mins,
               cell, valid_max, sentinel, reach_max, fin_lo, fin_hi, finish, gravity, horizon,
               progress_delta, speed_margin, dt_path, ramp_coast, ramp_press, ride_past,
               ray_floor, ray_spacing, line_cap, buf):
        """window_line for ks = [k0] (k1 NONE), [k0, k1] (k1 an index or FIN) or [k0, chosen]
        (k1 CHOOSE: next_target off k0's ride, beyond k0's piece and the pieces in excl) ->
        (the resampled line float32 capped at line_cap, the second target: index, FIN, or NONE)"""
        n = 0
        buf[0, 0] = ox
        buf[0, 1] = oy
        buf[0, 2] = oz
        n = 1
        cpx, cpy, cpz = ox, oy, oz
        cvx, cvy, cvz = vx, vy, vz
        nsteps = 1 if k1 == NONE else 2
        prev = -1
        t2 = NONE
        broke = False
        for step in range(nsteps):
            if step == 0:
                k = k0
            elif k1 == CHOOSE:
                qprev = pj[prev] if prev >= 0 else -1
                k = next_target(cpx, cpy, cpz, cvx, cvy, cvz, qprev, excl, cen, rad, O, ostart,
                                ocount, pj, n_p, dsurf, use_d, plow, pf_start, pf_count, pf_j,
                                pax, pay, phas, need, grid, mins, cell, valid_max, sentinel,
                                reach_max, fin_lo, fin_hi, gravity, horizon, progress_delta,
                                speed_margin)
            else:
                k = k1
            if step == 1:
                t2 = k
            prev = k if k >= 0 else -1
            spd = norm3(cvx, cvy, cvz)
            spd = spd if spd > ray_floor else ray_floor
            if step == 0 and riding_first and k != FIN:
                pt = np.empty((1, 3))
                pt[0, 0] = cpx
                pt[0, 1] = cpy
                pt[0, 2] = cpz
                _d, _j, io = nearest_pair(pt, O, ostart[k], ocount[k])
                nb0 = NRM[ostart[k] + io, 0]
                nb1 = NRM[ostart[k] + io, 1]
                nb2 = NRM[ostart[k] + io, 2]
                cn = norm3(cvx, cvy, cvz)
                cn = cn if cn > 1e-9 else 1e-9
                lx, ly, lz = ride_dir(nb0, nb1, nb2, cvx / cn, cvy / cn, cvz / cn)
                ext = -np.inf
                for i in range(ostart[k], ostart[k] + ocount[k]):
                    e = (O[i, 0] - cpx) * lx + (O[i, 1] - cpy) * ly + (O[i, 2] - cpz) * lz
                    if e > ext:
                        ext = e
                st = spd * 0.01
                m = max(2, int(max(ext, 0.0) / st))
                if n + m + 1 > buf.shape[0]:
                    raise ValueError("rampfast.window: the raw line buffer is full")
                for s in range(1, m + 1):
                    ss = s * spd * 0.01
                    buf[n, 0] = cpx + lx * ss
                    buf[n, 1] = cpy + ly * ss
                    buf[n, 2] = cpz + lz * ss
                    n += 1
                cpx, cpy, cpz = buf[n - 1, 0], buf[n - 1, 1], buf[n - 1, 2]
                cvx, cvy, cvz = lx * spd, ly * spd, lz * spd
                continue
            ts = np.arange(0.0, ramp_coast, dt_path)
            nt = ts.shape[0]
            path = np.empty((nt, 3))
            for t in range(nt):
                tt = ts[t]
                path[t, 0] = cpx + cvx * tt + 0.5 * 0.0 * tt ** 2
                path[t, 1] = cpy + cvy * tt + 0.5 * 0.0 * tt ** 2
                path[t, 2] = cpz + cvz * tt + 0.5 * (-gravity) * tt ** 2
            if k == FIN:
                jj = 0
                bd = np.inf
                for t in range(nt):
                    d = norm3(path[t, 0] - finish[0], path[t, 1] - finish[1],
                              path[t, 2] - finish[2])
                    if d < bd:
                        bd = d
                        jj = t
                pbx, pby, pbz = finish[0], finish[1], finish[2]
                has_nb = False
                nb0 = nb1 = nb2 = 0.0
            else:
                _d, jj, io = nearest_pair(path, O, ostart[k], ocount[k])
                pbx = O[ostart[k] + io, 0]
                pby = O[ostart[k] + io, 1]
                pbz = O[ostart[k] + io, 2]
                nb0 = NRM[ostart[k] + io, 0]
                nb1 = NRM[ostart[k] + io, 1]
                nb2 = NRM[ostart[k] + io, 2]
                has_nb = True
            tc = max(0.2, jj * dt_path, norm3(pbx - cpx, pby - cpy, pbz - cpz) / spd)
            jv = jj if jj < nt - 1 else nt - 1
            tv = ts[jv]
            vcx = cvx + 0.0 * tv
            vcy = cvy + 0.0 * tv
            vcz = cvz + (-gravity) * tv
            sp2 = norm3(vcx, vcy, vcz)
            sp2 = sp2 if sp2 > ray_floor else ray_floor
            if has_nb:
                dvn = vcx * nb0 + vcy * nb1 + vcz * nb2
                ux = vcx - dvn * nb0
                uy = vcy - dvn * nb1
                uz = vcz - dvn * nb2
                if norm3(ux, uy, uz) < 1e-3:
                    ax = pbx - cpx
                    ay = pby - cpy
                    az = pbz - cpz
                    dan = ax * nb0 + ay * nb1 + az * nb2
                    ux = ax - dan * nb0
                    uy = ay - dan * nb1
                    uz = az - dan * nb2
                ex_ = pbx - nb0 * ramp_press
                ey_ = pby - nb1 * ramp_press
                ez_ = pbz - nb2 * ramp_press
            else:
                ux = pbx - cpx
                uy = pby - cpy
                uz = pbz - cpz
                ex_, ey_, ez_ = pbx, pby, pbz
            un_ = norm3(ux, uy, uz)
            un_ = un_ if un_ > 1e-6 else 1e-6
            unx, uny, unz = ux / un_, uy / un_, uz / un_
            if norm3(cvx, cvy, cvz) >= 1.0:
                vvx, vvy, vvz = cvx, cvy, cvz
            else:
                vvx, vvy, vvz = unx * ray_floor, uny * ray_floor, unz * ray_floor
            ns = max(2, int(np.ceil(tc / 0.01))) + 1
            if n + ns + 1 > buf.shape[0]:
                raise ValueError("rampfast.window: the raw line buffer is full")
            ssv = np.linspace(0.0, 1.0, ns)
            for i in range(1, ns):
                s_ = ssv[i]
                s3 = s_ ** 3.0
                s2 = s_ * s_
                h00 = 2 * s3 - 3 * s2 + 1
                h10 = s3 - 2 * s2 + s_
                h01 = -2 * s3 + 3 * s2
                h11 = s3 - s2
                buf[n, 0] = (h00 * cpx + h10 * (vvx * tc)) + h01 * ex_ + h11 * (unx * sp2 * tc)
                buf[n, 1] = (h00 * cpy + h10 * (vvy * tc)) + h01 * ey_ + h11 * (uny * sp2 * tc)
                buf[n, 2] = (h00 * cpz + h10 * (vvz * tc)) + h01 * ez_ + h11 * (unz * sp2 * tc)
                n += 1
            if not has_nb:
                cpx, cpy, cpz = ex_, ey_, ez_
                cvx, cvy, cvz = unx * sp2, uny * sp2, unz * sp2
                broke = True
                break
            lx, ly, lz = ride_dir(nb0, nb1, nb2, unx, uny, unz)
            ext = -np.inf
            for i in range(ostart[k], ostart[k] + ocount[k]):
                e = (O[i, 0] - ex_) * lx + (O[i, 1] - ey_) * ly + (O[i, 2] - ez_) * lz
                if e > ext:
                    ext = e
            st = sp2 * 0.01
            m = max(2, int(max(ext, 0.0) / st))
            if n + m + 9 > buf.shape[0]:
                raise ValueError("rampfast.window: the raw line buffer is full")
            for s in range(1, m + 1):
                rr = s * sp2 * 0.01
                buf[n, 0] = ex_ + lx * rr
                buf[n, 1] = ey_ + ly * rr
                buf[n, 2] = ez_ + lz * rr
                n += 1
            cpx, cpy, cpz = buf[n - 1, 0], buf[n - 1, 1], buf[n - 1, 2]
            cvx, cvy, cvz = lx * sp2, ly * sp2, lz * sp2
        if not broke:
            cn = norm3(cvx, cvy, cvz)
            cn = cn if cn > 1e-9 else 1e-9
            dx, dy, dz = cvx / cn, cvy / cn, cvz / cn
            la = np.linspace(ride_past / 8, ride_past, 8)
            for i in range(8):
                buf[n, 0] = cpx + dx * la[i]
                buf[n, 1] = cpy + dy * la[i]
                buf[n, 2] = cpz + dz * la[i]
                n += 1
        line = resample(buf, n, ray_spacing)
        if line.shape[0] > line_cap:
            line = line[:line_cap].copy()
        return line, t2


    FAST = (nearest_pair, candidates, ride_face, next_target, window, gf_sample1)
else:
    FAST = None

