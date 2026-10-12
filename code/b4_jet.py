"""Fiberwise RM at a vGS point: jets of a curve in Phi_P0 cap slice tangent to e_RM, with the degenerate (rank-1)
second-order structure handled exactly.
Slice: F0 + C + E3, C a complement of ker J (J|C injective), E3 = <e, c2, c3> (complement of U in ker J). If Y (smooth,
15-dim, T Y = U + <e>) exists, Y cap slice is a smooth curve tangent to e, a graph over the e-coordinate:
  F(s) = F0 + s e + sum_{j>=2} s^j (y_j + lam_j l + mu_j d),  y_j in C,
where pi(2B(e, d)) = 0, pi(2B(e, l)) = v* != 0 (rank 1 on <c2, c3>; pi = projection to coker J).
Order k: r_k = [s^k] G; Im J part fixes y_k; pi(r_k) = 0 is, in the unknowns (mu_{k-2}, lam_{k-1}),
  A + mu B + mu^2 C + lam D  (mu^2 only for k = 4; D = v*),
fitted from 4 evaluations (+1 check) and solved exactly (common roots of the projected quadrics / linear system).
mu_{k-2} also changes y_{k-1} (through pi-invisible 2B(e, d) in r_{k-1}); recomputed for every trial value.
No root => no such curve => no smooth Y with tangent U + <e_RM> (given T Y = U + <e_RM>, i.e. P(F0) = P0).
Usage: python -I b4_jet.py OUT.json prime seed RMTAN.json N [dir=eRM|c2]
"""
import json
import os
import random
import sys
import time

import flint
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b2_humbert as B
from b3_arc import RN

T0 = time.time()


def log(*a):
    print('[%7.1fs]' % (time.time() - T0), *a, flush=True)


def main():
    out, p, seed, rmfile, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
    which = sys.argv[6] if len(sys.argv) > 6 else 'eRM'
    rng = random.Random(1000 + seed)
    pcoef = [rng.randrange(p) for _ in range(7)]
    acoef = [rng.randrange(p) for _ in range(3)]
    ctx3 = flint.nmod_mpoly_ctx.get(('x0', 'x1', 'x2'), modulus=p)
    x0, x1, x2 = ctx3.gens()
    pbin = sum(pcoef[k] * x0 ** (6 - k) * x1 ** k for k in range(7))
    abin = sum(acoef[k] * x0 ** (2 - k) * x1 ** k for k in range(3))
    F0 = pbin + x1 * (x2 ** 5 - 5 * abin * x2 ** 3 + 5 * abin ** 2 * x2)
    S6p = B.mons(3, 6)
    S6i = {m: k for k, m in enumerate(S6p)}

    def vec28(poly):
        v = np.zeros(28, dtype=np.int64)
        for e_, cf in poly.to_dict().items():
            v[S6i[tuple(int(t) for t in e_)]] = int(cf)
        return v

    F0v = vec28(F0)
    wr = random.Random(5000 + seed)
    IC = B.fit_ic(p, wr)
    W, H5 = B.h5_fit(p, wr)
    rk = lambda M: B.rk(M, p)
    nullsp = lambda M: B.nullsp(M, p)
    inv = lambda a: pow(int(a) % p, p - 2, p)

    def G(ws, n):
        Rg = RN(p, n)
        fc = [Rg.zero() for _ in range(7)]
        for m, k in S6i.items():
            e0, e1, e2 = m
            mono = flint.nmod_poly([0] * e0 + [1], p)
            comp = [int(F0v[k]) % p] + [int(ws[j][k]) % p if j in ws else 0 for j in range(1, n)]
            if any(comp):
                fc[e0 + e1] = Rg.add(fc[e0 + e1], [mono * c if c else Rg.z for c in comp])
        Is = [B.eval_poly_R(I, fc, Rg, 7) for I in IC]
        return B.eval_poly_R(H5, Is, Rg, 4)

    cols = []
    for k in range(28):
        v = np.zeros(28, dtype=np.int64); v[k] = 1
        cols.append(G({1: v}, 2)[1])
    vr = np.array([wr.randrange(p) for _ in range(28)], dtype=np.int64)
    # t-degree bound: f_i has t-degree <= i, I_{2k} is isobaric of weight 3 * degree, H5 has coefficient degree 2W
    L = 3 * 2 * W + 1

    def coeffs(poly):
        c = [int(x) for x in poly.coeffs()]
        assert len(c) <= L
        return np.array(c + [0] * (L - len(c)), dtype=np.int64)

    Jm = np.array([coeffs(c) for c in cols], dtype=np.int64).T % p
    rJ = rk(Jm)
    Wb = nullsp(Jm)
    Lk = nullsp(Jm.T)
    pi = lambda vec: (Lk @ (vec % p)) % p
    # complement C of ker J: coordinate vectors
    Cidx = []
    for k in range(28):
        if rk(Jm[:, Cidx + [k]]) == len(Cidx) + 1:
            Cidx.append(k)
    assert len(Cidx) == rJ
    JC = Jm[:, Cidx]
    rows = []
    for i in range(L):
        if rk(JC[rows + [i]]) == len(rows) + 1:
            rows.append(i)
        if len(rows) == rJ:
            break
    JCi = np.array([int(x) for x in flint.nmod_mat(rJ, rJ, [int(x) for x in JC[rows].flatten()], p).inv().entries()],
                   dtype=np.int64).reshape(rJ, rJ)

    def ysolve(r):
        """y in C with J y = -r (r in Im J); checked"""
        assert not pi(r).any()
        yc = (JCi @ ((-r[rows]) % p)) % p
        y = np.zeros(28, dtype=np.int64)
        y[Cidx] = yc
        assert not ((Jm @ y + r) % p).any()
        return y

    gens3 = [x0, x1, x2]
    gl3 = np.array([vec28(gens3[k] * F0.derivative(i)) for i in range(3) for k in range(3)], dtype=np.int64)
    stab = np.array([vec28(gens3[k] * F0.derivative(i)) for i in range(3) for k in range(3)
                     if not (k == 2 and i in (0, 1))], dtype=np.int64)
    vgs = [vec28(x0 ** (6 - k) * x1 ** k) for k in range(7)]
    dD5 = -5 * x2 ** 3 + 10 * abin * x2
    vgs += [vec28(x1 * x0 ** (2 - k) * x1 ** k * dD5) for k in range(3)]
    U = np.vstack([np.array(vgs, dtype=np.int64), stab]) % p
    rU = rk(U)
    rm = json.load(open(rmfile))
    RM = np.array([[d.get(",".join(map(str, m)), 0) for m in S6p] for d in rm["basis"]], dtype=np.int64) % p
    RG = np.vstack([RM, gl3]) % p
    Nn = nullsp(np.vstack([Wb, RG]).T)
    Eb = (Nn[:, :len(Wb)] @ Wb) % p
    eRM = next(row % p for row in Eb if rk(np.vstack([U, row[None, :]])) > rU)
    comp = [eRM]
    for row in Wb:
        if rk(np.vstack([U] + [c[None, :] for c in comp] + [row[None, :]])) > rU + len(comp):
            comp.append(row % p)
        if len(comp) == 3:
            break
    e = comp[0] if which == 'eRM' else comp[1]
    others = [comp[1], comp[2]] if which == 'eRM' else [comp[0], comp[2]]

    def Q2(v):
        return coeffs(G({1: v}, 3)[2])

    def Bf(v, w):
        return ((Q2((v + w) % p) - Q2(v) - Q2(w)) * inv(2)) % p

    pb1, pb2 = pi(2 * Bf(e, others[0])), pi(2 * Bf(e, others[1]))
    res = {"prime": p, "seed": seed, "N": N, "dir": which, "rank_J": rJ, "dim_kerJ": len(Wb), "dim_U": rU,
           "rank_pi2B_e_on_E3": rk(np.vstack([pb1, pb2])), "pi_Q_e_zero": not pi(Q2(e)).any()}
    assert res["pi_Q_e_zero"], 'e not in the second-order cone'
    if res["rank_pi2B_e_on_E3"] != 1:
        log('rank of pi 2B(e, .) on <c2, c3> is', res["rank_pi2B_e_on_E3"], '(scheme assumes 1)')
    if pb1.any():
        ell = others[0]
        kap = next((int(pb2[i]) * inv(pb1[i])) % p for i in range(len(pb1)) if pb1[i])
        d = (others[1] - kap * others[0]) % p
    else:
        ell, d = others[1], others[0]
    vstar = pi(2 * Bf(e, ell))
    assert vstar.any() and not pi(2 * Bf(e, d)).any()
    # projector killing vstar on the coker coordinates
    Pperp = nullsp(vstar[None, :])                              # rows z with z . vstar = 0
    log('setup done: rank J %d, ker J %d, U %d; coker %d' % (rJ, len(Wb), rU, len(Lk)))

    i0 = next(i for i in range(len(vstar)) if vstar[i])
    ws0 = {1: e % p}
    r2 = coeffs(G(ws0, 3)[2])
    results = []

    def step(k, ws, base, lam, trail):
        """state: ws[1..k-3] final, ws[k-2] = base[k-2] + lam[k-2] l (+ mu_{k-2} d, unknown), base[k-1] provisional"""
        if k > N:
            wf = dict(ws)
            wf[N] = base[N]
            fin = G(wf, N + 1)
            ok = all(fin[j].degree() < 0 for j in range(N + 1))
            nz = [j for j in sorted(wf) if (wf[j] % p).any()]
            info = dict(status="lifted to order %d" % N, final_check=ok, trail=trail, nonzero_orders=nz,
                        w={str(j): [int(x) for x in wf[j] % p] for j in nz}, S6_monomials=[list(m) for m in S6p])
            if nz and max(nz) <= 2:
                # exact polynomial arc? evaluate G on F0 + s w1 + s^2 w2 to high order
                M = 3 * N
                Gh = G({1: wf[1], 2: wf[2]}, M)
                info["quadratic_arc_G_zero_to_order"] = next((j for j in range(M) if Gh[j].degree() >= 0), M) - 1
            results.append(info)
            log('branch lifted to order %d (final check %s): %s; nonzero orders %s' % (N, ok, trail, nz))
            return

        def eval_rk(m_, l_):
            w = dict(ws)
            if k >= 4:
                w[k - 2] = (base[k - 2] + lam[k - 2] * ell + m_ * d) % p
                yk1 = ysolve(coeffs(G(w, k)[k - 1]))
            else:
                yk1 = base[2]
            w[k - 1] = (yk1 + l_ * ell) % p
            return coeffs(G(w, k + 1)[k]), w, yk1

        pts = [(0, 0), (1, 0), (2, 0), (0, 1)]
        vals = [pi(eval_rk(m_, l_)[0]) for m_, l_ in pts]
        A = vals[0]
        Dl = (vals[3] - A) % p
        Cq = ((vals[2] - 2 * vals[1] + A) * inv(2)) % p
        Bq = (vals[1] - A - Cq) % p
        chk = pi(eval_rk(3, 1)[0])
        assert not ((A + 3 * Bq + 9 * Cq + Dl - chk) % p).any(), 'fit failed at order %d' % k
        assert not ((Dl - vstar) % p).any(), 'lam coefficient != v*'
        a_, b_, c_ = (Pperp @ A) % p, (Pperp @ Bq) % p, (Pperp @ Cq) % p
        g = None
        for i in range(len(a_)):
            P = flint.nmod_poly([int(a_[i]), int(b_[i]), int(c_[i])], p)
            if P.degree() < 0:
                continue
            g = P if g is None else g.gcd(P)
        if k == 3:
            assert not Bq.any() and not Cq.any()
            if g is not None:
                results.append(dict(status="obstructed at order 3", trail=trail))
                log('branch obstructed at order 3:', trail)
                return
            musol = [0]
        elif g is None:
            musol = ['free']
        elif g.degree() == 0:
            results.append(dict(status="obstructed at order %d (no mu_%d)" % (k, k - 2), trail=trail))
            log('branch obstructed at order %d:' % k, trail)
            return
        else:
            fac = g.factor()[1]
            musol = [int(-f[0]) * inv(f[1]) % p for f, m in fac if f.degree() == 1]
            nonlin = [f.degree() for f, m in fac if f.degree() > 1]
            if nonlin:
                results.append(dict(status="order %d: mu_%d has roots only over extensions (degrees %s)"
                                    % (k, k - 2, nonlin), trail=trail))
                log('branch: order %d has non-F_p roots, degrees %s' % (k, nonlin), trail)
            if not musol:
                return
        for mval in musol:
            note = 'mu_%d=%s' % (k - 2, mval) if k >= 4 else 'order3'
            if mval == 'free':
                mval = wr.randrange(p)
                note = 'mu_%d free (random %d)' % (k - 2, mval)
            resid = (A + mval * Bq + mval * mval * Cq) % p
            lval = (-int(resid[i0]) * inv(vstar[i0])) % p
            rfin, w, yk1 = eval_rk(mval, lval)
            assert not pi(rfin).any()
            ws2, base2, lam2 = dict(ws), dict(base), dict(lam)
            if k >= 4:
                ws2[k - 2] = (base[k - 2] + lam[k - 2] * ell + mval * d) % p
                base2[k - 1] = yk1
            lam2[k - 1] = lval
            ws2[k - 1] = (base2[k - 1] + lval * ell) % p
            base2[k] = ysolve(rfin)
            log('order %d: %s, lam_%d = %d' % (k, note, k - 1, lval))
            step(k + 1, ws2, base2, lam2, trail + [note])

    step(3, ws0, {2: ysolve(r2)}, {}, [])
    res["branches"] = results
    res["any_lifted"] = any(r["status"].startswith("lifted") for r in results)
    res["seconds"] = round(time.time() - T0, 1)
    log(json.dumps(res))
    with open(out, "w", newline="\n") as fh:
        fh.write(json.dumps(res, indent=1) + "\n")



if __name__ == "__main__":
    main()
