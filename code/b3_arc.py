"""Fiberwise RM at a vGS point: formal arcs in Phi_P0 = {G(F) == 0} (b2_humbert.py) along a chosen first-order direction.
F(e) = F0 + sum_k e^k w_k.  Order k >= 2: residual r_k = [e^k] G(F0 + sum_{j<k} e^j w_j); for k >= 3 adjust w_{k-1} by
kappa in ker J (changes order k by 2 B(w_1, kappa)) so that r_k in Im J (possible iff pi(r_k) lies in pi(2B(w_1, ker J)),
pi = projection to coker J); then w_k = particular solution of J w_k = -r_k plus a random ker J part. The arc exists to
order N iff every step is solvable (necessary condition for w_1 to be tangent to a component of Phi_P0).
Directions: rm (e_RM + random u in U), c2 (the second second-order zero of b2_humbert), rand (random in ker J).
Also: transversality rank of pi(2B(e_RM, ker J)) (2 = codim of U + <e_RM> in ker J), multiplicity of the two
second-order zeros, and membership of e_RM, c2 in the lattice-keeping / hexaflex-keeping tangents (b1_hodge.py files).
Usage: python -I b3_arc.py OUT.json prime seed RMTAN.json N DIRECTION
"""
import json
import random
import sys
import time

import os

import flint
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b2_humbert as B

T0 = time.time()


def log(*a):
    print('[%7.1fs]' % (time.time() - T0), *a, flush=True)


class RN:
    def __init__(self, p, n):
        self.p, self.n = p, n
        self.z = flint.nmod_poly([], p)

    def mul(self, a, b):
        out = []
        for k in range(self.n):
            s = self.z
            for i in range(k + 1):
                if a[i].degree() >= 0 and b[k - i].degree() >= 0:
                    s = s + a[i] * b[k - i]
            out.append(s)
        return out

    def add(self, a, b):
        return [a[i] + b[i] for i in range(self.n)]

    def scal(self, a, s):
        return [a[i] * s for i in range(self.n)]

    def one(self):
        return [flint.nmod_poly([1], self.p)] + [self.z] * (self.n - 1)

    def zero(self):
        return [self.z] * self.n


def main():
    out, p, seed, rmfile, N, direction = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], \
        int(sys.argv[5]), sys.argv[6]
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
        for e, cf in poly.to_dict().items():
            v[S6i[tuple(int(t) for t in e)]] = int(cf)
        return v

    F0v = vec28(F0)
    wr = random.Random(5000 + seed)
    IC = B.fit_ic(p, wr)
    W, H5 = B.h5_fit(p, wr)
    log('IC and H5 fitted (H5 weighted degree %d)' % W)
    rk = lambda M: B.rk(M, p)
    nullsp = lambda M: B.nullsp(M, p)

    def G(ws, n):
        """G(F0 + sum_{k=1}^{n-1} e^k ws[k]) in F_p[t][e]/(e^n)"""
        Rg = RN(p, n)
        fc = [Rg.zero() for _ in range(7)]
        for m, k in S6i.items():
            e0, e1, e2 = m
            i = e0 + e1
            mono = flint.nmod_poly([0] * e0 + [1], p)
            comp = [int(F0v[k]) % p] + [int(ws[j][k]) % p if j in ws else 0 for j in range(1, n)]
            if any(comp):
                fc[i] = Rg.add(fc[i], [mono * c if c else Rg.z for c in comp])
        Is = [B.eval_poly_R(I, fc, Rg, 7) for I in IC]
        return B.eval_poly_R(H5, Is, Rg, 4)

    # first order
    cols = []
    for k in range(28):
        v = np.zeros(28, dtype=np.int64); v[k] = 1
        cols.append(G({1: v}, 2)[1])
    vr = np.array([wr.randrange(p) for _ in range(28)], dtype=np.int64)
    L = max([c.degree() for c in cols] + [c.degree() for c in G({1: vr}, 3)]) + 1

    def coeffs(poly):
        c = [int(x) for x in poly.coeffs()]
        assert len(c) <= L, 't-degree bound exceeded'
        return np.array(c + [0] * (L - len(c)), dtype=np.int64)
    Jm = np.array([coeffs(c) for c in cols], dtype=np.int64).T % p
    rJ = rk(Jm)
    Wb = nullsp(Jm)
    Lk = nullsp(Jm.T)
    pi = lambda vec: (Lk @ (vec % p)) % p
    gens3 = [x0, x1, x2]
    gl3 = np.array([vec28(gens3[k] * F0.derivative(i)) for i in range(3) for k in range(3)], dtype=np.int64)
    stab = np.array([vec28(gens3[k] * F0.derivative(i)) for i in range(3) for k in range(3)
                     if not (k == 2 and i in (0, 1))], dtype=np.int64)
    vgs = [vec28(x0 ** (6 - k) * x1 ** k) for k in range(7)]
    dD5 = -5 * x2 ** 3 + 10 * abin * x2
    vgs += [vec28(x1 * x0 ** (2 - k) * x1 ** k * dD5) for k in range(3)]
    U = np.vstack([np.array(vgs, dtype=np.int64), stab]) % p
    rU = rk(U)
    Ub = nullsp(nullsp(U))
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
    res = {"prime": p, "seed": seed, "N": N, "direction": direction, "rank_J": rJ, "dim_kerJ": len(Wb), "dim_U": rU}

    def Q2(v):
        return coeffs(G({1: v}, 3)[2])

    def Bf(v, w):
        return ((Q2((v + w) % p) - Q2(v) - Q2(w)) * pow(2, p - 2, p)) % p

    # transversality at e_RM and c2; lattice / hexaflex membership
    for name, vec in (("eRM", eRM), ("c2", comp[1])):
        Bm = np.array([pi(2 * Bf(vec, k_)) for k_ in Wb], dtype=np.int64).T % p      # coker x 17
        res["rank_pi2B_" + name] = rk(Bm)
    for tag in ("pim1", "hex"):
        try:
            d = json.load(open(rmfile.replace(".json", "_" + tag + ".json")))
        except OSError:
            continue
        T = np.array([[b.get(",".join(map(str, m)), 0) for m in S6p] for b in d["basis"]], dtype=np.int64) % p
        TG = np.vstack([T, gl3, U]) % p
        r0 = rk(TG)
        for name, vec in (("eRM", eRM), ("c2", comp[1]), ("c3", comp[2])):
            res["%s_in_%s+gl3+U" % (name, tag)] = rk(np.vstack([TG, vec[None, :]])) == r0
    log(json.dumps({k: v for k, v in res.items() if k.startswith(('rank_pi', 'eRM_', 'c2_', 'c3_'))}))

    # the arc
    if direction == 'rm':
        w1 = eRM.copy()
        for row in Ub:
            w1 = (w1 + wr.randrange(p) * row) % p
    elif direction == 'c2':
        w1 = comp[1].copy()
        for row in Ub:
            w1 = (w1 + wr.randrange(p) * row) % p
    elif direction == 'u':                                      # validation: tangent to the vGS locus (arcs exist)
        w1 = np.zeros(28, dtype=np.int64)
        for row in Ub:
            w1 = (w1 + wr.randrange(p) * row) % p
    elif direction == 'rm0':                                    # e_RM with u = 0
        w1 = eRM.copy()
    else:
        w1 = np.zeros(28, dtype=np.int64)
        for row in Wb:
            w1 = (w1 + wr.randrange(p) * row) % p
    ws = {1: w1}
    B2 = [(2 * Bf(w1, k_)) % p for k_ in Wb]                      # full vectors, 17
    piB2 = np.array([pi(b) for b in B2], dtype=np.int64).T % p
    res["rank_pi2B_w1"] = rk(piB2)
    status = "lifted"
    for k in range(2, N + 1):
        r = coeffs(G(ws, k + 1)[k])
        if k >= 3:
            target = (-pi(r)) % p
            if target.any():
                c = B.rref_solve(piB2, target, p) if rk(np.hstack([piB2, target[:, None]])) == rk(piB2) else None
                if c is None:
                    # rref_solve needs full column rank; use a particular solution via nullspace of [piB2 | -target]
                    Aug = np.hstack([piB2, ((-target) % p)[:, None]])
                    if rk(Aug) > rk(piB2):
                        status = "obstructed at order %d" % k
                        log(status)
                        break
                    Nz = nullsp(Aug)
                    sol = next(v for v in Nz if v[-1] % p)
                    c = (sol[:-1] * pow(int(sol[-1]), p - 2, p)) % p
                kap = np.zeros(28, dtype=np.int64)
                for ci, k_ in zip(c, Wb):
                    kap = (kap + int(ci) * k_) % p
                ws[k - 1] = (ws[k - 1] + kap) % p
                r = coeffs(G(ws, k + 1)[k])
        elif pi(r).any():
            status = "obstructed at order 2"
            log(status)
            break
        assert not pi(r).any()
        # particular solution of J w = -r  (Aug [w; 1] = 0 <=> J w + r = 0)
        Aug = np.hstack([Jm, (r % p)[:, None]])
        Nz = nullsp(Aug)
        sol = next(v for v in Nz if v[-1] % p)
        wk = (sol[:-1] * pow(int(sol[-1]), p - 2, p)) % p
        # canonical particular solution: remove its ker J part outside U? (keep it; the next order's adjustment can
        # change w_k by any element of ker J) plus a random U part
        for row in Ub:
            wk = (wk + wr.randrange(p) * row) % p
        ws[k] = wk
        log('order %d solved' % k)
    if status == "lifted":
        fin = G(ws, N + 1)
        res["final_check_zero_to_N"] = all(fin[k].degree() < 0 for k in range(N + 1))
        log('final check, G == 0 mod e^%d: %s' % (N + 1, res["final_check_zero_to_N"]))
    res["status"] = status
    res["seconds"] = round(time.time() - T0, 1)
    log(json.dumps(res))
    with open(out, "w", newline="\n") as fh:
        fh.write(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
