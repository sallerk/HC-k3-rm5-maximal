"""Fiberwise RM at a vGS point, second order, via the Humbert surface H_5.
Phi_P0 = {F in S_6 : every fibre w^2 = f_t(x1, x2) = F(t x1, x1, x2) of the pencil through P0 = (0:0:1) lies on H_5},
cut out by G(F)(t) = H5(IC(f_t)) == 0 in F_p[t].
- IC = Igusa-Clebsch invariants (I2, I4, I6, I10 = disc), defined from the roots (Mestre's sums over 15, 10, 60 terms);
  their coefficient polynomials are fitted here on split sextics (isobaric monomials, weight 3d).
- H5: Elkies-Kumar 1209.3527, Cor. 15: (I2:I4:I6:I10) = (6(4g+1), 9g^2, 9(4h+9g^3+2g^2), 4h^2);
  the weighted-homogeneous implicit equation is interpolated here (smallest weighted degree with a kernel).
- Checks: y^2 = x^5 + 1 has I2 = I4 = I6 = 0; SL2/GL2 invariance of the ratios; TTV curves y^2 = D5(x; a) + c on H5,
  random sextics not; G(F0) == 0 at the vGS point.
- First order: J = d/de G(F0 + e v) (v in S_6, 28 coords); checks dim ker J = 17 and that the RM tangent (from b1_hodge.py) lies in
  ker J + gl_3 F0.
- Second order: Q(v) = [e^2] G(F0 + e v). U = V + S (vGS + stabiliser of P0, tangent to a sublocus of Phi_P0, so
  Q(U) in Im J: checked); e_RM = the direction of ((RM tangent + gl_3 F0) cap ker J) mod U. A smooth 15-dim Y in Phi_P0
  with T Y = U + <e_RM> needs Q(e_RM) in Im J and B(u, e_RM) in Im J for all u in U.
Usage: python -I b2_humbert.py OUT.json prime seed RMTAN.json
"""
import itertools
import json
import random
import sys

import flint
import numpy as np


def mons(nv, n):
    return [e for e in itertools.product(range(n + 1), repeat=nv) if sum(e) == n][::-1]


def rref_solve(A, b, p):
    """solve A c = b (overdetermined, consistent) mod p; returns c or None"""
    m, n = A.shape
    M = flint.nmod_mat(m, n + 1, [int(x) for x in np.hstack([A % p, (b % p)[:, None]]).flatten()], p)
    R, r = M.rref()
    R = np.array([int(x) for x in R.entries()], dtype=np.int64).reshape(m, n + 1)[:r]
    piv = [int(np.nonzero(R[i])[0][0]) for i in range(r)]
    if n in piv:
        return None
    if r < n:
        return None
    c = np.zeros(n, dtype=np.int64)
    for i, j in enumerate(piv):
        c[j] = R[i, n]
    return c


def rk(M, p):
    M = np.asarray(M, dtype=np.int64) % p
    if M.size == 0:
        return 0
    return flint.nmod_mat(M.shape[0], M.shape[1], [int(x) for x in M.flatten()], p).rank()


def nullsp(M, p):
    M = np.asarray(M, dtype=np.int64) % p
    X, nl = flint.nmod_mat(M.shape[0], M.shape[1], [int(x) for x in M.flatten()], p).nullspace()
    X = np.array([int(x) for x in X.entries()], dtype=np.int64).reshape(M.shape[1], M.shape[1])
    return X[:, :nl].T                      # rows = basis


# ---------------------------------------------------------------- Igusa-Clebsch from roots
PAIRINGS = []
for a in itertools.permutations(range(6)):
    pr = tuple(sorted(tuple(sorted(a[2 * i:2 * i + 2])) for i in range(3)))
    if pr not in PAIRINGS:
        PAIRINGS.append(pr)
SPLITS = []
for T1 in itertools.combinations(range(6), 3):
    T2 = tuple(i for i in range(6) if i not in T1)
    if 0 in T1:
        SPLITS.append((T1, T2))
assert len(PAIRINGS) == 15 and len(SPLITS) == 10


def ic_roots(f6, r, p):
    d2 = {}
    for i in range(6):
        for j in range(6):
            d2[(i, j)] = (r[i] - r[j]) ** 2 % p
    I2 = sum(d2[a] * d2[b] * d2[c] for a, b, c in PAIRINGS) % p
    I4, I6 = 0, 0
    for T1, T2 in SPLITS:
        tri = d2[(T1[0], T1[1])] * d2[(T1[1], T1[2])] * d2[(T1[2], T1[0])] % p * \
            d2[(T2[0], T2[1])] * d2[(T2[1], T2[2])] * d2[(T2[2], T2[0])] % p
        I4 += tri
        for sg in itertools.permutations(T2):
            I6 += tri * d2[(T1[0], sg[0])] * d2[(T1[1], sg[1])] * d2[(T1[2], sg[2])]
    I10 = 1
    for i in range(6):
        for j in range(i + 1, 6):
            I10 = I10 * d2[(i, j)] % p
    return [I2 * f6 ** 2 % p, I4 * f6 ** 4 % p, I6 * f6 ** 6 % p, I10 * pow(f6, 10, p) % p]


def coeffs_from_roots(f6, r, p):
    poly = [1]                               # ascending coefficients of prod (x - r_i)
    for a in r:
        new = [0] * (len(poly) + 1)
        for k, c in enumerate(poly):
            new[k + 1] = (new[k + 1] + c) % p
            new[k] = (new[k] - a * c) % p
        poly = new
    return [c * f6 % p for c in poly]        # f_0 .. f_6


def compositions(n, k):
    if k == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for rest in compositions(n - i, k - 1):
            yield (i,) + rest


def fit_ic(p, rng):
    out = []
    for d in (2, 4, 6, 10):
        ms = [e for e in compositions(d, 7) if sum(k * e[k] for k in range(7)) == 3 * d]
        rows, vals = [], []
        while len(rows) < len(ms) + 40:
            r = rng.sample(range(p), 6)
            f6 = rng.randrange(1, p)
            f = coeffs_from_roots(f6, r, p)
            rows.append([int(np.prod([pow(f[k], e[k], p) for k in range(7)], dtype=object) % p) for e in ms])
            vals.append(ic_roots(f6, r, p)[(2, 4, 6, 10).index(d)])
        c = rref_solve(np.array(rows, dtype=np.int64), np.array(vals, dtype=np.int64), p)
        assert c is not None, 'IC fit failed for degree %d' % d
        out.append([(e, int(v)) for e, v in zip(ms, c) if v])
    return out


def ic_eval(IC, f, p):
    return [sum(v * int(np.prod([pow(f[k], e[k], p) for k in range(7)], dtype=object)) for e, v in I) % p for I in IC]


# ---------------------------------------------------------------- H5 implicit equation
def h5_fit(p, rng, maxW=60):
    for W in range(1, maxW + 1):
        ms = [(a, b, c, d) for d in range(W // 5 + 1) for c in range(W // 3 + 1) for b in range(W // 2 + 1)
              for a in range(W + 1) if a + 2 * b + 3 * c + 5 * d == W]
        if not ms:
            continue
        rows = []
        for _ in range(len(ms) + 30):
            g, h, lam = rng.randrange(p), rng.randrange(p), rng.randrange(1, p)
            I = [lam * 6 * (4 * g + 1) % p, lam ** 2 * 9 * g * g % p,
                 lam ** 3 * 9 * (4 * h + 9 * g ** 3 + 2 * g * g) % p, lam ** 5 * 4 * h * h % p]
            rows.append([pow(I[0], a, p) * pow(I[1], b, p) * pow(I[2], c, p) * pow(I[3], d, p) % p
                         for a, b, c, d in ms])
        K = nullsp(np.array(rows, dtype=np.int64), p)
        if len(K):
            assert len(K) == 1, 'H5 kernel dim %d at W = %d' % (len(K), W)
            return W, [(m, int(v)) for m, v in zip(ms, K[0]) if v]
    raise RuntimeError('no H5 equation up to W = %d' % maxW)


def h5_eval(H5, I, p):
    return sum(v * pow(I[0], a, p) * pow(I[1], b, p) * pow(I[2], c, p) * pow(I[3], d, p) for (a, b, c, d), v in H5) % p


# ---------------------------------------------------------------- truncated ring F_p[t][e]/(e^3)
class R3:
    def __init__(self, p):
        self.p = p
        self.z = flint.nmod_poly([], p)

    def mul(self, a, b):
        return [a[0] * b[0], a[0] * b[1] + a[1] * b[0], a[0] * b[2] + a[1] * b[1] + a[2] * b[0]]

    def add(self, a, b):
        return [a[i] + b[i] for i in range(3)]

    def scal(self, a, s):
        return [a[i] * s for i in range(3)]

    def one(self):
        return [flint.nmod_poly([1], self.p), self.z, self.z]

    def zero(self):
        return [self.z, self.z, self.z]


def eval_poly_R(terms, xs, Rg, nvar):
    """sum v * prod xs[k]^e[k] in the ring; caches powers"""
    pw = [[Rg.one()] for _ in range(nvar)]
    acc = Rg.zero()
    for e, v in terms:
        m = Rg.one()
        for k in range(nvar):
            while len(pw[k]) <= e[k]:
                pw[k].append(Rg.mul(pw[k][-1], xs[k]))
            if e[k]:
                m = Rg.mul(m, pw[k][e[k]])
        acc = Rg.add(acc, Rg.scal(m, v))
    return acc


def main():
    out, p, seed, rmfile = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    rng = random.Random(1000 + seed)
    pcoef = [rng.randrange(p) for _ in range(7)]
    acoef = [rng.randrange(p) for _ in range(3)]
    ctx3 = flint.nmod_mpoly_ctx.get(('x0', 'x1', 'x2'), modulus=p)
    x0, x1, x2 = ctx3.gens()
    pbin = sum(pcoef[k] * x0 ** (6 - k) * x1 ** k for k in range(7))
    abin = sum(acoef[k] * x0 ** (2 - k) * x1 ** k for k in range(3))
    F0 = pbin + x1 * (x2 ** 5 - 5 * abin * x2 ** 3 + 5 * abin ** 2 * x2)
    S6p = mons(3, 6)
    S6i = {m: k for k, m in enumerate(S6p)}

    def vec28(poly):
        v = np.zeros(28, dtype=np.int64)
        for e, cf in poly.to_dict().items():
            v[S6i[tuple(int(t) for t in e)]] = int(cf)
        return v

    F0v = vec28(F0)
    res = {"prime": p, "seed": seed}
    wr = random.Random(5000 + seed)
    IC = fit_ic(p, wr)
    res["ic_terms"] = [len(I) for I in IC]
    # checks on the invariants
    f = [1, 0, 0, 0, 0, 1, 0]                     # x^5 + 1 (f_0 .. f_6)
    res["x5p1_I2_I4_I6"] = ic_eval(IC, f, p)[:3]
    ok = True
    for _ in range(5):
        f = [wr.randrange(p) for _ in range(7)]
        a, b, c, d = [wr.randrange(p) for _ in range(4)]
        ctx2 = flint.nmod_mpoly_ctx.get(('X', 'Z'), modulus=p)
        X, Z = ctx2.gens()
        fp = sum(f[k] * X ** k * Z ** (6 - k) for k in range(7))
        g = fp.compose(a * X + b * Z, c * X + d * Z)
        gd = g.to_dict()
        fg = [int(gd.get((k, 6 - k), 0)) for k in range(7)]
        I, J_ = ic_eval(IC, f, p), ic_eval(IC, fg, p)
        det = (a * d - b * c) % p
        # I_{2k}(f o M) = det^{6k} I_{2k}(f)  (weight 3 * degree)
        ok &= all(J_[i] == I[i] * pow(det, 3 * dd, p) % p for i, dd in enumerate((2, 4, 6, 10)))
    res["ic_gl2_covariance_ok"] = bool(ok)
    W, H5 = h5_fit(p, wr)
    res["H5_weighted_degree"] = W
    res["H5_terms"] = len(H5)
    # TTV curves y^2 = D5(x; a) + c and random sextics
    ttv = []
    for _ in range(5):
        a, c = wr.randrange(1, p), wr.randrange(p)
        f = [c, 5 * a * a % p, 0, (-5 * a) % p, 0, 1, 0]
        ttv.append(h5_eval(H5, ic_eval(IC, f, p), p))
    res["H5_on_TTV"] = ttv
    res["H5_on_random"] = [h5_eval(H5, ic_eval(IC, [wr.randrange(p) for _ in range(7)], p), p) for _ in range(5)]
    print(json.dumps(res), flush=True)
    assert not any(ttv) and all(res["H5_on_random"])

    Rg = R3(p)

    def G_series(v):
        """G(F0 + e v) in F_p[t][e]/(e^3): list of 3 nmod_poly in t"""
        fc = [Rg.zero() for _ in range(7)]
        for m, k in S6i.items():
            e0, e1, e2 = m
            i = e0 + e1                       # f_t coefficient of x1^i x2^(6-i)
            c0, c1 = int(F0v[k]) % p, int(v[k]) % p
            mono = [0] * e0 + [1]
            if c0:
                fc[i] = Rg.add(fc[i], [flint.nmod_poly(mono, p) * c0, Rg.z, Rg.z])
            if c1:
                fc[i] = Rg.add(fc[i], [Rg.z, flint.nmod_poly(mono, p) * c1, Rg.z])
        Is = [eval_poly_R(I, fc, Rg, 7) for I in IC]
        return eval_poly_R([(m, v_) for m, v_ in H5], Is, Rg, 4)

    def coeffs(poly, L):
        c = [int(x) for x in poly.coeffs()]
        return np.array(c + [0] * (L - len(c)), dtype=np.int64)

    zero = np.zeros(28, dtype=np.int64)
    G00 = G_series(zero)
    res["G_F0_is_zero"] = G00[0].degree() < 0
    print('G(F0) == 0:', res["G_F0_is_zero"], flush=True)
    cols = []
    for k in range(28):
        v = zero.copy(); v[k] = 1
        cols.append(G_series(v)[1])
    vr = np.array([wr.randrange(p) for _ in range(28)], dtype=np.int64)
    L = max([c.degree() for c in cols] + [c.degree() for c in G_series(vr)]) + 1
    Jm = np.array([coeffs(c, L) for c in cols], dtype=np.int64).T % p          # L x 28
    rJ = rk(Jm, p)
    Wb = nullsp(Jm, p)
    res["rank_J"] = rJ
    res["dim_kerJ"] = len(Wb)
    gens3 = [x0, x1, x2]
    gl3 = np.array([vec28(gens3[k] * F0.derivative(i)) for i in range(3) for k in range(3)], dtype=np.int64)
    stab = np.array([vec28(gens3[k] * F0.derivative(i)) for i in range(3) for k in range(3)
                     if not (k == 2 and i in (0, 1))], dtype=np.int64)
    vgs = [vec28(x0 ** (6 - k) * x1 ** k) for k in range(7)]
    dD5 = -5 * x2 ** 3 + 10 * abin * x2
    vgs += [vec28(x1 * x0 ** (2 - k) * x1 ** k * dD5) for k in range(3)]
    vgs = np.array(vgs, dtype=np.int64)
    rg = rk(gl3, p)
    rW = rk(Wb, p)
    res["stab_in_kerJ"] = rk(np.vstack([Wb, stab]), p) == rW
    res["vGS_in_kerJ"] = rk(np.vstack([Wb, vgs]), p) == rW
    res["dim_kerJ_plus_gl3_mod_gl3"] = rk(np.vstack([Wb, gl3]), p) - rg
    rm = json.load(open(rmfile))
    assert rm["prime"] == p and rm["seed"] == seed
    RM = np.array([[d.get(",".join(map(str, m)), 0) for m in S6p] for d in rm["basis"]], dtype=np.int64) % p
    rRM = rk(RM, p)
    dWG = rk(np.vstack([Wb, gl3]), p)
    res["RMtan_in_kerJ_plus_gl3"] = rk(np.vstack([Wb, gl3, RM]), p) == dWG
    U = np.vstack([vgs, stab]) % p
    rU = rk(U, p)
    res["dim_U"] = rU
    # E = (RM + gl3) cap ker J
    RG = np.vstack([RM, gl3]) % p
    rRG = rk(RG, p)
    # intersection of row spaces: solve a Wb = b RG
    Nn = nullsp(np.vstack([Wb, RG]).T, p)                    # coefficient vectors (a, b)
    Eb = (Nn[:, :len(Wb)] @ Wb) % p
    rE = rk(Eb, p)
    res["dim_(RM+gl3)_cap_kerJ"] = rE
    res["U_in_E"] = rk(np.vstack([Eb, U]), p) == rE
    # pick e_RM in E not in span(U)
    eRM = None
    for row in Eb:
        if rk(np.vstack([U, row[None, :]]), p) > rU:
            eRM = row % p
            break
    assert eRM is not None
    print(json.dumps({k: res[k] for k in res if k.startswith(('dim', 'rank', 'stab', 'vGS', 'RM', 'U_', 'G_'))}),
          flush=True)

    def Qv(v):
        return coeffs(G_series(v)[2], L)

    def in_imJ(vecs):
        return rk(np.hstack([Jm, np.array(vecs, dtype=np.int64).T % p]), p) - rJ

    # validation: Q on U lands in Im J (random combinations)
    QU = []
    for _ in range(4):
        u = np.zeros(28, dtype=np.int64)
        for row in U:
            u = (u + wr.randrange(p) * row) % p
        QU.append(Qv(u))
    res["Q_U_outside_ImJ"] = in_imJ(QU)
    Qe = Qv(eRM)
    res["Q_eRM_outside_ImJ"] = in_imJ([Qe])
    Bs = []
    Ub = nullsp(nullsp(U, p), p)                               # row basis of span(U)
    for u in Ub:
        Bs.append(((Qv((u + eRM) % p) - Qv(u) - Qe) * pow(2, p - 2, p)) % p)
    res["B_U_eRM_outside_ImJ"] = in_imJ(Bs)
    # discrimination: the coker-valued quadratic map on ker J / U (3-dim), basis c1 = e_RM, c2, c3
    comp = [eRM]
    for row in Wb:
        if rk(np.vstack([U] + [c[None, :] for c in comp] + [row[None, :]]), p) > rU + len(comp):
            comp.append(row % p)
        if len(comp) == 3:
            break
    res["dim_kerJ_mod_U"] = len(comp)
    BUc = []
    for c in comp:
        for u in Ub:
            BUc.append(((Qv((u + c) % p) - Qv(u) - Qv(c)) * pow(2, p - 2, p)) % p)
    res["B_U_kerJ_outside_ImJ"] = in_imJ(BUc)
    # coker projection: left kernel of J
    Lk = nullsp(Jm.T, p)                                       # rows y with y J = 0
    proj = lambda vec: (Lk @ (vec % p)) % p
    Qc = [Qv(c) for c in comp]
    Bc = {}
    for i in range(3):
        for j in range(i + 1, 3):
            Bc[(i, j)] = ((Qv((comp[i] + comp[j]) % p) - Qc[i] - Qc[j]) * pow(2, p - 2, p)) % p
    # quadratic form q(x) = sum x_i^2 Q_i + 2 sum_{i<j} x_i x_j B_ij in coker coordinates; its 6 coefficient vectors
    qcoef = [proj(Qc[0]), proj(Qc[1]), proj(Qc[2]), proj(2 * Bc[(0, 1)]), proj(2 * Bc[(0, 2)]), proj(2 * Bc[(1, 2)])]
    Qm = np.array(qcoef, dtype=np.int64).T % p                 # coker_dim x 6 (monomials x0^2, x1^2, x2^2, x0x1, x0x2, x1x2)
    res["coker_dim"] = int(len(Lk))
    res["rank_quadric_system"] = rk(Qm, p)
    # zero set of q on P^2(F_p): count points (the cone of second-order-unobstructed directions mod U)
    Qm = Qm[np.any(Qm, axis=1)]
    pts = []

    def roots_common(polys):
        g = None
        for P in polys:
            if P.degree() < 0:
                continue
            g = P if g is None else g.gcd(P)
        if g is None:
            return 'all'
        if g.degree() == 0:
            return []
        return [int(-f[0]) * pow(int(f[1]), p - 2, p) % p for f, m in g.factor()[1] if f.degree() == 1]

    if len(Qm):
        # chart x0 = 1, x1 = a: q(1, a, b) = (Q2) b^2 + (Q4 + a Q5) b + (Q0 + a Q3 + a^2 Q1)
        for a in range(p):
            polys = [flint.nmod_poly([int(r[0] + a * r[3] + a * a * r[1]) % p, int(r[4] + a * r[5]) % p, int(r[2])], p)
                     for r in Qm]
            rc = roots_common(polys)
            if rc == 'all':
                pts.append((1, a, '*'))
            else:
                pts += [(1, a, b) for b in rc]
            if len(pts) > 50:
                break
        # line x0 = 0: (0, 1, b) and (0, 0, 1)
        polys = [flint.nmod_poly([int(r[1]), int(r[5]), int(r[2])], p) for r in Qm]
        rc = roots_common(polys)
        pts += [(0, 1, '*')] if rc == 'all' else [(0, 1, b) for b in rc]
        if not any(int(r[2]) for r in Qm):
            pts.append((0, 0, 1))
    else:
        pts = ['all of P^2']
    res["second_order_cone_points_P2"] = pts[:20]
    res["second_order_cone_npoints"] = len(pts) if len(pts) <= 50 else '>50'
    print(json.dumps({k: res[k] for k in ("dim_kerJ_mod_U", "B_U_kerJ_outside_ImJ", "coker_dim", "rank_quadric_system",
                                          "second_order_cone_npoints", "second_order_cone_points_P2")}), flush=True)
    res["note"] = "outside_ImJ = rank increase of [J | vectors] over rank J (0 = in Im J)"
    print(json.dumps({k: res[k] for k in ("Q_U_outside_ImJ", "Q_eRM_outside_ImJ", "B_U_eRM_outside_ImJ")}), flush=True)
    with open(out, "w", newline="\n") as fh:
        fh.write(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
