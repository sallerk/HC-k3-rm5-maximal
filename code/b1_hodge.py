"""Hodge-locus tangent of the class of Q_{c,1} (and of its mu_6-isotypic parts) on the diagonal family.
The point F0 is fixed by the seed. G = l1 A1 + l2 A2 + q A3 exactly; P_Q = det of the Jacobian of
(l1, l2, q, A1, A2, A3) in R^G_12 (Villaflor Loyola's formula, up to a constant). R^G = R^F(x) (x) R^F(y) (bigraded).
mu_6 acts by y -> zeta y; on P Omega / G^3 with y-degree b by zeta^(b+3). Rational isotypic sectors by b mod 6:
'm1' b = 0 (eigenvalue -1: H^2(K3) (x) H^2(K3) of the double plane w^2 = F), 'p1' b = 3, 'o3' b = 1, 5, 'o6' b = 2, 4.
T_HL(P) = {Phi : Phi P = 0 in R^G_18}. Reports, on D = {f(x) + f(y)}, dim T_HL - 9 for P and each sector, and checks
W(Q) subset T_HL(P_Q) in all of S_6 (validates the det formula).
Usage: python -I b1_hodge.py OUT.json prime seed [one|all|hex] [j=1|2] [RMTAN.json]
"""
import itertools
import json
import random
import sys

import flint
import numpy as np


def mons(nv, n):
    return [e for e in itertools.product(range(n + 1), repeat=nv) if sum(e) == n][::-1]


def main():
    out, p, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rng = random.Random(1000 + seed)
    gen = next(x for x in range(2, p) if all(pow(x, (p - 1) // q, p) != 1 for q in {2, 3, 5} | {
        q for q in range(7, 200) if (p - 1) % q == 0 and all(q % r for r in range(2, q))}))
    z5 = pow(gen, (p - 1) // 5, p)
    z12 = pow(gen, (p - 1) // 12, p)
    cs = [pow(z12, k, p) for k in (1, 3, 5, 7, 9, 11)]
    jq = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    alpha = (pow(z5, jq, p) + pow(z5, 5 - jq, p)) % p
    ctx = flint.nmod_mpoly_ctx.get(('x0', 'x1', 'x2', 'y0', 'y1', 'y2'), modulus=p)
    X0, X1, X2, Y0, Y1, Y2 = ctx.gens()
    pcoef = [rng.randrange(p) for _ in range(7)]
    acoef = [rng.randrange(p) for _ in range(3)]
    pbin = sum(pcoef[k] * X0 ** (6 - k) * X1 ** k for k in range(7))
    abin = sum(acoef[k] * X0 ** (2 - k) * X1 ** k for k in range(3))
    F0 = pbin + X1 * (X2 ** 5 - 5 * abin * X2 ** 3 + 5 * abin ** 2 * X2)
    gens = [X0, X1, X2, Y0, Y1, Y2]
    Fy = F0.compose(Y0, Y1, Y2, *gens[3:])
    G = F0 + Fy
    # plane-curve Jacobian ring: NF matrices N[n] : S_n(3 vars) -> R_n, and standard monomials
    M3 = {n: mons(3, n) for n in range(19)}
    I3 = {n: {m: k for k, m in enumerate(M3[n])} for n in range(19)}
    dF = [{(e[0], e[1], e[2]): int(v) for e, v in F0.derivative(i).to_dict().items()} for i in range(3)]
    N, STD = {}, {}
    for n in range(19):
        rows = []
        if n >= 5:
            for i in range(3):
                for m in M3[n - 5]:
                    r = [0] * len(M3[n])
                    for e, v in dF[i].items():
                        r[I3[n][(e[0] + m[0], e[1] + m[1], e[2] + m[2])]] = v
                    rows.append(r)
        s = len(M3[n])
        if rows:
            E, r = flint.nmod_mat(len(rows), s, [x for row in rows for x in row], p).rref()
            E = np.array([int(x) for x in E.entries()], dtype=np.int64).reshape(len(rows), s)[:r]
        else:
            E, r = np.zeros((0, s), dtype=np.int64), 0
        piv = [int(np.nonzero(E[i])[0][0]) for i in range(r)]
        std = [j for j in range(s) if j not in set(piv)]
        Nn = np.zeros((len(std), s), dtype=np.int64)
        for k, j in enumerate(std):
            Nn[k, j] = 1
        if r:
            Nn[:, piv] = (-E[:, std].T) % p
        N[n], STD[n] = Nn, std
    rdim = [len(STD[n]) for n in range(19)]
    res = {"prime": p, "seed": seed, "quadric_j": jq, "dimR_F": rdim, "plane_sextic_smooth": rdim[13] == 0}
    assert rdim[13] == 0

    # multiplication maps by 3-var monomials alpha (deg <= 6): R_a -> R_{a+|alpha|}
    mult = {}
    for k in range(7):
        for al in M3[k]:
            for a in range(13 - k):
                Mm = np.zeros((rdim[a + k], rdim[a]), dtype=np.int64)
                for col, j in enumerate(STD[a]):
                    m = M3[a][j]
                    v = np.zeros(len(M3[a + k]), dtype=np.int64)
                    v[I3[a + k][(m[0] + al[0], m[1] + al[1], m[2] + al[2])]] = 1
                    Mm[:, col] = (N[a + k] @ v) % p
                mult[(al, a)] = Mm

    def blocks(poly):
        """6-var poly -> {(a, b): NF matrix in R_a (x) R_b}"""
        raw = {}
        for e, v in poly.to_dict().items():
            e = tuple(int(t) for t in e)
            a, b = e[0] + e[1] + e[2], e[3] + e[4] + e[5]
            if a > 12 or b > 12:
                continue
            C = raw.setdefault((a, b), np.zeros((len(M3[a]), len(M3[b])), dtype=np.int64))
            C[I3[a][e[:3]], I3[b][e[3:]]] = (C[I3[a][e[:3]], I3[b][e[3:]]] + int(v)) % p
        return {ab: (N[ab[0]] @ C % p) @ N[ab[1]].T % p for ab, C in raw.items()}

    TGT = [(A, 18 - A) for A in range(6, 13)]
    toff, o = {}, 0
    for A, B in TGT:
        toff[(A, B)] = o
        o += rdim[A] * rdim[B]
    res["dim_RG18"] = o
    S6 = mons(6, 6)
    S6i = {m: k for k, m in enumerate(S6)}

    def mult_matrix(Pb):
        """rows: monomials of S_6 (6 vars); row = NF(monomial * P) in R^G_18"""
        Mx = np.zeros((len(S6), o), dtype=np.int64)
        for k, m in enumerate(S6):
            al, be = m[:3], m[3:]
            for (a, b), C in Pb.items():
                A, B = a + sum(al), b + sum(be)
                if (A, B) not in toff:
                    continue
                blk = mult[(al, a)] @ C % p @ mult[(be, b)].T % p
                Mx[k, toff[(A, B)]:toff[(A, B)] + blk.size] = (Mx[k, toff[(A, B)]:toff[(A, B)] + blk.size]
                                                              + blk.flatten()) % p
        return Mx

    def rk(M):
        M = np.asarray(M, dtype=np.int64) % p
        if M.size == 0:
            return 0
        return flint.nmod_mat(M.shape[0], M.shape[1], [int(x) for x in M.flatten()], p).rank()

    def vec6(poly):
        v = np.zeros(len(S6), dtype=np.int64)
        for e, c in poly.to_dict().items():
            v[S6i[e]] = int(c)
        return v

    # diagonal family and gl_3
    S6p = mons(3, 6)
    Dm = []
    for m in S6p:
        f = X0 ** m[0] * X1 ** m[1] * X2 ** m[2]
        Dm.append(vec6(f + f.compose(Y0, Y1, Y2, *gens[3:])))
    Dm = np.array(Dm, dtype=np.int64)
    res["rank_D"] = rk(Dm)
    sectors = {"m1": {0, 6, 12}, "p1": {3, 9}, "o3": {1, 5, 7, 11}, "o6": {2, 4, 8, 10}}
    res["per_c"] = []
    for ci, c in enumerate(cs):
        l1, l2 = Y0 - c * X0, Y1 - c * X1
        qq = c * c * X2 ** 2 - alpha * c * X2 * Y2 + Y2 ** 2 + (alpha * alpha - 4) * c * c * abin
        F_a = F0.compose(c * X0, Y1, Y2, *gens[3:])
        F_b = F0.compose(c * X0, c * X1, Y2, *gens[3:])
        A1, r1 = divmod(Fy - F_a, l1)
        A2, r2 = divmod(F_a - F_b, l2)
        A3, r3 = divmod(F0 + F_b, qq)
        assert r1 == 0 and r2 == 0 and r3 == 0 and l1 * A1 + l2 * A2 + qq * A3 == G
        rows = [qq, A1, A2, A3]
        cols = [lambda f, c=c: f.derivative(0) + c * f.derivative(3),
                lambda f, c=c: f.derivative(1) + c * f.derivative(4),
                lambda f: f.derivative(2), lambda f: f.derivative(5)]
        Jm = [[cols[j](rows[i]) for j in range(4)] for i in range(4)]
        P = ctx.from_dict({})
        for perm in itertools.permutations(range(4)):
            sgn = 1
            for i in range(4):
                for j in range(i + 1, 4):
                    if perm[i] > perm[j]:
                        sgn = -sgn
            t = Jm[0][perm[0]] * Jm[1][perm[1]] * Jm[2][perm[2]] * Jm[3][perm[3]]
            P = P + t if sgn > 0 else P - t
        Pb = blocks(P)
        info = {"c": c, "P_nonzero_blocks": sorted([[int(a), int(b)] for (a, b), C in Pb.items() if C.any()])}
        Mfull = mult_matrix(Pb)
        info["rank_mult_P"] = rk(Mfull)
        # W(Q) in S_6 and the check W(Q) . P = 0
        Wl = []
        for f, k in ((l1, 5), (l2, 5), (qq, 4), (A1, 1), (A2, 1), (A3, 2)):
            for m in mons(6, k):
                Wl.append(vec6(f * X0 ** m[0] * X1 ** m[1] * X2 ** m[2] * Y0 ** m[3] * Y1 ** m[4] * Y2 ** m[5]))
        Wl = np.array(Wl, dtype=np.int64)
        info["dim_W"] = rk(Wl)
        info["W_in_THL"] = not ((Wl @ Mfull) % p).any()
        info["dim_THL_full_S6"] = len(S6) - info["rank_mult_P"]
        info["dim_THL_cap_D_mod_gl3"] = len(S6p) - rk(Dm @ Mfull % p) - 9
        for sname, bs in sectors.items():
            Ps = {ab: C for ab, C in Pb.items() if ab[1] in bs}
            if not any(C.any() for C in Ps.values()):
                info["sector_" + sname] = "zero"
                continue
            Ms = mult_matrix(Ps)
            info["sector_" + sname] = {"rank_mult": rk(Ms),
                                       "dim_THL_cap_D_mod_gl3": len(S6p) - rk(Dm @ Ms % p) - 9}
            if sname == "m1":
                Mq_m1 = Ms
        res["per_c"].append(info)
        print(json.dumps(info), flush=True)
        if ci == 0 and len(sys.argv) > 6:
            # save a basis of T_HL(delta_-) cap D (as plane sextics f, f(x) + f(y) in D) (read by b2, b4)
            DQ = Dm @ Mq_m1 % p
            Kn, nk = flint.nmod_mat(DQ.shape[1], DQ.shape[0], [int(x) for x in DQ.T.flatten()], p).nullspace()
            Kn = np.array([int(x) for x in Kn.entries()], dtype=np.int64).reshape(DQ.shape[0], DQ.shape[0])[:, :nk]
            basis = [{",".join(map(str, S6p[j])): int(Kn[j, k]) for j in range(len(S6p)) if Kn[j, k]}
                     for k in range(nk)]
            with open(sys.argv[6], "w", newline="\n") as fh:
                fh.write(json.dumps({"prime": p, "seed": seed, "c": c, "dim": int(nk), "basis": basis}) + "\n")
        if ci == 0 and len(sys.argv) > 4 and sys.argv[4] in ("one", "hex"):
            break
    if len(sys.argv) > 4 and sys.argv[4] == "hex":
        # plane Pi_c = {x1 = y1 = y0 - c x0 = 0} (hexaflex line x1 = 0 of F0), c = cs[0]; G = x1 B1 + y1 B2 + l1 B3
        c = cs[0]
        l1 = Y0 - c * X0
        G_a = G.compose(X0, ctx.from_dict({}), X2, Y0, Y1, Y2)
        G_b = G_a.compose(X0, X1, X2, Y0, ctx.from_dict({}), Y2)
        B1, s1 = divmod(G - G_a, X1)
        B2, s2 = divmod(G_a - G_b, Y1)
        B3, s3 = divmod(G_b, l1)
        assert s1 == 0 and s2 == 0 and s3 == 0 and X1 * B1 + Y1 * B2 + l1 * B3 == G
        # det Jac(x1, y1, l1, B1, B2, B3): column x0 += c column y0 clears row l1 -> 3x3 minor in (D0, d_x2, d_y2)
        cols = [lambda f: f.derivative(0) + c * f.derivative(3), lambda f: f.derivative(2),
                lambda f: f.derivative(5)]
        Jm = [[cols[j](g) for j in range(3)] for g in (B1, B2, B3)]
        Ppi = ctx.from_dict({})
        for perm in itertools.permutations(range(3)):
            sgn = 1
            for i in range(3):
                for j in range(i + 1, 3):
                    if perm[i] > perm[j]:
                        sgn = -sgn
            t = Jm[0][perm[0]] * Jm[1][perm[1]] * Jm[2][perm[2]]
            Ppi = Ppi + t if sgn > 0 else Ppi - t
        Pb = blocks(Ppi)
        hx = {"c": c, "P_nonzero_blocks": sorted([[int(a), int(b)] for (a, b), C in Pb.items() if C.any()])}
        Mpi = mult_matrix(Pb)
        Wp = []
        for f, k in ((X1, 5), (Y1, 5), (l1, 5), (B1, 1), (B2, 1), (B3, 1)):
            for m in mons(6, k):
                Wp.append(vec6(f * X0 ** m[0] * X1 ** m[1] * X2 ** m[2] * Y0 ** m[3] * Y1 ** m[4] * Y2 ** m[5]))
        Wp = np.array(Wp, dtype=np.int64)
        hx["dim_W"] = rk(Wp)
        hx["W_in_THL"] = not ((Wp @ Mpi) % p).any()
        hx["dim_THL_full_S6"] = len(S6) - rk(Mpi)
        Zn, nl = flint.nmod_mat(Wp.shape[0], Wp.shape[1], [int(x) for x in (Wp % p).flatten()], p).nullspace()
        Zw = np.array([int(x) for x in Zn.entries()], dtype=np.int64).reshape(Wp.shape[1], Wp.shape[1])[:, :nl]
        DW = Dm @ Zw % p                     # f in D cap W(Pi)  <=>  f^T DW = 0
        DQ = Dm @ Mq_m1 % p                  # f in T_HL(delta_-)
        hx["W_cap_D_mod_gl3"] = len(S6p) - rk(DW) - 9
        hx["THL_cap_D_mod_gl3"] = len(S6p) - rk(Dm @ Mpi % p) - 9
        Mpi_m1 = None
        for sname, bs in sectors.items():
            Ps = {ab: C for ab, C in Pb.items() if ab[1] in bs}
            if not any(C.any() for C in Ps.values()):
                hx["sector_" + sname] = "zero"
                continue
            Ms = mult_matrix(Ps)
            hx["sector_" + sname] = {"rank_mult": rk(Ms), "dim_THL_cap_D_mod_gl3": len(S6p) - rk(Dm @ Ms % p) - 9}
            if sname == "m1":
                Mpi_m1 = Ms
        hx["RM_cap_W(Pi)_cap_D_mod_gl3"] = len(S6p) - rk(np.hstack([DQ, DW])) - 9
        hx["RM_cap_THL(Pi)_cap_D_mod_gl3"] = len(S6p) - rk(np.hstack([DQ, Dm @ Mpi % p])) - 9
        if Mpi_m1 is not None:
            hx["RM_cap_THL(Pi_m1)_cap_D_mod_gl3"] = len(S6p) - rk(np.hstack([DQ, Dm @ Mpi_m1 % p])) - 9
        if len(sys.argv) > 6:
            # save bases (plane sextics f) of T_HL(Pi_m1) cap D (lattice H5 kept) and W(Pi) cap D (hexaflex kept)
            for tag, Mx in (("pim1", Dm @ Mpi_m1 % p), ("hex", DW)):
                Kn, nk = flint.nmod_mat(Mx.shape[1], Mx.shape[0], [int(x) for x in Mx.T.flatten()], p).nullspace()
                Kn = np.array([int(x) for x in Kn.entries()], dtype=np.int64).reshape(Mx.shape[0], Mx.shape[0])[:, :nk]
                basis = [{",".join(map(str, S6p[j])): int(Kn[j, k]) for j in range(len(S6p)) if Kn[j, k]}
                         for k in range(nk)]
                with open(sys.argv[6].replace(".json", "_" + tag + ".json"), "w", newline="\n") as fh:
                    fh.write(json.dumps({"prime": p, "seed": seed, "dim": int(nk), "basis": basis}) + "\n")
        res["hex"] = hx
        print(json.dumps(hx), flush=True)
    with open(out, "w", newline="\n") as fh:
        fh.write(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
