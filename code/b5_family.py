"""Analyse a quadratic arc F(s) = F0 + s w1 + s^2 w2 found by b4_jet.py (lifted branch with nonzero orders [1, 2]).
(1) exactness: G(F(s)) == 0 as a polynomial in s (degree <= 2 * 2W = 48): evaluate to order 49.
(2) fibrewise isotriviality: are the fibres w^2 = F(s)(t x1, x1, x2) isomorphic to those of F0 (equal IC point)?
(3) print F(s) - F0 and F0 (coefficients mod p).
Usage: python -I b5_family.py B4OUT.json prime seed
"""
import json
import os
import random
import sys

import flint
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b2_humbert as B
from b3_arc import RN


def main():
    fn, p, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    d = json.load(open(fn))
    br = next(b for b in d["branches"] if b.get("nonzero_orders") == [1, 2])
    S6p = [tuple(m) for m in br["S6_monomials"]]
    w1 = np.array(br["w"]["1"], dtype=np.int64)
    w2 = np.array(br["w"]["2"], dtype=np.int64)
    rng = random.Random(1000 + seed)
    pcoef = [rng.randrange(p) for _ in range(7)]
    acoef = [rng.randrange(p) for _ in range(3)]
    ctx = flint.nmod_mpoly_ctx.get(('x0', 'x1', 'x2', 's'), modulus=p)
    x0, x1, x2, s = ctx.gens()
    pbin = sum(pcoef[k] * x0 ** (6 - k) * x1 ** k for k in range(7))
    abin = sum(acoef[k] * x0 ** (2 - k) * x1 ** k for k in range(3))
    F0 = pbin + x1 * (x2 ** 5 - 5 * abin * x2 ** 3 + 5 * abin ** 2 * x2)
    mono = lambda m: x0 ** m[0] * x1 ** m[1] * x2 ** m[2]
    W1 = sum(int(w1[i]) * mono(m) for i, m in enumerate(S6p) if w1[i])
    W2 = sum(int(w2[i]) * mono(m) for i, m in enumerate(S6p) if w2[i])
    Fs = F0 + s * W1 + s * s * W2
    wr = random.Random(5000 + seed)
    IC = B.fit_ic(p, wr)
    W, H5 = B.h5_fit(p, wr)
    n = 4 * W + 2
    Rg = RN(p, n)
    fc = [Rg.zero() for _ in range(7)]
    for ex, cf in Fs.to_dict().items():
        e0, e1, e2, es = (int(t) for t in ex)
        comp = [Rg.z] * n
        comp[es] = flint.nmod_poly([0] * e0 + [1], p) * int(cf)
        fc[e0 + e1] = Rg.add(fc[e0 + e1], comp)
    Is = [B.eval_poly_R(I, fc, Rg, 7) for I in IC]
    Gs = B.eval_poly_R(H5, Is, Rg, 4)
    out = {"G_nonzero_s_orders_up_to_%d" % (n - 1): [j for j in range(n) if Gs[j].degree() >= 0]}
    # fibrewise IC comparison at random (t, s)
    def fib(F, t, sv):
        fd = [0] * 7
        for ex, cf in F.to_dict().items():
            e0, e1, e2, es = (int(u) for u in ex)
            fd[e0 + e1] = (fd[e0 + e1] + int(cf) * pow(t, e0, p) * pow(sv, es, p)) % p
        return fd
    iso = []
    for _ in range(6):
        t, sv = wr.randrange(1, p), wr.randrange(1, p)
        I0 = B.ic_eval(IC, fib(Fs, t, 0), p)
        Is_ = B.ic_eval(IC, fib(Fs, t, sv), p)
        # weighted-projective equality: I4/I2^2, I6/I2^3, I10/I2^5
        inv = lambda a: pow(a, p - 2, p)
        r0 = [I0[1] * inv(I0[0] ** 2) % p, I0[2] * inv(I0[0] ** 3) % p, I0[3] * inv(I0[0] ** 5) % p]
        r1 = [Is_[1] * inv(Is_[0] ** 2) % p, Is_[2] * inv(Is_[0] ** 3) % p, Is_[3] * inv(Is_[0] ** 5) % p]
        iso.append(r0 == r1)
    out["fibres_isomorphic_to_F0_fibres"] = iso
    out["F0_at_P0_after_deformation"] = "coefficient of x2^6 in w2: %d" % int(w2[S6p.index((0, 0, 6))])
    print(json.dumps(out))
    print("pcoef", pcoef, "acoef", acoef)
    print("w1 =", W1)
    print("w2 =", W2)


if __name__ == "__main__":
    main()
