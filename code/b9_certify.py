"""Certify the inputs of b8_exact.py and two side facts.
(a) H5 integer equation (as lifted in b8) vanishes on the Elkies-Kumar parametrisation, exactly in Q[g, h];
    H5 irreducible over Q.
(b) integer IC polynomials (b8 lift) == Igusa-Clebsch from roots (b2_humbert convention), exactly over Q, at random
    integer-root sextics with random leading coefficient (Schwartz-Zippel on roots; 6 points per invariant).
(c) (b, nu, eps) -> (g, h) has generic rank 2 (fibre family dominates H5).
(d) the plane sextic of the 8-parameter family is smooth at a random parameter point (dim R_13 of the Jacobian ring
    = 0, mod p), and P0 = (0:0:1) is off the curve (eps != 0).
Usage: python -I b9_certify.py
"""
import itertools
import os
import random
import sys
from fractions import Fraction

import flint
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b2_humbert as B
import b8_exact as E


def lift():
    ics, h5s = [], []
    for p in E.PRIMES:
        rng = random.Random(p)
        ics.append({e: v for I in B.fit_ic(p, rng) for e, v in I})
        W, H = B.h5_fit(p, rng)
        lead = H[0][1]
        h5s.append({m: v * pow(lead, p - 2, p) % p for m, v in H})
    keys = sorted(set().union(*[set(d) for d in ics]))
    IC = {e: E.crt([d.get(e, 0) for d in ics], E.PRIMES) for e in keys}
    M = 1
    for p in E.PRIMES:
        M *= p
    hk = sorted(set().union(*[set(d) for d in h5s]))
    H5 = {m: E.ratrec(E.crt([d.get(m, 0) for d in h5s], E.PRIMES) % M, M) for m in hk}
    den = sp.ilcm(*[c.q for c in H5.values()])
    return IC, {m: int(c * den) for m, c in H5.items()}


def main():
    IC, H5 = lift()
    g, h = sp.symbols('g h')
    I2, I4, I6, I10 = sp.symbols('I2 I4 I6 I10')
    Hpoly = sum(c * I2 ** a * I4 ** b * I6 ** cc * I10 ** d for (a, b, cc, d), c in H5.items())
    sub = {I2: 6 * (4 * g + 1), I4: 9 * g ** 2, I6: 9 * (4 * h + 9 * g ** 3 + 2 * g ** 2), I10: 4 * h ** 2}
    print('(a) H5 on EK parametrisation == 0:', sp.expand(Hpoly.subs(sub)) == 0,
          '| factor_list of H5 over Q:', [(sp.Poly(f, I2, I4, I6, I10).total_degree(), m)
                                          for f, m in sp.factor_list(Hpoly)[1]])
    rng = random.Random(4242)
    ok = True
    for _ in range(6):
        r = [rng.randrange(-10 ** 6, 10 ** 6) for _ in range(6)]
        f6 = rng.randrange(1, 10 ** 6)
        poly = [Fraction(1)]
        for a in r:
            new = [Fraction(0)] * (len(poly) + 1)
            for k, c in enumerate(poly):
                new[k + 1] += c
                new[k] -= a * c
            poly = new
        f = [c * f6 for c in poly]
        d2 = {(i, j): (r[i] - r[j]) ** 2 for i in range(6) for j in range(6)}
        J2 = sum(d2[a] * d2[b] * d2[c] for a, b, c in B.PAIRINGS)
        J4, J6 = 0, 0
        for T1, T2 in B.SPLITS:
            tri = (d2[(T1[0], T1[1])] * d2[(T1[1], T1[2])] * d2[(T1[2], T1[0])]
                   * d2[(T2[0], T2[1])] * d2[(T2[1], T2[2])] * d2[(T2[2], T2[0])])
            J4 += tri
            for sg in itertools.permutations(T2):
                J6 += tri * d2[(T1[0], sg[0])] * d2[(T1[1], sg[1])] * d2[(T1[2], sg[2])]
        J10 = 1
        for i in range(6):
            for j in range(i + 1, 6):
                J10 *= d2[(i, j)]
        roots = [J2 * f6 ** 2, J4 * f6 ** 4, J6 * f6 ** 6, J10 * f6 ** 10]
        poly_vals = []
        for d in (2, 4, 6, 10):
            poly_vals.append(sum(v * sp.prod([f[k] ** e[k] for k in range(7)]) for e, v in IC.items() if sum(e) == d))
        ok &= all(Fraction(roots[i]) == Fraction(poly_vals[i]) for i in range(4))
    print('(b) integer IC polys == IC from roots at 6 random exact sextics:', ok)
    b, nu, eps = sp.symbols('b nu eps')
    gv = -(4 * b ** 2 + eps * nu) / (30 * b ** 2)
    hv = -(125 * b ** 6 * eps ** 2 - 64 * b ** 5 - 75 * b ** 4 * eps ** 3 * nu + 40 * b ** 3 * eps * nu
           + 15 * b ** 2 * eps ** 4 * nu ** 2 - 40 * b * eps ** 2 * nu ** 2 - eps ** 5 * nu ** 3 + 16 * nu ** 2) / (100000 * b ** 5)
    Jm = sp.Matrix([[sp.diff(x, v) for v in (b, nu, eps)] for x in (gv, hv)])
    pt = {b: sp.Rational(3, 7), nu: sp.Rational(-5, 11), eps: sp.Rational(2, 13)}
    print('(c) rank of d(g, h)/d(b, nu, eps) at a rational point:', Jm.subs(pt).rank())
    p = 1000003
    rng = random.Random(77)
    ctx = flint.nmod_mpoly_ctx.get(('x0', 'x1', 'x2'), modulus=p)
    x0, x1, x2 = ctx.gens()
    inv = lambda a: pow(a % p, p - 2, p)
    pc = [rng.randrange(1, p) for _ in range(7)]
    bc = [rng.randrange(1, p) for _ in range(4)]
    ep = 5 * bc[0] ** 2 * inv(pc[0]) % p
    pb = sum(pc[k] * x0 ** (6 - k) * x1 ** k for k in range(7))
    Bc = sum(bc[k] * x0 ** (3 - k) * x1 ** k for k in range(4))
    num = pb * (x1 - ep * x2) + x1 * x1 * x2 ** 5 - 5 * Bc * x1 * x2 ** 3 + 5 * Bc * Bc * x2
    Fq, rem = divmod(num, x1)
    F = Fq + ep * inv(4) * x2 ** 6
    mons13 = B.mons(3, 13)
    mons8 = B.mons(3, 8)
    idx = {m: i for i, m in enumerate(mons13)}
    rows = []
    for dv in (F.derivative(0), F.derivative(1), F.derivative(2)):
        for m in mons8:
            prod = dv * x0 ** m[0] * x1 ** m[1] * x2 ** m[2]
            row = [0] * len(mons13)
            for ex, cf in prod.to_dict().items():
                row[idx[tuple(int(u) for u in ex)]] = int(cf)
            rows.append(row)
    Mx = flint.nmod_mat(len(rows), len(mons13), [x for r_ in rows for x in r_], p)
    print('(d) remainder zero:', rem == 0, '| dim R_13 =', len(mons13) - Mx.rank(), '(0 = smooth)',
          '| F(0,0,1) =', int(F.to_dict().get((0, 0, 6), 0)))


if __name__ == '__main__':
    main()
