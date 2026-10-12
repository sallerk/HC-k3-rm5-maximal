"""Exact (over Q) checks for the fibre family  y^2 = f = D5(z; b) + nu (1 - eps z) + (eps/4) z^6.
(1) integer Igusa-Clebsch polynomials (roots convention of b2_humbert, times f6^{2,4,6,10}) and the integer H5
    equation, by CRT over 5 primes (balanced residues), validated against a 6th prime.
(2) symbolic: H5(IC(f)) == 0 in Q[b, nu, eps]  (exact identity).
(3) symbolic: (g, h) of Elkies-Kumar Cor. 15 as rational functions of (b, nu, eps), and the Y_-(5) branch function
    z^2 = 2(6250h^2 - ...) (EK Thm 16): is it c * square in Q(b, nu, eps)?  (=> RM defined over Qbar(b, nu, eps),
    so along every curve in the parameter space, in particular along every pencil of the 8-dim plane family).
(4) mod-p Legendre test of z^2 at random (b, nu, eps), two primes (cheap cross-check of (3)).
Usage: python -I b8_exact.py
"""
import os
import random
import sys

import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b2_humbert as B

PRIMES = [1000003, 1000033, 1000037, 1000039, 1000081]
CHECK = 1000099


def crt(res, mods):
    M = 1
    for m in mods:
        M *= m
    x = 0
    for r, m in zip(res, mods):
        Mi = M // m
        x += r * Mi * pow(Mi, -1, m)
    x %= M
    return x - M if x > M // 2 else x


def ratrec(a, M):
    """rational reconstruction of a mod M (Wang)"""
    r0, r1, s0, s1 = M, a % M, 0, 1
    bound = int((M // 2) ** 0.5)
    while r1 > bound:
        q = r0 // r1
        r0, r1, s0, s1 = r1, r0 - q * r1, s1, s0 - q * s1
    return sp.Rational(r1, s1)


def main():
    ics, h5s = [], []
    for p in PRIMES + [CHECK]:
        rng = random.Random(p)
        ics.append({e: v for e, v in [(e, v) for I in B.fit_ic(p, rng) for e, v in I]})
        W, H = B.h5_fit(p, rng)
        lead = H[0][1]
        h5s.append({m: v * pow(lead, p - 2, p) % p for m, v in H})       # normalise first coefficient to 1
    keys = sorted(set().union(*[set(d) for d in ics]))
    IC = {e: crt([d.get(e, 0) for d in ics[:5]], PRIMES) for e in keys}
    ok = all(IC[e] % CHECK == ics[5].get(e, 0) for e in keys)
    hkeys = sorted(set().union(*[set(d) for d in h5s]))
    M = 1
    for p in PRIMES:
        M *= p
    H5 = {m: ratrec(crt([d.get(m, 0) for d in h5s[:5]], PRIMES) % M, M) for m in hkeys}
    den = sp.ilcm(*[c.q for c in H5.values()])
    H5 = {m: int(c * den) for m, c in H5.items()}
    okh = all(H5[m] % CHECK == h5s[5].get(m, 0) * den % CHECK for m in hkeys)
    print('(1) IC integer lift validated at 6th prime:', ok, '| max |coef|:', max(abs(v) for v in IC.values()))
    print('    H5 integer lift validated:', okh, '| H5 =', H5)
    b, nu, eps = sp.symbols('b nu eps')
    f = [nu, 5 * b * b - nu * eps, 0, -5 * b, 0, 1, eps / 4]
    I = []
    for d in (2, 4, 6, 10):
        terms = [(e, v) for e, v in IC.items() if sum(e) == d and v]
        I.append(sp.expand(sum(v * sp.Mul(*[f[k] ** e[k] for k in range(7)]) for e, v in terms)))
    Hval = sp.expand(sum(c * I[0] ** a * I[1] ** bb * I[2] ** cc * I[3] ** dd for (a, bb, cc, dd), c in H5.items()))
    print('(2) H5(IC(f)) == 0 identically in Q[b, nu, eps]:', Hval == 0)
    # (3) (g, h): I ~ lam-weighted (6(4g+1), 9g^2, 9(4h+9g^3+2g^2), 4h^2)
    g = sp.Symbol('g')
    u = sp.cancel(I[1] / I[0] ** 2)
    quad = sp.Poly(sp.numer(sp.together((1 - 64 * u) * g ** 2 - 32 * u * g - 4 * u)), g)
    facs = sp.factor_list(quad.as_expr(), g)
    print('(3) factors of the g-quadratic:', [(sp.Poly(fc, g).degree(), m) for fc, m in facs[1]])
    found = False
    for fc, m in facs[1]:
        P = sp.Poly(fc, g)
        if P.degree() != 1:
            continue
        gv = sp.cancel(-P.all_coeffs()[1] / P.all_coeffs()[0])
        lam = sp.cancel(I[0] / (6 * (4 * gv + 1)))
        hv = sp.cancel((I[2] / (9 * lam ** 3) - 9 * gv ** 3 - 2 * gv ** 2) / 4)
        c4 = sp.cancel(9 * gv ** 2 * lam ** 2 - I[1])
        c10 = sp.cancel(4 * hv ** 2 * lam ** 5 - I[3])
        if c4 != 0 or c10 != 0:
            print('    root g =', gv, 'inconsistent')
            continue
        found = True
        z2 = sp.factor(2 * (6250 * hv ** 2 - 4500 * gv ** 2 * hv - 1350 * gv * hv - 108 * hv - 972 * gv ** 5
                            - 324 * gv ** 4 - 27 * gv ** 3))
        print('    g =', sp.factor(gv))
        print('    h =', sp.factor(hv))
        print('    z^2 =', z2)
        nz, dz = sp.fraction(sp.together(z2))
        for lab, part in (('numerator', nz), ('denominator', dz)):
            cst, fl = sp.factor_list(sp.expand(part))
            print('    %s: constant %s, odd-multiplicity factors %s' % (lab, cst, [fc2 for fc2, m2 in fl if m2 % 2]))
    print('    consistent root found:', found)
    # (4) Legendre cross-check
    for p in (1000003, 31981):
        rng = random.Random(p + 1)
        ICp = [[(e, v % p) for e, v in IC.items() if sum(e) == d] for d in (2, 4, 6, 10)]
        leg = lambda a: 0 if a % p == 0 else (1 if pow(a % p, (p - 1) // 2, p) == 1 else -1)
        out = []
        for _ in range(40):
            bv, nv, ev = rng.randrange(p), rng.randrange(1, p), rng.randrange(1, p)
            fp = [nv, (5 * bv * bv - nv * ev) % p, 0, (-5 * bv) % p, 0, 1, ev * pow(4, p - 2, p) % p]
            Ip = B.ic_eval(ICp, fp, p)
            uu = Ip[1] * pow(Ip[0] * Ip[0], p - 2, p) % p
            import flint
            Q = flint.nmod_poly([(-4 * uu) % p, (-32 * uu) % p, (1 - 64 * uu) % p], p)
            syms = set()
            for fct, m in Q.factor()[1]:
                if fct.degree() != 1:
                    continue
                gg = int(-fct[0]) * pow(int(fct[1]), p - 2, p) % p
                la = Ip[0] * pow(6 * (4 * gg + 1), p - 2, p) % p
                hh = (Ip[2] * pow(9 * pow(la, 3, p), p - 2, p) - 9 * gg ** 3 - 2 * gg * gg) * pow(4, p - 2, p) % p
                if (4 * hh * hh * pow(la, 5, p) - Ip[3]) % p == 0 and (9 * gg * gg * la * la - Ip[1]) % p == 0:
                    syms.add(leg(2 * (6250 * hh * hh - 4500 * gg * gg * hh - 1350 * gg * hh - 108 * hh
                                      - 972 * gg ** 5 - 324 * gg ** 4 - 27 * gg ** 3)))
            out.append(tuple(sorted(syms)))
        print('(4) p = %d, Legendre(5) = %d: symbol sets at 40 random (b, nu, eps): %s'
              % (p, leg(5), {s: out.count(s) for s in set(out)}))


if __name__ == '__main__':
    main()
