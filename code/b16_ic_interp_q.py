"""Second, independent implementation of the Igusa-Clebsch invariants (cf. b8/b9, which lift by CRT mod primes):
interpolation over Q of I2, I4, I6, I10 as isobaric polynomials in the coefficients a0..a6 (a_i = coefficient of z^i),
from the root formulas (Mestre's sums over 15, 10, 60 terms, times lc^{2,4,6,10}), on integer-root sextics with random
leading coefficient; held-out samples are checked, and I10 is compared with the discriminant of the generic sextic.
Then, exactly over Q: the invariants of f = D5(z; b) + nu (1 - eps z) + (eps/4) z^6 as displayed in the note, the
proportionality (I2:I4:I6:I10) = lambda^(1,2,3,5) (J2, J4, J6, J10)(g, h) with lambda = 125 b^2 (Elkies-Kumar Cor. 15),
and Phi(g, h) = 5 (Q/(2000 b^5))^2 (Elkies-Kumar Thm 16).
Normalisation check, independent of the family: the LMFDB genus-2 curve 529.a.529.1 (End(J) = Z[(1 + sqrt5)/2] over Q
according to the LMFDB): our invariants equal the LMFDB values up to weighted scaling; the curve lies on the
Elkies-Kumar parametrisation, at a single point (g, h), with Phi(g, h) a square in Q; with I2 halved there is no such
point.
Usage: python -I b16_ic_interp_q.py
"""
import itertools
import random

import sympy as sp
from flint import fmpq_mat

R6 = range(6)


def matchings(s):
    if not s:
        yield []
        return
    for j in range(1, len(s)):
        for m in matchings(s[1:j] + s[j + 1:]):
            yield [(s[0], s[j])] + m


M15 = list(matchings(list(R6)))
SPLITS = [(T, tuple(i for i in R6 if i not in T)) for T in itertools.combinations(R6, 3) if 0 in T]
I6T = [(T1, T2, P) for T1, T2 in SPLITS for P in itertools.permutations(T2)]
assert (len(M15), len(SPLITS), len(I6T)) == (15, 10, 60)


def d2(r, i, j):
    return (r[i] - r[j]) ** 2


def tri(r, T):
    return d2(r, T[0], T[1]) * d2(r, T[1], T[2]) * d2(r, T[2], T[0])


def ic_roots(lc, r):
    i2 = sum(d2(r, *m[0]) * d2(r, *m[1]) * d2(r, *m[2]) for m in M15)
    i4 = sum(tri(r, T1) * tri(r, T2) for T1, T2 in SPLITS)
    i6 = sum(tri(r, T1) * tri(r, T2) * d2(r, T1[0], P[0]) * d2(r, T1[1], P[1]) * d2(r, T1[2], P[2])
             for T1, T2, P in I6T)
    i10 = 1
    for i, j in itertools.combinations(R6, 2):
        i10 *= d2(r, i, j)
    return {2: lc ** 2 * i2, 4: lc ** 4 * i4, 6: lc ** 6 * i6, 10: lc ** 10 * i10}


X = sp.Symbol('X')
A = sp.symbols('a0:7')


def coeffs(lc, r):
    return [int(c) for c in sp.Poly(lc * sp.prod([X - ri for ri in r]), X).all_coeffs()[::-1]]


def interpolate(k, rng):
    ms = [c for c in itertools.combinations_with_replacement(range(7), k) if sum(c) == 3 * k]
    n = len(ms) + 25
    samples = []
    for _ in range(n + 5):
        lc, r = rng.choice([1, 2, 3, -1, -2, 5]), [rng.randint(-9, 9) for _ in R6]
        samples.append((coeffs(lc, r), ic_roots(lc, r)[k]))
    M = fmpq_mat(n, len(ms) + 1)
    for i, (cs, val) in enumerate(samples[:n]):
        for j, m in enumerate(ms):
            M[i, j] = int(sp.prod([cs[q] for q in m]))
        M[i, len(ms)] = int(val)
    Rr, rk = M.rref()
    assert rk == len(ms)
    sol = [None] * len(ms)
    for i in range(rk):
        piv = next(j for j in range(len(ms) + 1) if Rr[i, j] != 0)
        assert piv < len(ms)
        sol[piv] = sp.Rational(int(Rr[i, len(ms)].p), int(Rr[i, len(ms)].q))
    poly = sp.expand(sum(sol[j] * sp.prod([A[q] for q in ms[j]]) for j in range(len(ms))))
    held = all(poly.subs(dict(zip(A, cs))) == val for cs, val in samples[n:])
    return poly, len(ms), held


def main():
    rng = random.Random(20261011)
    IC = {}
    for k in (2, 4, 6, 10):
        IC[k], nm, held = interpolate(k, rng)
        print(f'I{k}: {nm} isobaric monomials, held-out samples agree: {held}')
    gen = sum(A[i] * X ** i for i in range(7))
    print('I10 == discriminant of the generic sextic:', sp.expand(sp.discriminant(gen, X) - IC[10]) == 0)
    print('I2 == 6a3^2 - 16a2a4 + 40a1a5 - 240a0a6:',
          sp.expand(IC[2] - (6 * A[3] ** 2 - 16 * A[2] * A[4] + 40 * A[1] * A[5] - 240 * A[0] * A[6])) == 0)

    b, nu, eps = sp.symbols('b nu eps')
    f = X ** 5 - 5 * b * X ** 3 + 5 * b ** 2 * X + nu * (1 - eps * X) + eps / 4 * X ** 6
    sub = dict(zip(A, sp.Poly(f, X).all_coeffs()[::-1]))
    I = {k: sp.expand(IC[k].subs(sub)) for k in IC}
    W = 5 * b ** 2 - eps * nu
    Q = eps ** 2 * W ** 3 - 16 * nu ** 2
    R = eps ** 2 * W ** 3 - 8 * b * (8 * b ** 4 - 5 * b ** 2 * eps * nu + 5 * eps ** 2 * nu ** 2) + 16 * nu ** 2
    disp = {2: 50 * (7 * b ** 2 - 2 * eps * nu), 4: sp.Rational(625, 4) * (4 * b ** 2 + eps * nu) ** 2,
            6: -sp.Rational(625, 8) * (9 * b * eps ** 2 * W ** 3 - 3776 * b ** 6 - 40 * b ** 4 * eps * nu
                                       + 40 * b ** 2 * eps ** 2 * nu ** 2 + 144 * b * nu ** 2 + 75 * eps ** 3 * nu ** 3),
            10: sp.Rational(3125, 256) * R ** 2}
    print('I2, I4, I6, I10 of f as displayed:', [sp.expand(I[k] - disp[k]) == 0 for k in (2, 4, 6, 10)])
    g = -(4 * b ** 2 + eps * nu) / (30 * b ** 2)
    h = -R / (100000 * b ** 5)
    J = {2: 6 * (4 * g + 1), 4: 9 * g ** 2, 6: 9 * (4 * h + 9 * g ** 3 + 2 * g ** 2), 10: 4 * h ** 2}
    lam = 125 * b ** 2
    print('I_k == (125 b^2)^w J_k(g, h), w = 1, 2, 3, 5:',
          [sp.simplify(I[k] - lam ** (k // 2) * J[k]) == 0 for k in (2, 4, 6, 10)])
    Phi = 2 * (6250 * h ** 2 - 4500 * g ** 2 * h - 1350 * g * h - 108 * h - 972 * g ** 5 - 324 * g ** 4 - 27 * g ** 3)
    print('Phi(g, h) == 5 (Q/(2000 b^5))^2:', sp.simplify(Phi - 5 * (Q / (2000 * b ** 5)) ** 2) == 0)

    # normalisation check, independent of the family. LMFDB genus 2 curve 529.a.529.1: y^2 + (x^3 + x + 1) y = -x^5,
    # i.e. y^2 = x^6 - 4x^5 + 2x^4 + 2x^3 + x^2 + 2x + 1; the LMFDB lists End(J) = Z[(1 + sqrt5)/2] over Q and the
    # Igusa-Clebsch invariants (284, 2401, 246639, -67712)
    xl = sp.Poly(X ** 6 - 4 * X ** 5 + 2 * X ** 4 + 2 * X ** 3 + X ** 2 + 2 * X + 1, X).all_coeffs()[::-1]
    Il = {k: IC[k].subs(dict(zip(A, xl))) for k in IC}
    lm = {2: 284, 4: 2401, 6: 246639, 10: -67712}
    mu = Il[2] / lm[2]
    print('529.a.529.1: our invariants', [Il[k] for k in (2, 4, 6, 10)], '== mu^w * LMFDB values, mu =', mu, ':',
          [Il[k] == mu ** (k // 2) * lm[k] for k in (2, 4, 6, 10)])
    pts = ek_points(Il)
    print('529.a.529.1: all (g, h) on the parametrisation:', pts)
    for g1, h1 in pts:
        Phi1 = 2 * (6250 * h1 ** 2 - 4500 * g1 ** 2 * h1 - 1350 * g1 * h1 - 108 * h1 - 972 * g1 ** 5 - 324 * g1 ** 4
                    - 27 * g1 ** 3)
        print('  Phi =', Phi1, '| square in Q:', sp.sqrt(Phi1).is_rational, '| sqrt =', sp.sqrt(Phi1))
    print('control, 529.a.529.1 with I2 halved: all (g, h):', ek_points({**Il, 2: Il[2] / 2}))


def ek_points(I):
    """All (g, h) in Qbar^2 with (I2 : I4 : I6 : I10) = (J2 : J4 : J6 : J10)(g, h) in P(1, 2, 3, 5), I2 != 0:
    g from I4 J2^2 = I2^2 J4 (quadratic), then h from I6 J2^3 = I2^3 J6 (linear), then test I10 J2^5 = I2^5 J10."""
    gg, hh = sp.symbols('gg hh')
    J2, J4 = 6 * (4 * gg + 1), 9 * gg ** 2
    J6, J10 = 9 * (4 * hh + 9 * gg ** 3 + 2 * gg ** 2), 4 * hh ** 2
    out = []
    for g in sp.solve(I[4] * J2 ** 2 - I[2] ** 2 * J4, gg):
        for h in sp.solve((I[6] * J2 ** 3 - I[2] ** 3 * J6).subs(gg, g), hh):
            if sp.simplify((I[10] * J2 ** 5 - I[2] ** 5 * J10).subs({gg: g, hh: h})) == 0:
                out.append((g, h))
    return out


if __name__ == '__main__':
    main()
