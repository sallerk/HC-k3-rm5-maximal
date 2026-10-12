"""Exact check of every formula displayed in the note (sympy over Q), using the integer Igusa-Clebsch polynomials
lifted in b8/b9 (root convention: I2 = f6^2 sum_15 ..., I10 = f6^10 prod (ri - rj)^2).
Checks: expanded form of F; fibre identity; I2..I10 of f_{b,nu,eps}; I_k = lambda^k J_k(g, h) with lambda = 125 b^2;
g, h; Phi(g, h) = Q^2/(800000 b^10) = 5 (Q/(2000 b^5))^2 with Q = eps^2 (5b^2 - eps nu)^3 - 16 nu^2;
R = eps^2 (5b^2 - eps nu)^3 - 8b(8b^4 - 5b^2 eps nu + 5 eps^2 nu^2) + 16 nu^2; I2 = 6f3^2 - 16f2f4 + 40f1f5 - 240f0f6.
Usage: python -I b15_note_formulas.py
"""
import os
import sys

import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b9_certify as C


def main():
    IC, H5 = C.lift()
    b, nu, eps = sp.symbols('b nu eps')
    f = [nu, 5 * b ** 2 - eps * nu, 0, -5 * b, 0, 1, eps / 4]
    I = []
    for d in (2, 4, 6, 10):
        I.append(sp.expand(sum(v * sp.Mul(*[f[k] ** e[k] for k in range(7)]) for e, v in IC.items() if sum(e) == d)))
    a = sp.symbols('a0:7')
    I2gen = sp.expand(sum(v * sp.Mul(*[a[k] ** e[k] for k in range(7)]) for e, v in IC.items() if sum(e) == 2))
    print('I2 = 6f3^2 - 16f2f4 + 40f1f5 - 240f0f6:',
          sp.expand(I2gen - (6 * a[3] ** 2 - 16 * a[2] * a[4] + 40 * a[1] * a[5] - 240 * a[0] * a[6])) == 0)
    W = 5 * b ** 2 - eps * nu
    Q = eps ** 2 * W ** 3 - 16 * nu ** 2
    R = eps ** 2 * W ** 3 - 8 * b * (8 * b ** 4 - 5 * b ** 2 * eps * nu + 5 * eps ** 2 * nu ** 2) + 16 * nu ** 2
    I6disp = -sp.Rational(625, 8) * (9 * b * eps ** 2 * W ** 3 - 3776 * b ** 6 - 40 * b ** 4 * eps * nu
                                    + 40 * b ** 2 * eps ** 2 * nu ** 2 + 144 * b * nu ** 2 + 75 * eps ** 3 * nu ** 3)
    disp = [50 * (7 * b ** 2 - 2 * eps * nu), sp.Rational(625, 4) * (4 * b ** 2 + eps * nu) ** 2, I6disp,
            sp.Rational(3125, 256) * R ** 2]
    print('displayed I2, I4, I6, I10 correct:', [sp.expand(I[i] - disp[i]) == 0 for i in range(4)])
    g = -(4 * b ** 2 + eps * nu) / (30 * b ** 2)
    h = -R / (100000 * b ** 5)
    J = [6 * (4 * g + 1), 9 * g ** 2, 9 * (4 * h + 9 * g ** 3 + 2 * g ** 2), 4 * h ** 2]
    lam = 125 * b ** 2
    print('I_k = lambda^w J_k (w = 1, 2, 3, 5):',
          [sp.simplify(I[i] - lam ** w * J[i]) == 0 for i, w in enumerate((1, 2, 3, 5))])
    Phi = 2 * (6250 * h ** 2 - 4500 * g ** 2 * h - 1350 * g * h - 108 * h - 972 * g ** 5 - 324 * g ** 4 - 27 * g ** 3)
    print('Phi = Q^2/(800000 b^10):', sp.simplify(Phi - Q ** 2 / (800000 * b ** 10)) == 0,
          '| = 5 (Q/(2000 b^5))^2:', sp.simplify(Phi - 5 * (Q / (2000 * b ** 5)) ** 2) == 0)
    print('H5(I) == 0:', sp.expand(sum(c * I[0] ** p_ * I[1] ** q * I[2] ** r * I[3] ** s
                                       for (p_, q, r, s), c in H5.items())) == 0)
    # expanded form of F and fibre identity
    x0, x1, x2, t, z = sp.symbols('x0 x1 x2 t z')
    nk = sp.symbols('n0:7')
    bk = sp.symbols('c0:4')
    nub = sum(nk[k] * x0 ** (6 - k) * x1 ** k for k in range(7))
    bb = sum(bk[k] * x0 ** (3 - k) * x1 ** k for k in range(4))
    D5 = lambda w, c: w ** 5 - 5 * c * w ** 3 + 5 * c ** 2 * w
    nup = sp.expand((nub - nk[0] * x0 ** 6) / x1)
    bp = sp.expand((bb - bk[0] * x0 ** 3) / x1)
    Fexp = nub - eps * nup * x2 - 5 * bb * x2 ** 3 + 10 * bk[0] * x0 ** 3 * bp * x2 + 5 * x1 * bp ** 2 * x2 \
        + x1 * x2 ** 5 + eps / 4 * x2 ** 6
    Fdef = (nub * (x1 - eps * x2) + x1 ** 2 * x2 ** 5 - 5 * bb * x1 * x2 ** 3 + 5 * bb ** 2 * x2) / x1 + eps / 4 * x2 ** 6
    con = {nk[0]: 5 * bk[0] ** 2 / eps}
    print('expanded F == definition under 5 beta^2 = eps nu0:',
          sp.simplify((Fexp - Fdef).subs(con)) == 0)
    fib = sp.expand(Fexp.subs(con).subs({x0: t, x1: 1, x2: z}))
    fam = sp.expand((D5(z, bb) + nub * (1 - eps * z) + eps / 4 * z ** 6).subs(con).subs({x0: t, x1: 1}))
    print('F(t, 1, z) == f_{b(t,1), nu(t,1), eps}(z):', sp.simplify(fib - fam) == 0)
    # splitting line x1 = eps x2: F restricted to it is (5/eps) (eps x2^3/2 - b(x0, eps x2))^2
    lin = sp.expand(Fexp.subs(con).subs(x1, eps * x2))
    sq = 5 / eps * (eps * x2 ** 3 / 2 - bb.subs(x1, eps * x2)) ** 2
    print('F(x0, eps x2, x2) == (5/eps)(eps x2^3/2 - b(x0, eps x2))^2:', sp.simplify(lin - sq) == 0)


if __name__ == '__main__':
    main()
