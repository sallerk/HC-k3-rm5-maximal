"""(a) dimension in moduli (= rank[d F/d params | gl3 . F] - 9) of
      vGS:  F = p + x1 D5(x2; a)                                   (p binary sextic, a binary quadric)
      new:  F = p (1 - eps x2/x1) + x1 D5(x2; Bc/x1) + (eps/4) x2^6, Bc binary cubic, p6 = 5 beta^2/eps
    at random rational points, ranks mod a large prime. vGS: 7 expected (Thm 1.1(5)); new: 8 if maximal.
(b) negative control for the Y_-(5) Legendre test in b6: random lines (g, h) = (g0 + g1 t, h0 + h1 t) in H5;
    the symbol of z^2 should be mixed (z^2 is not a square along a general line).
Usage: python -I b7_dim_control.py prime seed
"""
import random
import sys

import flint
import sympy as sp


def main():
    p, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(9000 + seed)
    x0, x1, x2 = sp.symbols('x0 x1 x2')
    P = sp.symbols('p0:7')            # p = sum P[k] x0^(6-k) x1^k
    A = sp.symbols('a0:3')
    Bs = sp.symbols('b0:4')           # Bc = sum Bs[k] x0^(3-k) x1^k, beta = Bs[0]
    eps = sp.Symbol('eps')
    D5 = lambda z, b: z ** 5 - 5 * b * z ** 3 + 5 * b ** 2 * z
    pb = sum(P[k] * x0 ** (6 - k) * x1 ** k for k in range(7))
    ab = sum(A[k] * x0 ** (2 - k) * x1 ** k for k in range(3))
    Bc = sum(Bs[k] * x0 ** (3 - k) * x1 ** k for k in range(4))
    Fv = sp.expand(pb + x1 * D5(x2, ab))
    num = sp.expand((pb * (x1 - eps * x2) + x1 ** 2 * D5(x2, Bc / x1)) * x1 ** 4)   # clear 1/x1^4 from D5
    num = sp.expand(num.subs(P[0], 5 * Bs[0] ** 2 / eps) * eps)
    q, r = sp.div(sp.Poly(num, x0, x1, x2), sp.Poly(x1 ** 5 * eps, x0, x1, x2))
    assert r.is_zero, 'new family not polynomial'
    Fn = sp.expand(q.as_expr() + eps / 4 * x2 ** 6)
    mons = sorted(sp.Poly(sp.expand((x0 + x1 + x2) ** 6), x0, x1, x2).monoms())

    def coeffvec(expr):
        d = sp.Poly(expr, x0, x1, x2).as_dict()
        return [d.get(m, 0) for m in mons]

    def rank_mod(F, params):
        sub = {v: sp.Rational(rng.randrange(1, 10 ** 6), rng.randrange(1, 10 ** 3)) for v in params}
        cols = [coeffvec(sp.diff(F, v).subs(sub)) for v in params]
        F0 = F.subs(sub)
        X = (x0, x1, x2)
        cols += [coeffvec(sp.expand(X[i] * sp.diff(F0, X[j]))) for i in range(3) for j in range(3)]
        red = lambda c: int(sp.Rational(c).p % p) * pow(int(sp.Rational(c).q % p), p - 2, p) % p
        M = flint.nmod_mat(len(mons), len(cols), [red(cols[j][i]) for i in range(len(mons)) for j in range(len(cols))], p)
        return M.rank(), len(params)

    rv, nv = rank_mod(Fv, list(P) + list(A))
    rn, nn = rank_mod(Fn, list(P[1:]) + list(Bs) + [eps])
    print('(a) vGS: %d params, rank %d -> dim in moduli %d' % (nv, rv, rv - 9))
    print('(a) new: %d params, rank %d -> dim in moduli %d' % (nn, rn, rn - 9))
    print('    new F at x1 = 0:', sp.factor(Fn.subs(x1, 0)))
    leg = lambda a: 0 if a % p == 0 else (1 if pow(a % p, (p - 1) // 2, p) == 1 else -1)
    zsq = lambda g, h: 2 * (6250 * h * h - 4500 * g * g * h - 1350 * g * h - 108 * h - 972 * g ** 5 - 324 * g ** 4
                           - 27 * g ** 3) % p
    for trial in range(4):
        g0, g1, h0, h1 = (rng.randrange(p) for _ in range(4))
        s = [leg(zsq(g0 + g1 * t, h0 + h1 * t)) for t in (rng.randrange(p) for _ in range(40))]
        print('(b) random line %d in H5: Legendre(z^2) counts +1: %d, -1: %d' % (trial, s.count(1), s.count(-1)))


if __name__ == '__main__':
    main()
