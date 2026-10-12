"""Explicit member for the note: eps = 5, beta = nu6 = 1 (5 beta^2 = eps nu6). Try small integer nu, b; report the
first with (i) 4F in Z[x0,x1,x2] smooth (Jacobian ring R_13 = 0 mod two primes => smooth over Q), (ii) exact rank over
Q of [dF/d(11 parameters) | gl_3 . F] equal to 17 (=> 8 moduli), (iii) the fibre restriction identity, and (iv) the
vGS control at a member (eps = beta = 0): rank 16.
Usage: python -I b14_example.py
"""
import itertools
import os
import sys

import flint
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b2_humbert as B

x0, x1, x2 = xs = sp.symbols('x0 x1 x2')
P = sp.symbols('p0:7')
Bs = sp.symbols('b0:4')
eps = sp.Symbol('eps')
t, z = sp.symbols('t z')
D5 = lambda w, c: w ** 5 - 5 * c * w ** 3 + 5 * c ** 2 * w


def family():
    pb = sum(P[k] * x0 ** (6 - k) * x1 ** k for k in range(7))
    Bc = sum(Bs[k] * x0 ** (3 - k) * x1 ** k for k in range(4))
    num = sp.expand((pb * (x1 - eps * x2) + x1 ** 2 * D5(x2, Bc / x1)) * x1 ** 4)
    num = sp.expand(num.subs(P[0], 5 * Bs[0] ** 2 / eps) * eps)
    q, r = sp.div(sp.Poly(num, *xs), sp.Poly(x1 ** 5 * eps, *xs))
    assert r.is_zero
    return sp.expand(q.as_expr() + eps / 4 * x2 ** 6), pb, Bc


def smooth_mod(F, p):
    Fi = sp.Poly(sp.expand(4 * F), *xs)
    ctx = flint.nmod_mpoly_ctx.get(('x0', 'x1', 'x2'), modulus=p)
    g = ctx.from_dict({tuple(m): int(c) % p for m, c in Fi.as_dict().items()})
    m13, m8 = B.mons(3, 13), B.mons(3, 8)
    idx = {m: i for i, m in enumerate(m13)}
    X = ctx.gens()
    rows = []
    for dv in (g.derivative(0), g.derivative(1), g.derivative(2)):
        for m in m8:
            pr = dv * X[0] ** m[0] * X[1] ** m[1] * X[2] ** m[2]
            row = [0] * len(m13)
            for ex, cf in pr.to_dict().items():
                row[idx[tuple(int(u) for u in ex)]] = int(cf)
            rows.append(row)
    return len(m13) - flint.nmod_mat(len(rows), len(m13), [x for r_ in rows for x in r_], p).rank() == 0


def rank_q(F, params, sub):
    mons = sorted(sp.Poly(sp.expand((x0 + x1 + x2) ** 6), *xs).monoms())
    cv = lambda e: [sp.Poly(e, *xs).as_dict().get(m, 0) for m in mons]
    cols = [cv(sp.expand(sp.diff(F, v).subs(sub))) for v in params]
    F0 = sp.expand(F.subs(sub))
    cols += [cv(sp.expand(xs[i] * sp.diff(F0, xs[j]))) for i in range(3) for j in range(3)]
    return sp.Matrix(cols).T.rank()


def main():
    F, pb, Bc = family()
    params = list(P[1:]) + list(Bs) + [eps]
    for nu_low, b_low in itertools.product([(0, 0, 0, 0, 1, 1), (0, 1, 0, 0, 0, 1), (1, 0, 0, 0, 1, 2),
                                            (0, 0, 1, 0, 0, 1)], [(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]):
        sub = {eps: 5, Bs[0]: 1}
        sub.update({P[k + 1]: nu_low[k] for k in range(6)})
        sub.update({Bs[k + 1]: b_low[k] for k in range(3)})
        Fx = sp.expand(F.subs(sub))
        if not (smooth_mod(Fx, 1000003) and smooth_mod(Fx, 32003)):
            continue
        rk = rank_q(F, params, sub)
        nu_e = sp.expand(pb.subs(P[0], 1).subs(sub))
        b_e = sp.expand(Bc.subs(sub))
        fib = sp.expand(Fx.subs({x0: t, x1: 1, x2: z}))
        fam = sp.expand(D5(z, b_e.subs({x0: t, x1: 1})) + nu_e.subs({x0: t, x1: 1}) * (1 - 5 * z) + sp.Rational(5, 4) * z ** 6)
        print('nu =', nu_e, '| b =', b_e, '| eps = 5')
        print('4F =', sp.factor(4 * Fx))
        print('smooth (R_13 = 0 mod 1000003 and 32003): True | rank over Q:', rk, '-> moduli', rk - 9,
              '| fibre identity:', sp.expand(fib - fam) == 0)
        break
    # vGS control at an explicit member: nu as above, a = x0^2 + x1^2 (b = x1 a, eps = 0)
    pbv = x0 ** 6 + x1 ** 6 + x0 * x1 ** 5
    A = sp.symbols('a0:3')
    av = sum(A[k] * x0 ** (2 - k) * x1 ** k for k in range(3))
    Fv = sp.expand(pb + x1 * D5(x2, av))
    subv = {P[0]: 1, P[1]: 0, P[2]: 0, P[3]: 0, P[4]: 0, P[5]: 1, P[6]: 1, A[0]: 1, A[1]: 0, A[2]: 1}
    print('vGS control rank over Q:', rank_q(Fv, list(P) + list(A), subv), '(expect 16)')


if __name__ == '__main__':
    main()
