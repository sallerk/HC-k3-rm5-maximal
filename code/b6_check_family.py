"""Checks on the explicit family found by b4/b5 (seed 0 at p = 31981; INFERRED closed form):
  fibres  y^2 = D5(z; b) + nu (1 - eps z) + (eps/4) z^6
  plane   F = p (1 - eps x2/x1) + x1 D5(x2; Bc/x1) + (eps/4) x2^6,  Bc binary cubic with 5 beta^2 = eps p6
          (beta = x0^3 coefficient of Bc, p6 = x0^6 coefficient of p; this makes F a polynomial).
(1) fibre family on H5 for random (b, nu, eps)?
(2) plane family: G(F) == 0 in t for random (p, Bc) and eps = 5 beta^2 / p6?
(3) monodromy of the RM choice along the pencil: Elkies-Kumar Thm 16, Y_-(5): z^2 = 2(6250h^2 - 4500g^2h - 1350gh
    - 108h - 972g^5 - 324g^4 - 27g^3) over H5 = P^2_{g,h} (Cor. 15). Along the pencil, the RM choice is global iff
    z^2(g(t), h(t)) is a square in kbar(t): test the Legendre symbol of z^2 at many t (constant => square up to a
    constant; mixed => monodromy swaps the two RM actions => no global RM). Controls: vGS (eps = 0).
Usage: python -I b6_check_family.py prime seed
"""
import os
import random
import sys

import flint
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b2_humbert as B


def D5coef(b, p):
    """coefficients f_0..f_5 of D5(z; b) = z^5 - 5 b z^3 + 5 b^2 z"""
    return [0, 5 * b * b % p, 0, (-5 * b) % p, 0, 1]


def main():
    p, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(7000 + seed)
    IC = B.fit_ic(p, rng)
    W, H5 = B.h5_fit(p, rng)
    inv = lambda a: pow(a % p, p - 2, p)
    # (1) fibre family
    vals = []
    for _ in range(8):
        b, nu, eps = rng.randrange(p), rng.randrange(1, p), rng.randrange(1, p)
        f = D5coef(b, p) + [0]
        f[0] = (f[0] + nu) % p
        f[1] = (f[1] - nu * eps) % p
        f[6] = eps * inv(4) % p
        vals.append(B.h5_eval(H5, B.ic_eval(IC, f, p), p))
    print('(1) H5 on fibre family (8 random b, nu, eps):', vals)
    # (2) plane family with random p, Bc
    ctx = flint.nmod_mpoly_ctx.get(('x0', 'x1', 'x2'), modulus=p)
    x0, x1, x2 = ctx.gens()

    def plane(pc, bc, eps):
        pb = sum(pc[k] * x0 ** (6 - k) * x1 ** k for k in range(7))
        Bc = sum(bc[k] * x0 ** (3 - k) * x1 ** k for k in range(4))
        num = pb * (x1 - eps * x2) + x1 * x1 * x2 ** 5 - 5 * Bc * x1 * x2 ** 3 + 5 * Bc * Bc * x2
        Fq, r = divmod(num, x1)
        assert r == 0, 'not divisible by x1'
        return Fq + eps * inv(4) * x2 ** 6

    def fibre(F, t):
        fd = [0] * 7
        for ex, cf in F.to_dict().items():
            e0, e1, e2 = (int(u) for u in ex)
            fd[e0 + e1] = (fd[e0 + e1] + int(cf) * pow(t, e0, p)) % p
        return fd

    def gh_of(I):
        """(g, h) in F_p with I ~ (6(4g+1), 9g^2, 9(4h+9g^3+2g^2), 4h^2), or None"""
        I2, I4, I6, I10 = I
        u = I4 * inv(I2 * I2) % p           # = g^2 / (4 (4g+1)^2)
        # g^2 - 4u (4g+1)^2 = 0  ->  (1 - 64u) g^2 - 32u g - 4u = 0
        P = flint.nmod_poly([(-4 * u) % p, (-32 * u) % p, (1 - 64 * u) % p], p)
        out = []
        for fct, m in P.factor()[1]:
            if fct.degree() != 1:
                continue
            g = int(-fct[0]) * inv(int(fct[1])) % p
            lam = I2 * inv(6 * (4 * g + 1)) % p
            h = (I6 * inv(9 * pow(lam, 3, p)) - 9 * g ** 3 - 2 * g * g) * inv(4) % p
            if (4 * h * h * pow(lam, 5, p) - I10) % p == 0 and (9 * g * g * lam * lam - I4) % p == 0:
                out.append((g, h))
        return out

    def zsq(g, h):
        return 2 * (6250 * h * h - 4500 * g * g * h - 1350 * g * h - 108 * h - 972 * g ** 5 - 324 * g ** 4
                    - 27 * g ** 3) % p

    leg = lambda a: 0 if a % p == 0 else (1 if pow(a % p, (p - 1) // 2, p) == 1 else -1)
    for trial in range(3):
        pc = [rng.randrange(1, p) for _ in range(7)]
        bc = [rng.randrange(p) for _ in range(4)]
        for label in ('vGS (eps=0, beta=0)', 'new (eps = 5 beta^2/p6)'):
            if label.startswith('vGS'):
                bcc, eps = [0] + bc[1:], 0
            else:
                bcc = bc[:]
                eps = 5 * bcc[0] * bcc[0] * inv(pc[0]) % p
            F = plane(pc, bcc, eps)
            Gz, syms, nogh = [], [], 0
            for _ in range(40):
                t = rng.randrange(1, p)
                f = fibre(F, t)
                I = B.ic_eval(IC, f, p)
                if I[3] == 0 or I[0] == 0:
                    continue
                Gz.append(B.h5_eval(H5, I, p))
                gh = gh_of(I)
                if not gh:
                    nogh += 1
                    continue
                syms.append(tuple(sorted(set(leg(zsq(g, h)) for g, h in gh))))
            print('(2/3) trial %d %-26s H5 on 40 fibres: %s | (g,h) not found: %d | Legendre(z^2) sets: %s'
                  % (trial, label, 'all 0' if not any(Gz) else Gz[:5], nogh, sorted(set(syms))))


if __name__ == '__main__':
    main()
