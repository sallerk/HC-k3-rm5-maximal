Supporting code and data for the note
"An explicit eight-dimensional family of K3 surfaces with real multiplication by Q(sqrt5)"
==========================================================================================

Status: accompanies the note (Kevin Saller, 2026); not peer reviewed.

What the note proves: let nu be a binary sextic, b a binary cubic in (x0, x1) and eps a number with
5 b0^2 = eps nu0 (b0, nu0 the coefficients of x0^3, x0^6), and let
    F = [nu (x1 - eps x2) + x1^2 x2^5 - 5 b x1 x2^3 + 5 b^2 x2] / x1 + (eps/4) x2^6,
a ternary sextic. For eps = b0 = 0 this is the seven-dimensional family F = nu + x1 D5(x2; a) of van
Geemen and Schuett (D5(z; c) = z^5 - 5 c z^3 + 5 c^2 z).
- Theorem A: for very general (nu, b, eps), the double plane S: w^2 = F is a K3 surface with Picard
  number 2 (Gram matrix [[-2, 3], [3, -2]]) and real multiplication by Q(sqrt5), induced by an
  algebraic cycle on S x S; the family has eight moduli.
- Theorem B: the Hodge conjecture holds for S x S and for the Hilbert square S^[2].
The lines through P0 = (0:0:1) cut S in the genus-two curves y^2 = D5(z; b) + nu (1 - eps z) + (eps/4) z^6
(Lemma 2.1). Proposition 3.1 shows by explicit formulas that these curves lie on the Humbert surface H5,
using the equations of Elkies and Kumar for H5 and for the Hilbert modular surface Y_-(5), and that
their real multiplication is defined over the base of the pencil.

The proofs use three computations (Section 8 of the note), all done by the scripts here:
- the identities of Proposition 3.1, in exact rational arithmetic, with two independent
  implementations of the Igusa-Clebsch invariants (b8_exact.py + b9_certify.py + b15_note_formulas.py,
  and b16_ic_interp_q.py), and the check of the normalisation of the invariants with the LMFDB
  curve 529.a.529.1 in Section 3 (b16_ic_interp_q.py);
- the smoothness of the explicit member of Lemma 2.1(4) (b14_example.py);
- the rank 17 over Q of Lemma 5.1, and 16 for a member of the van Geemen-Schuett family
  (b14_example.py).
The other scripts are sanity checks (b6, b7) or record how the family was found (b1-b5); the proofs
do not use them.

Not checked by any script: the arguments of the note themselves (Corollary 3.2, Proposition 4.1,
the proofs of Theorems A and B, Remark 7.1) are proofs, not computations. The search b1-b5 was run
at one random member of the van Geemen-Schuett family over F_31981 only. The Legendre-symbol tests
in b6 and b7 are heuristic.

Licence: the code in code/ is under the MIT licence; the data in inputs/ and outputs/ and this text
are under CC BY 4.0 (see the LICENSE files of the repository). No third-party files are included.


1. Requirements
---------------
Python 3 with sympy, python-flint and numpy (most scripts import all three, directly or through
b2_humbert.py). Tested with Python 3.12.9, sympy 1.13.1, python-flint 0.9.0 and numpy 2.1.3. Every
script runs in a few seconds on a desktop PC.


2. How to run
-------------
Each script prints its results to the screen. The reference results are in outputs/ (the .txt files
there are the screen output; the .json files are the result files of b1, b2, b4). From the top folder
of this repository:

    mkdir rerun
    cd rerun

Computations used in the proofs (no arguments):

    python -I ../code/b8_exact.py          > b8_exact.txt
    python -I ../code/b9_certify.py        > b9_certify.txt
    python -I ../code/b15_note_formulas.py > b15_note_formulas.txt
    python -I ../code/b16_ic_interp_q.py   > b16_ic_interp_q.txt
    python -I ../code/b14_example.py       > b14_example.txt

Sanity checks (arguments: prime, seed):

    python -I ../code/b6_check_family.py 31981 0     > b6_s0_p31981.txt
    python -I ../code/b6_check_family.py 1000003 1   > b6_s1_p1000003.txt
    python -I ../code/b7_dim_control.py  1000003 0   > b7_s0_p1000003.txt

The search that found the family (prime 31981, seed 0; each step reads the files of the previous one):

    python -I ../code/b1_hodge.py b1hex_tan_s0.json 31981 0 hex 1 rmtan_s0_p31981.json
    python -I ../code/b2_humbert.py b2h_s0_p31981.json 31981 0 rmtan_s0_p31981.json
    python -I ../code/b4_jet.py b4_s0_eRM_N10.json 31981 0 rmtan_s0_p31981.json 10 eRM
    python -I ../code/b5_family.py b4_s0_eRM_N10.json 31981 0 > b5_s0_p31981.txt

b1_hodge.py writes rmtan_s0_p31981.json and, next to it, rmtan_s0_p31981_hex.json and
rmtan_s0_p31981_pim1.json; the reference copies of these three files are in inputs/. b3_arc.py is a
module used by b4 and b5 (it can also be run on its own; usage in its header).

Then compare the files in rerun/ with those of the same name in outputs/ and inputs/ (for instance with
"diff" or "fc"). They should be identical, except for the field "seconds" (the runtime) in
b4_s0_eRM_N10.json.


3. What each script checks
--------------------------
Each script has a header with the details.

b8_exact.py: integer polynomials for the Igusa-Clebsch invariants I2, I4, I6, I10 (as defined from
  the roots in Section 3 of the note) and for the implicit equation H5 of the Humbert surface in
  (I2 : I4 : I6 : I10), obtained by interpolation modulo five primes and Chinese remaindering and
  validated modulo a sixth prime; then, exactly in Q[b, nu, eps]: H5 vanishes on the invariants of
  f = D5(z; b) + nu (1 - eps z) + (eps/4) z^6; the Elkies-Kumar coordinates (g, h) of f; and
  Phi(g, h) (the right-hand side of the equation of Y_-(5)) is 5 times a square.
b9_certify.py: (a) the lifted equation H5 vanishes exactly on the Elkies-Kumar parametrisation of H5
  and is irreducible over Q; (b) the lifted invariants agree exactly with the root formulas on random
  sextics with integer roots; (c) the map (b, nu, eps) -> (g, h) has rank 2; (d) a random member of the
  family is smooth modulo a prime.
b15_note_formulas.py: every formula displayed in Lemma 2.1(1)-(2) and Proposition 3.1, exactly over Q
  (I2 = 6f3^2 - 16f2f4 + 40f1f5 - 240f0f6; I2, I4, I6, I10 of f; I_k = lambda^w J_k(g, h) with
  lambda = 125 b^2; Phi(g, h) = Q^2/(800000 b^10) = 5 (Q/(2000 b^5))^2; the expanded form of F and
  F(t, 1, z) = f_{b(t,1), nu(t,1), eps}(z)), using the invariants of b8/b9; and the identity
  F(x0, eps x2, x2) = (5/eps)(eps x2^3/2 - b(x0, eps x2))^2 used in the proof of Theorem A (the line
  x1 = eps x2 splits in the double cover).
b16_ic_interp_q.py: the second implementation: I2, I4, I6, I10 interpolated over Q from the root
  formulas (held-out samples; I10 equals the discriminant of the generic sextic), and the identities
  of Proposition 3.1 again, exactly over Q. Then the normalisation check with the LMFDB genus-two
  curve 529.a.529.1, y^2 = x^6 - 4x^5 + 2x^4 + 2x^3 + x^2 + 2x + 1, whose Jacobian has endomorphism
  ring Z[(1 + sqrt5)/2] over Q according to the LMFDB: its invariants agree with those listed in the
  LMFDB up to weighted scaling by -2; it lies on the Elkies-Kumar parametrisation exactly at one
  point, (g, h) = (-49/54, 23/243), with Phi(g, h) = (1658/81)^2; with I2 replaced by I2/2 there is
  no such point.
b14_example.py: the member eps = 5, nu = x0^6 + x0 x1^5 + x1^6, b = x0^3 + x1^3 of Lemma 2.1(4): the
  polynomial 4F, its smoothness (the Jacobian ring vanishes in degree 13 modulo 1000003 and 32003),
  the rank 17 over Q of Lemma 5.1, and the rank 16 for the member nu = x0^6 + x0 x1^5 + x1^6,
  a = x0^2 + x1^2 of the van Geemen-Schuett family.
b6_check_family.py: random members of the family modulo a prime: all fibres of the pencil through
  P0 lie on H5, and the Legendre symbol of Phi(g, h) is constant along the pencil.
b7_dim_control.py: the number of moduli at random points modulo a prime (7 for the van Geemen-Schuett
  family, 8 for the new family); and the Legendre symbol of Phi along random lines in H5 (mixed, as
  expected), a control for b6.
b1_hodge.py, b2_humbert.py, b3_arc.py, b4_jet.py, b5_family.py: the search. At a random member of
  the van Geemen-Schuett family over F_31981, b1 computes the first-order directions in which the
  part in H^2(S) (x) H^2(S) of the class of a plane on the fourfold F(x) + F(y) = 0 in P^5 stays a
  Hodge class (8 dimensions modulo GL_3, as expected for the locus with real multiplication), b2 the
  condition that every fibre of the pencil through P0 lies on H5 to first and
  second order, b3/b4 formal arcs along which this condition holds to higher order, and b5 checks
  that the arc found by b4 is an exact quadratic arc F0 + s w1 + s^2 w2 and prints its coefficients,
  from which the closed form of F was read off.
