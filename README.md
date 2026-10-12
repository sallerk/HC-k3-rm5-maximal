# HC-k3-rm5-maximal

Code and data supporting the note *An explicit eight-dimensional family of K3 surfaces with real multiplication by
Q(√5)* (Kevin Saller, 2026).

**Result.** The note gives an explicit family of double planes w^2 = F(x_0,x_1,x_2), branched along plane sextics that
depend on a binary sextic ν, a binary cubic b and a number ε subject to one quadratic relation. Its very general member
S is a K3 surface of degree two with Picard number 2 and real multiplication by Q(√5) on its transcendental lattice,
and the family has eight moduli, the maximal number for real multiplication by a real quadratic field. It contains the
seven-dimensional family of van Geemen and Schütt (the members with ε = b_0 = 0). The real multiplication comes from a pencil of
genus-two curves on S whose Jacobians have real multiplication by the ring of integers of Q(√5), defined over the base
of the pencil; the note shows that such a pencil gives an algebraic cycle on S × S inducing the real multiplication.
It follows that the Hodge conjecture holds for S × S and for the Hilbert square S^[2].

The computations used in the proofs are exact identities over Q (the Igusa–Clebsch invariants of the curves of the
pencil and the equations of Elkies and Kumar for the Humbert surface of discriminant 5), an exact rank over Q (eight
moduli) and the smoothness of an explicit member. [`README.txt`](README.txt) gives the requirements, the rerun
instructions and what each script checks.

**Status:** not yet peer reviewed.

**Disclosure:** the arguments and computations were produced with the assistance of an AI system (Claude, by
Anthropic). They were checked with separately written code and in a separate internal review, which were also carried
out with AI assistance. What was not checked is listed in README.txt.

## Licences

- **Code** (`code/`): MIT License, see [`LICENSE`](LICENSE).
- **Everything else** (README files, data in `inputs/` and computed outputs in `outputs/`): Creative Commons
  Attribution 4.0 International (CC BY 4.0), see [`LICENSE-CC-BY-4.0`](LICENSE-CC-BY-4.0).
- No third-party files are included.
