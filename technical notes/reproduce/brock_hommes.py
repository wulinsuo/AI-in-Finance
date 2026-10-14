"""Reproduce every number quoted in technical notes/brock_hommes.tex
(The Brock-Hommes Adaptive Belief System: A Complete Mathematical Argument).

Run:  python "technical notes/reproduce/brock_hommes.py"   (from any directory)

Each check prints the computed value next to the value printed in the note,
compared at the note's printed precision. Exit status is 1 if any check fails.
"""
import math
import sys

# ---------------------------------------------------------------- helper ----
_n_checks, _fails = 0, []


def check(label, computed, quoted, dp=None, rel=None, tol=None):
    """dp: quoted value printed to dp decimals, passes if computed rounds to it.
    rel: relative tolerance. tol: absolute tolerance (Monte Carlo claims)."""
    global _n_checks
    _n_checks += 1
    if math.isinf(computed) or math.isinf(quoted):
        ok = computed == quoted
    elif dp is not None:
        ok = abs(computed - quoted) <= 0.5 * 10 ** (-dp) + 1e-12
    elif rel is not None:
        ok = abs(computed - quoted) <= rel * abs(quoted)
    else:
        ok = abs(computed - quoted) <= tol
    shown = f"{computed:.{dp + 3}f}" if dp is not None else f"{computed:.6g}"
    print(f"{'OK' if ok else 'MISMATCH':8s} {label}: computed {shown}, quoted {quoted}")
    if not ok:
        _fails.append(label)


def claim(label, ok, detail=""):
    global _n_checks
    _n_checks += 1
    print(f"{'OK' if ok else 'MISMATCH':8s} {label}{': ' + detail if detail else ''}")
    if not ok:
        _fails.append(label)


def finish(name):
    print(f"\nSUMMARY {name}: {_n_checks} checks, {len(_fails)} mismatches")
    sys.exit(1 if _fails else 0)


# ------------------------------------------------------------- the note ----
R, g, C = 1.01, 1.5, 0.05


def n2star(beta):                       # eq. nstar
    return 1.0 / (1.0 + math.exp(-beta * C))


def eig(beta):                          # Theorem stability: lambda = g n2*/R
    return g * n2star(beta) / R


check("2R (upper end of regime (ii))", 2 * R, 2.02, dp=2)
claim("R < g < 2R, so Corollary (ii) applies", R < g < 2 * R)
check("1/C", 1 / C, 20, dp=0)
check("R/(g-R)", R / (g - R), 2.0612, dp=4)
beta_star = math.log(R / (g - R)) / C   # eq. betastar
check("critical intensity beta*", beta_star, 14.47, dp=2)

rows = [  # beta, n2*, lambda  (Table brockhommes)
    (0, 0.5000, 0.7426),
    (5, 0.5622, 0.8349),
    (10, 0.6225, 0.9244),
    (beta_star, 0.6733, 1.0000),
    (18, 0.7109, 1.0559),
    (25, 0.7773, 1.1544),
]
for beta, nq, lq in rows:
    tag = "beta*" if beta == beta_star else f"{beta:g}"
    check(f"Table: n2*(beta={tag})", n2star(beta), nq, dp=4)
    check(f"Table: lambda(beta={tag})", eig(beta), lq, dp=4)
# Stability column: stable iff lambda < 1
for beta, want in [(0, True), (5, True), (10, True), (18, False), (25, False)]:
    claim(f"Table: steady state {'stable' if want else 'unstable'} at beta={beta}",
          (eig(beta) < 1) == want, f"lambda={eig(beta):.4f}")
claim("n2*(beta) rises monotonically from 0.5 towards (never reaching) 1",
      all(n2star(a) < n2star(b) < 1 for a, b in [(0, 5), (5, 10), (10, 18), (18, 25), (25, 200)]))
claim("eigenvalue crosses 1 exactly at beta*", abs(eig(beta_star) - 1) < 1e-12,
      f"lambda(beta*)={eig(beta_star):.12f}; at the printed 14.47: {eig(14.47):.5f}")
claim("C = 0 gives n2* = 1/2 for every beta (cost-asymmetry remark)",
      all(abs(1 / (1 + math.exp(-b * 0.0)) - 0.5) < 1e-15 for b in (0, 5, 50)))

finish("brock_hommes")
