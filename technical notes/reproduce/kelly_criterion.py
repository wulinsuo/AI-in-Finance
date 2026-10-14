"""Reproduce every number quoted in technical notes/kelly_criterion.tex
(The Kelly Criterion: A Complete Mathematical Argument).

Run:  python "technical notes/reproduce/kelly_criterion.py"   (from any directory)

Each check prints the computed value next to the value printed in the note,
compared at the note's printed precision. Exit status is 1 if any check fails.
"""
import math
import sys

from scipy.optimize import brentq

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
p, b = 0.6, 1.0
q = 1 - p


def g_ell(f, ell):
    """Growth rate p ln(1+bf) + q ln(1 - ell f); -inf at ruin (eq. partial-g)."""
    loss = 1 - ell * f
    return -math.inf if loss <= 1e-12 else p * math.log(1 + b * f) + q * math.log(loss)


def g(f):                                    # eq. gdef
    return g_ell(f, 1.0)


# --- Section: Worked Numerical Example (full-loss bet) ---
check("edge E[R] = pb - q", p * b - q, 0.2, dp=1)
fstar = p - q / b                            # eq. kellyformula
check("Kelly fraction f* = p - q/b", fstar, 0.2, dp=1)
for f, qv in [(0.0, 0.0000), (0.1, 0.0150), (0.2, 0.0201), (0.3, 0.0147), (0.4, -0.0024),
              (0.5, -0.0340), (0.6, -0.0845), (0.8, -0.2911), (1.0, -math.inf)]:
    check(f"Table growth: g({f})", g(f), qv, dp=4)
check("maximum growth g(f*)", g(fstar), 0.0201, dp=4)
root = brentq(g, 0.25, 0.99)
check("positive root of g (g turns negative)", root, 0.3894, dp=4)
claim("root lies between f=0.3 and f=0.4", 0.3 < root < 0.4)
check("'roughly 39%' of wealth", 100 * root, 39, dp=0)
check("g(0.5)", g(0.5), -0.0340, dp=4)

# --- Section: Partial losses ---
ell = 0.5
f05 = p / ell - q / b                         # eq. kelly-partial-formula
check("p/ell for ell=0.5", p / ell, 1.2, dp=1)
check("f*_0.5 = p/ell - q/b", f05, 0.8, dp=1)
check("f*_0.5 / f* ('four times')", f05 / fstar, 4, dp=0)
thr = p * b / (b + q)                         # eq. leverage-threshold
check("leverage threshold pb/(b+q)", thr, 0.4286, dp=4)
claim("ell = 0.3 lies below the threshold", 0.3 < thr)
f03 = p / 0.3 - q / b
check("f*_0.3 = 0.6/0.3 - 0.4", f03, 1.6, dp=1)
tab = [  # f, g_0.5, g_0.3   (Table partial-growth); None = '---'
    (0.0, 0.0000, 0.0000), (0.4, 0.1126, 0.1507), (0.8, 0.1483, 0.2429),
    (1.0, 0.1386, 0.2732), (1.2, 0.1066, 0.2946), (1.6, -0.0705, 0.3117),
    (2.0, -math.inf, 0.2927), (3.0, None, -0.0893), (10 / 3, None, -math.inf),
]
for f, a, c in tab:
    if a is not None:
        check(f"Table partial: g_0.5({f:.4g})", g_ell(f, 0.5), a, dp=4)
    else:
        claim(f"Table partial: g_0.5({f:.4g}) shown as '---' (beyond ruin at f=2)",
              g_ell(f, 0.5) == -math.inf)
    check(f"Table partial: g_0.3({f:.4g})", g_ell(f, 0.3), c, dp=4)
check("ruin point 1/ell for ell=0.5", 1 / 0.5, 2.0, dp=1)
check("ruin point 1/ell for ell=0.3 (3.333...)", 1 / 0.3, 10 / 3, dp=3)
check("g_1.0(f*)", g(fstar), 0.0201, dp=4)
check("g_0.5(f*_0.5)", g_ell(f05, 0.5), 0.1483, dp=4)
check("g_0.3(f*_0.3)", g_ell(f03, 0.3), 0.3117, dp=4)
check("g_0.5(1.6)", g_ell(1.6, 0.5), -0.0705, dp=4)
claim("f*_0.5 and f*_0.3 are the maximisers of their growth curves",
      all(g_ell(f05, 0.5) >= g_ell(f05 + h, 0.5) and g_ell(f03, 0.3) >= g_ell(f03 + h, 0.3)
          for h in (-0.01, 0.01)))

# --- Section: Fractional Kelly ---
check("g(0.1) (5 dp)", g(0.1), 0.01504, dp=5)
check("g(0.2) (5 dp)", g(0.2), 0.02014, dp=5)
ratio = g(0.1) / g(0.2)
check("half-Kelly ratio g(0.1)/g(0.2)", ratio, 0.747, dp=3)
check("half-Kelly share of maximum growth, percent", 100 * ratio, 74.7, dp=1)

finish("kelly_criterion")
