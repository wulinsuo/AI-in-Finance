"""Reproduce every number quoted in technical notes/kaplan_meier.tex
(The Kaplan-Meier Estimator: A Complete Derivation).

Run:  python "technical notes/reproduce/kaplan_meier.py"   (from any directory)

Each check prints the computed value next to the value printed in the note,
compared at the note's printed precision. Exit status is 1 if any check fails.
"""
import math
import sys

from scipy.stats import norm

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
# Ten loan accounts, months to default; event=0 marks censoring ("+").
data = [(5, 1), (8, 0), (12, 1), (12, 1), (18, 0), (20, 1), (24, 0), (30, 1), (30, 0), (36, 0)]

event_times = sorted({y for y, e in data if e == 1})
check("number of distinct default times m", len(event_times), 4, dp=0)
claim("distinct default times are (5, 12, 20, 30)", event_times == [5, 12, 20, 30])

# Risk set: subjects with y >= t (a censoring tied with an event counts as at risk).
quoted = {  # t: (n, d, 1-d/n, S, Greenwood term)
    5: (10, 1, 0.9000, 0.9000, 0.01111),
    12: (8, 2, 0.7500, 0.6750, 0.04167),
    20: (5, 1, 0.8000, 0.5400, 0.05000),
    30: (3, 1, 0.6667, 0.3600, 0.16667),
}
S, gw_sum, gw, S_at = 1.0, 0.0, {}, {}
for t in event_times:
    n = sum(1 for y, _ in data if y >= t)
    d = sum(1 for y, e in data if y == t and e == 1)
    S *= 1 - d / n                                  # eq. km
    term = d / (n * (n - d))                        # eq. greenwood summand
    gw[t] = term
    gw_sum += term
    S_at[t] = S
    qn, qd, qf, qS, qg = quoted[t]
    check(f"Table: n_i at t={t}", n, qn, dp=0)
    check(f"Table: d_i at t={t}", d, qd, dp=0)
    check(f"Table: 1-d/n at t={t}", 1 - d / n, qf, dp=4)
    check(f"Table: S_hat({t})", S, qS, dp=4)
    check(f"Table: Greenwood term at t={t}", term, qg, dp=5)

claim("only the account censored at month 36 remains after t=30",
      sum(1 for y, _ in data if y > 30) == 1)
S30 = S
check("S_hat(30)", S30, 0.360, dp=3)
check("percent default-free beyond 30 months", 100 * S30, 36, dp=0)
check("S_hat(30)^2", S30**2, 0.1296, dp=4)
check("sum of Greenwood terms", gw_sum, 0.26944, dp=5)
var30 = S30**2 * gw_sum
check("Greenwood variance of S_hat(30)", var30, 0.0349, dp=4)
se30 = math.sqrt(var30)
check("standard error of S_hat(30)", se30, 0.187, dp=3)
z = norm.ppf(0.975)
lo, hi = S30 - z * se30, S30 + z * se30
claim("95% CI lower limit is below 0 (hence truncated to 0)", lo < 0, f"raw lower {lo:.4f}")
check("95% CI upper limit", hi, 0.73, dp=2)
check("Greenwood term, final risk set (n=3)", gw[30], 0.1667, dp=4)
check("Greenwood term, first risk set (n=10)", gw[5], 0.0111, dp=4)
claim("final term is more than an order of magnitude larger than the first",
      gw[30] / gw[5] > 10, f"ratio {gw[30] / gw[5]:.1f}")
check("S_hat(5)", S_at[5], 0.900, dp=3)

finish("kaplan_meier")
