"""Reproduce every number quoted in technical notes/roll_spread_estimator.tex
(Roll's (1984) Implied Spread Estimator: A Complete Derivation).

Run:  python "technical notes/reproduce/roll_spread_estimator.py"   (from any directory)

Each check prints the computed value next to the value printed in the note,
compared at the note's printed precision. Exit status is 1 if any check fails.
"""
import math
import sys

import numpy as np

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
s, var_eps = 0.10, 0.0009

check("fundamental s.d. per trade, cents", 100 * math.sqrt(var_eps), 3, dp=0)
check("true Var(dP) = sigma_eps^2 + s^2/2 (eq. var)", var_eps + s**2 / 2, 0.0059, dp=4)
check("true Cov(dP_t, dP_t-1) = -s^2/4 (eq. cov1)", -s**2 / 4, -0.0025, dp=4)

# Table sim: eps_t, Q_t as printed; m_t, P_t and dP_t are derived from them.
eps = [None, 0.02, -0.01, 0.03, -0.02, 0.01, -0.03, 0.02, -0.01, 0.02, -0.02]
Q = [+1, -1, -1, +1, +1, -1, +1, -1, +1, -1, +1]
m_q = [50.00, 50.02, 50.01, 50.04, 50.02, 50.03, 50.00, 50.02, 50.01, 50.03, 50.01]
P_q = [50.05, 49.97, 49.96, 50.09, 50.07, 49.98, 50.05, 49.97, 50.06, 49.98, 50.06]
dP_q = [None, -0.08, -0.01, 0.13, -0.02, -0.09, 0.07, -0.08, 0.09, -0.08, 0.08]
m = [50.00]
for e in eps[1:]:
    m.append(m[-1] + e)                              # Assumption: random walk
P = [mi + s / 2 * qi for mi, qi in zip(m, Q)]         # Assumption: bid-ask bounce
dP = np.diff(P)
for t in range(11):
    check(f"Table sim: m_{t}", m[t], m_q[t], dp=2)
    check(f"Table sim: P_{t}", P[t], P_q[t], dp=2)
    if t:
        check(f"Table sim: dP_{t}", dP[t - 1], dP_q[t], dp=2)

n = len(dP)
mean = dP.mean()
check("sample mean of dP", mean, 0.001, dp=3)
check("sample variance of dP (n-1 divisor)", dP.var(ddof=1), 0.00712, dp=5)
dev = dP - mean
pairs = n - 1
check("number of consecutive pairs", pairs, 9, dp=0)
gamma1 = np.sum(dev[1:] * dev[:-1]) / (n - 1)       # eq. sampleestimator
check("sample first-order autocovariance gamma_1", gamma1, -0.003779, dp=6)
check("sqrt(-gamma_1)", math.sqrt(-gamma1), 0.0615, dp=4)
s_hat = 2 * math.sqrt(max(0.0, -gamma1))
check("Roll estimate s_hat", s_hat, 0.123, dp=3)
check("estimation error vs true 10 cents, cents", 100 * (s_hat - s), 2.3, dp=1)
check("Summary: 'within about two cents'", 100 * (s_hat - s), 2, dp=0)

finish("roll_spread_estimator")
