"""Reproduce every number quoted in technical notes/vasicek_single_factor.tex
(The Single-Factor (Vasicek) Model of Portfolio Credit Risk: A Complete Derivation).

Run:  python "technical notes/reproduce/vasicek_single_factor.py"   (from any directory)

Each check prints the computed value next to the value printed in the note,
compared at the note's printed precision. Exit status is 1 if any check fails.
"""
import math
import sys

from scipy.integrate import quad
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
def wcdr(pd, rho, alpha):                                  # eq. wcdr
    return norm.cdf((norm.ppf(pd) + math.sqrt(rho) * norm.ppf(alpha)) / math.sqrt(1 - rho))


def F_L(x, pd, rho):                                       # eq. cdf
    return norm.cdf((math.sqrt(1 - rho) * norm.ppf(x) - norm.ppf(pd)) / math.sqrt(rho))


def p_z(z, pd, rho):                                       # eq. pz
    return norm.cdf((norm.ppf(pd) - math.sqrt(rho) * z) / math.sqrt(1 - rho))


PD, LGD, rho, alpha = 0.03, 0.60, 0.20, 0.99
check("N^-1(0.03)", norm.ppf(PD), -1.8808, dp=4)
check("N^-1(0.99)", norm.ppf(alpha), 2.3263, dp=4)
check("sqrt(0.20)", math.sqrt(rho), 0.4472, dp=4)
check("sqrt(0.80)", math.sqrt(1 - rho), 0.8944, dp=4)
num = norm.ppf(PD) + math.sqrt(rho) * norm.ppf(alpha)
check("numerator N^-1(PD) + sqrt(rho) N^-1(alpha)", num, -0.8404, dp=4)
arg = num / math.sqrt(1 - rho)
check("argument of N", arg, -0.9396, dp=4)
w = wcdr(PD, rho, alpha)
check("WCDR(0.99), percent", 100 * w, 17.37, dp=2)
claim("'correlation multiplies the one-in-a-hundred outcome by nearly six'",
      5.5 <= w / PD < 6, f"ratio {w / PD:.2f}")
EAD = 100 * 1000.0
K = LGD * (w - PD) * EAD                                   # eq. capital
check("capital LGD (WCDR - PD) EAD, dollars", K, 8622, dp=0)

# Table rho: WCDR(0.99) by asset correlation, and the (constant) expected loss rate
for r, qv in [(0.00, 3.00), (0.05, 8.14), (0.10, 11.37), (0.20, 17.37), (0.30, 23.42), (0.50, 36.94)]:
    check(f"Table rho: WCDR(0.99) at rho={r:.2f}, percent", 100 * wcdr(PD, r, alpha), qv, dp=2)
    el = quad(lambda z: p_z(z, PD, r) * norm.pdf(z), -12, 12)[0] if r > 0 else PD
    check(f"Table rho: expected loss rate at rho={r:.2f}, percent", 100 * el, 3.00, dp=2)

# Stressed PD
PDs = 0.08
check("N^-1(0.08)", norm.ppf(PDs), -1.4051, dp=4)
check("sqrt(0.20) N^-1(0.99)", math.sqrt(rho) * norm.ppf(alpha), 1.0404, dp=4)
arg_s = (norm.ppf(PDs) + math.sqrt(rho) * norm.ppf(alpha)) / math.sqrt(1 - rho)
check("stressed argument of N", arg_s, -0.4077, dp=4)
check("stressed WCDR(0.99), percent", 100 * wcdr(PDs, rho, alpha), 34.17, dp=2)

# Remark: a useful check
check("F_L(17.37%) with PD=3%, rho=0.20", F_L(0.1737, PD, rho), 0.9900, dp=4)
check("integral of p(z) phi(z) dz (mean = PD)", quad(lambda z: p_z(z, PD, rho) * norm.pdf(z), -12, 12)[0],
      0.030000, dp=6)

# Basel corporate correlation, eq. baselrho: from 0.24 (PD -> 0) down towards 0.12 (PD -> 1)
def basel_rho(pd):
    wgt = (1 - math.exp(-50 * pd)) / (1 - math.exp(-50))
    return 0.12 * wgt + 0.24 * (1 - wgt)


check("Basel rho at PD -> 0", basel_rho(0.0), 0.24, dp=2)
check("Basel rho at PD = 1", basel_rho(1.0), 0.12, dp=2)

finish("vasicek_single_factor")
