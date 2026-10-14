"""Reproduce every number quoted in technical notes/almgren_chriss.tex
(The Almgren-Chriss Optimal Execution Model: A Complete Derivation).

Run:  python "technical notes/reproduce/almgren_chriss.py"   (from any directory)

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
X, N, tau = 250.0, 5, 1.0          # shares, intervals, interval length
T = N * tau
sigma, eta, lam = 0.02, 0.01, 0.5  # $/share/interval, temp impact, risk aversion
eta_t = eta / tau                  # gamma neglected: eta~ = eta/tau

c = 1 + lam * sigma**2 * tau**2 / (2 * eta_t)
check("cosh(kappa~ tau) = 1 + lam sigma^2 tau^2 / (2 eta~)", c, 1.01, dp=2)
kt = math.acosh(c) / tau
check("exact discrete kappa~", kt, 0.14130, dp=5)

t = np.arange(N + 1) * tau
x_opt = X * np.sinh(kt * (T - t)) / np.sinh(kt * T)
x_uni = X * (1 - np.arange(N + 1) / N)
for j, q in enumerate([250.0, 194.3, 142.4, 93.4, 46.2, 0.0]):
    check(f"Table traj: x_{j} optimal", x_opt[j], q, dp=1)
for j, q in enumerate([250.0, 200.0, 150.0, 100.0, 50.0, 0.0]):
    check(f"Table traj: x_{j} uniform", x_uni[j], q, dp=1)

# Sanity: the closed form really minimises the discrete objective (eq. objective)
A = np.zeros((N - 1, N - 1))
b = np.zeros(N - 1)
for i in range(N - 1):
    A[i, i] = 2 * eta_t * 2 + 2 * lam * sigma**2 * tau
    if i > 0:
        A[i, i - 1] = -2 * eta_t
    if i < N - 2:
        A[i, i + 1] = -2 * eta_t
b[0] = 2 * eta_t * X
x_num = np.linalg.solve(A, b)
claim("closed-form trajectory equals numerical minimiser of U",
      np.max(np.abs(x_num - x_opt[1:N])) < 1e-9,
      f"max diff {np.max(np.abs(x_num - x_opt[1:N])):.2e}")

check("shares sold in first interval, x_0 - x_1", X - x_opt[1], 55.7, dp=1)
check("uniform pace per interval", X / N, 50.0, dp=1)

kappa = math.sqrt(lam * sigma**2 / eta)
check("continuous urgency kappa = sqrt(lam sigma^2/eta)", kappa, 0.14142, dp=5)
check("exact kappa~ (restated)", kt, 0.14130, dp=5)
rel_gap = abs(kappa - kt) / kt
claim("kappa is 'within 0.1%' of kappa~", rel_gap <= 1e-3,
      f"actual relative gap {rel_gap:.4%}")
x1_cont = X * math.sinh(kappa * (T - tau)) / math.sinh(kappa * T)
check("main text's x_1 using continuous kappa", x1_cont, 194.2, dp=1)
check("exact x_1 (restated)", x_opt[1], 194.3, dp=1)

# Limiting behaviour visible in the same example
lam0 = 1e-12
k0 = math.acosh(1 + lam0 * sigma**2 * tau**2 / (2 * eta_t)) / tau
x0 = X * np.sinh(k0 * (T - t)) / np.sinh(k0 * T)
claim("lambda -> 0 reproduces the uniform column", np.max(np.abs(x0 - x_uni)) < 1e-4,
      f"max diff {np.max(np.abs(x0 - x_uni)):.2e}")
lam_big = lam * 1e3
kb = math.acosh(1 + lam_big * sigma**2 * tau**2 / (2 * eta_t)) / tau
xb = X * np.sinh(kb * (T - t)) / np.sinh(kb * T)
claim("lambda x 1000 liquidates within the first interval or two (x_2 < 1% of X)",
      xb[2] < 0.01 * X, f"x_1={xb[1]:.2f}, x_2={xb[2]:.3f}")

finish("almgren_chriss")
