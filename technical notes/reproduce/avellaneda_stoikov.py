"""Reproduce every number quoted in technical notes/avellaneda_stoikov.tex
(The Avellaneda-Stoikov Market-Making Model: A Complete Derivation).

Run:  python "technical notes/reproduce/avellaneda_stoikov.py"   (from any directory)

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
s, sigma, gamma, kappa, tau, q = 100.0, 2.0, 0.1, 1.5, 1.0, 5

r = s - q * gamma * sigma**2 * tau                      # Corollary reservation
check("reservation price r = s - q gamma sigma^2 (T-t)", r, 98.00, dp=2)
check("inventory skew s - r", s - r, 2.00, dp=2)
gap = gamma * sigma**2 * tau                            # r^a - r^b
check("r^a - r^b = gamma sigma^2 (T-t)  [note: 0.1(4)(1)]", gap, 0.40, dp=2)
rb = r - gap / 2                                        # eq. rb
ra = r + gap / 2                                        # eq. ra
check("reservation bid r^b", rb, 97.80, dp=2)
check("reservation ask r^a", ra, 98.20, dp=2)
check("1 + gamma/kappa", 1 + gamma / kappa, 1.0667, dp=4)
check("ln(1 + gamma/kappa)", math.log(1 + gamma / kappa), 0.0645, dp=4)
delta = math.log(1 + gamma / kappa) / gamma             # eq. markup
check("optimal markup delta*", delta, 0.645, dp=3)
pb, pa = rb - delta, ra + delta                         # eq. quotes
spread = pa - pb
check("optimal bid p^b", pb, 97.15, dp=2)
check("optimal ask p^a", pa, 98.85, dp=2)
check("total spread p^a - p^b (eq. totalspread)", spread, 1.69, dp=2)
claim("total spread equals gamma sigma^2 (T-t) + (2/gamma) ln(1+gamma/kappa)",
      abs(spread - (gap + 2 / gamma * math.log(1 + gamma / kappa))) < 1e-12)

# Table: inventory-aware vs naive quotes of the same total width
check("Table: inventory-aware bid", pb, 97.15, dp=2)
check("Table: inventory-aware ask", pa, 98.85, dp=2)
nb, na = s - spread / 2, s + spread / 2
check("Table: naive bid (centred on s)", nb, 99.15, dp=2)
check("Table: naive ask (centred on s)", na, 100.85, dp=2)
check("bid shift naive - aware = s - r", nb - pb, 2.00, dp=2)
check("ask shift naive - aware = s - r", na - pa, 2.00, dp=2)

finish("avellaneda_stoikov")
