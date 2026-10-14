"""Reproduce every number quoted in technical notes/sharpe_ratio_sampling.tex
(The Sampling Distribution of the Sharpe Ratio, and What a Search Does to It).

Run:  python "technical notes/reproduce/sharpe_ratio_sampling.py"   (from any directory)

Closed-form quantities are compared at the note's printed precision. The note
also reports Monte Carlo results; those cannot be matched digit for digit, so
each is compared with an exact value or a seeded simulation here, within four
combined Monte Carlo standard errors (labelled "MC"). Runtime: about a minute.
Exit status is 1 if any check fails.
"""
import math
import sys

import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.signal import lfilter

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
    if dp is not None:
        shown = f"{computed:.{dp + 3}f}"
    elif tol is not None:
        shown = f"{computed:.6g} (tolerance +/-{tol:.2g})"
    else:
        shown = f"{computed:.6g}"
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
rng = np.random.default_rng(20021984)
Y, k = 5, 252
T = Y * k
sY = 1 / math.sqrt(Y)
z95 = stats.norm.ppf(0.95)


def a_N(N):                                        # eq. maxmean
    hi = quad(lambda x: 1 - stats.norm.cdf(x) ** N, 0, 12, limit=200)[0]
    lo = quad(lambda x: stats.norm.cdf(x) ** N, -12, 0, limit=200)[0]
    return hi - lo


def z_hurdle(N, alpha=0.05):                       # eq. hurdle, in z units
    return stats.norm.ppf((1 - alpha) ** (1 / N))


def annual_sr(r):                                  # rows = periods
    return math.sqrt(k) * r.mean(axis=0) / r.std(axis=0, ddof=1)


def sim_sd_mean(draw, reps, chunk=2000):
    out = []
    for _ in range(reps // chunk):
        out.append(annual_sr(draw((T, chunk))))
    x = np.concatenate(out)
    return x.std(ddof=1), x.mean(), len(x)


# --- Remark: accuracy at T = 1260 ---
corr = (T - 1) / (T - 3)
check("correction factor (T-1)/(T-3) at T=1260", corr, 1.0016, dp=4)
claim("exact SE exceeds approximate by less than two parts in a thousand",
      math.sqrt(corr) - 1 < 0.002, f"excess {math.sqrt(corr) - 1:.5f}")
se_exact = math.sqrt(k / T * corr)
check("exact annualized SE", se_exact, 0.4476, dp=4)
check("approximate annualized SE 1/sqrt(5)", sY, 0.4472, dp=4)
check("t_{1259, 0.95}", stats.t.ppf(0.95, T - 1), 1.6461, dp=4)
check("normal 0.95 quantile", z95, 1.6449, dp=4)

# --- Monte Carlo after Proposition (non-normal returns), 200,000 reps in the note ---
se_note_sd = 0.4472 / math.sqrt(2 * 200_000)
check("MC: SD of annualized SR, normal returns (exact t-based value)", se_exact, 0.4474,
      tol=4 * se_note_sd)
sd_t, _, M = sim_sd_mean(lambda sh: rng.standard_t(4, size=sh), 40_000)
check("MC: SD of annualized SR, Student-t(4) returns (seeded sim)", sd_t, 0.4470,
      tol=4 * math.hypot(se_note_sd, sd_t / math.sqrt(2 * M)))
sd_e, mean_e, M = sim_sd_mean(lambda sh: rng.exponential(1.0, size=sh) - 1.0, 40_000)
check("MC: SD of annualized SR, shifted-exponential returns (seeded sim)", sd_e, 0.4496,
      tol=4 * math.hypot(se_note_sd, sd_e / math.sqrt(2 * M)))
se_note_mean = 0.4472 / math.sqrt(200_000)
check("MC: mean annualized SR, exponential returns (seeded sim)", mean_e, -0.012,
      tol=4 * math.hypot(se_note_mean, sd_e / math.sqrt(M)))
bias = -2.0 / (2 * T) * math.sqrt(k)               # -skew/(2T), annualized, 2nd order
check("MC: mean annualized SR, exponential returns (2nd-order bias -g3/2T)", bias, -0.012,
      tol=4 * se_note_mean)
check("'theoretical 0.4472 in all three cases'", sY, 0.4472, dp=4)
check("standard error 'of 0.45'", sY, 0.45, dp=2)

# --- Section: why the sampling frequency cancels ---
bars = 6.5 * 60 / 5
check("five-minute bars per trading day", bars, 78, dp=0)
check("observations in five years of five-minute bars", T * bars, 98_000, rel=0.005)
check("observations in five years of daily bars", T, 1260, dp=0)
claim("annualized SE identical for daily and five-minute sampling",
      abs(math.sqrt(k) / math.sqrt(T) - math.sqrt(k * bars) / math.sqrt(T * bars)) < 1e-15)
sr_pp = 0.5 / math.sqrt(k)
check("per-period SR for annual 0.5 on daily data", sr_pp, 0.0315, dp=4)
check("1 + SR^2/2, daily", 1 + sr_pp**2 / 2, 1.0005, dp=4)
check("1 + SR^2/2, annual returns", 1 + 0.5**2 / 2, 1.125, dp=3)
check("ratio of the two corrections ('two hundred and fifty times')",
      (0.5**2 / 2) / (sr_pp**2 / 2), 250, rel=0.01)

# --- Worked example ---
naive = z95 * sY
check("sigma_Y = 1/sqrt(5)", sY, 0.4472, dp=4)
check("naive one-sided 5% hurdle 1.6449 x 0.4472", naive, 0.736, dp=3)
check("naive hurdle, 2 dp", naive, 0.74, dp=2)
tab = {1: (0.000, 0.000, 0.736), 10: (1.539, 0.688, 1.148),
       100: (2.508, 1.121, 1.468), 1000: (3.241, 1.450, 1.737)}
aN = {}
for N, (qa, qb, qc) in tab.items():
    aN[N] = a_N(N)
    check(f"Table maxN: a_N, N={N}", aN[N], qa, dp=3)
    check(f"Table maxN: expected best SR, N={N}", aN[N] * sY, qb, dp=3)
    check(f"Table maxN: honest 5% hurdle, N={N}", z_hurdle(N) * sY, qc, dp=3)
check("honest/naive hurdle for N=100 ('almost exactly twice')",
      z_hurdle(100) / z95, 2.0, rel=0.01)

# Monte Carlo confirmation (note: 10,000 searches). Annualized SR of an i.i.d.
# normal zero-edge series is exactly sqrt(k/T) t_{T-1}, so draw that directly.
S = 100_000
mx = (stats.t.rvs(T - 1, size=(S, 100), random_state=rng) * math.sqrt(k / T)).max(axis=1)
sd_mx = mx.std(ddof=1)
check("MC: mean best in-sample SR over 100 strategies (seeded sim)", mx.mean(), 1.124,
      tol=4 * sd_mx * math.hypot(1 / math.sqrt(10_000), 1 / math.sqrt(S)))
h100 = z_hurdle(100) * sY
p_exact = 1 - stats.t.cdf(h100 * math.sqrt(T / k), T - 1) ** 100
check("MC: P(best > 1.468) (exact t-based value)", p_exact, 0.0495,
      tol=4 * math.sqrt(p_exact * (1 - p_exact) / 10_000))
p_naive = 1 - stats.t.cdf(naive * math.sqrt(T / k), T - 1) ** 100
check("MC: P(best > naive 0.736) (exact t-based value; prints as 99.4%)", p_naive, 0.993,
      tol=4 * math.sqrt(p_naive * (1 - p_naive) / 10_000))

# How much data a real edge needs
check("Y for 50% power, single test", (z95 / 0.5) ** 2, 10.8, dp=1)
check("'about eleven years'", (z95 / 0.5) ** 2, 11, dp=0)
check("hundred-configuration hurdle in z units", z_hurdle(100), 3.283, dp=3)
check("Y for 50% power, best of 100", (z_hurdle(100) / 0.5) ** 2, 43.1, dp=1)
check("'about forty-three years'", (z_hurdle(100) / 0.5) ** 2, 43, dp=0)
z80 = stats.norm.ppf(0.80)
check("Phi^-1(0.80)", z80, 0.842, dp=3)
check("Table power: single test, 80% power", ((z95 + z80) / 0.5) ** 2, 24.7, dp=1)
check("Table power: best of 100, 80% power", ((z_hurdle(100) + z80) / 0.5) ** 2, 68.1, dp=1)
check("Table power: single test, 50% power", (z95 / 0.5) ** 2, 10.8, dp=1)
check("Table power: best of 100, 50% power", (z_hurdle(100) / 0.5) ** 2, 43.1, dp=1)

# --- Discussion: serial correlation (eq. ar1) ---
for phi, q_formula, q_mc in [(0.1, 0.494, 0.494), (0.2, 0.548, 0.551)]:
    f = sY * math.sqrt((1 + phi) / (1 - phi))
    check(f"AR(1) formula SE at phi={phi}", f, q_formula, dp=3)
    burn, M = 200, 20_000
    vals = []
    for _ in range(M // 2000):
        e = rng.standard_normal((T + burn, 2000))
        x = lfilter([1.0], [1.0, -phi], e, axis=0)[burn:]
        vals.append(annual_sr(x))
    v = np.concatenate(vals)
    se_m = v.std(ddof=1) / math.sqrt(2 * len(v))
    check(f"MC: simulated SE at phi={phi} (seeded sim, 20,000 reps)", v.std(ddof=1), q_mc,
          tol=4 * math.sqrt(2) * se_m)
check("naive SE (3 dp)", sY, 0.447, dp=3)
check("SE inflation at phi=0.2, percent (formula sqrt(1.5)-1)",
      100 * (math.sqrt(1.5) - 1), 22, dp=0)
check("Summary: SE inflation at phi=0.2, percent (restated)",
      100 * (math.sqrt(1.5) - 1), 22, dp=0)
check("Summary: best-of-100 clears the naive hurdle, percent (exact; note quotes MC)",
      100 * p_naive, 99.3, tol=400 * math.sqrt(p_naive * (1 - p_naive) / 10_000))
check("Summary: expected best SR of a 100-configuration search", aN[100] * sY, 1.121, dp=3)

# --- Discussion: correlated trials (figures quoted from the main text's simulation) ---
rho = 0.5
var_max = quad(lambda x: x**2 * 100 * stats.norm.pdf(x) * stats.norm.cdf(x) ** 99, -12, 12)[0] - aN[100] ** 2
closed = math.sqrt(1 - rho) * aN[100] * sY     # exact E[max] of equicorrelated normals
sd_corr = sY * math.sqrt(rho + (1 - rho) * var_max)
check("MC: expected best SR at pairwise correlation 0.5 (closed form "
      "sqrt(1-rho) a_100 sigma_Y; main text used 2,000 reps)", closed, 0.80,
      tol=4 * sd_corr / math.sqrt(2000))
check("expected best SR, independent trials (restated)", aN[100] * sY, 1.121, dp=3)
n_eff = min(range(2, 101), key=lambda n: abs(a_N(n) * sY - 0.80))
claim("best SR of 0.80 corresponds to 'roughly eighteen' independent trials",
      17 <= n_eff <= 19, f"N_eff={n_eff} (at the closed-form {closed:.3f}: "
      f"{min(range(2, 101), key=lambda n: abs(a_N(n) * sY - closed))})")

finish("sharpe_ratio_sampling")
