"""Run every technical-note reproduction script in this folder and summarise.

Run:  python "technical notes/reproduce/run_all.py"   (from any directory)

Exit status is 1 if any script reports a mismatch or fails to run.
"""
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
SCRIPTS = ["almgren_chriss", "avellaneda_stoikov", "brock_hommes", "kaplan_meier",
           "kelly_criterion", "roll_spread_estimator", "sharpe_ratio_sampling",
           "vasicek_single_factor"]

results = []
for name in SCRIPTS:
    proc = subprocess.run([sys.executable, str(HERE / f"{name}.py")],
                          capture_output=True, text=True, encoding="utf-8")
    m = re.search(r"SUMMARY \S+: (\d+) checks, (\d+) mismatches", proc.stdout)
    checks, bad = (int(m.group(1)), int(m.group(2))) if m else (0, None)
    mismatches = [ln for ln in proc.stdout.splitlines() if ln.startswith("MISMATCH")]
    results.append((name, proc.returncode, checks, bad, mismatches, proc.stderr))

print(f"{'note':24s} {'checks':>6s} {'mismatches':>10s}  status")
for name, rc, checks, bad, _, _ in results:
    status = "OK" if rc == 0 else ("FAIL" if bad else "ERROR")
    print(f"{name:24s} {checks:6d} {bad if bad is not None else '?':>10}  {status}")
for name, rc, _, bad, mismatches, err in results:
    if rc != 0:
        print(f"\n--- {name} ---")
        for ln in mismatches:
            print("  " + ln)
        if bad is None:
            print(err.strip())
total = sum(r[2] for r in results)
failed = [r[0] for r in results if r[1] != 0]
print(f"\nTOTAL: {total} checks across {len(results)} notes; "
      f"{len(failed)} note(s) with mismatches or errors" + (f": {', '.join(failed)}" if failed else ""))
sys.exit(1 if failed else 0)
