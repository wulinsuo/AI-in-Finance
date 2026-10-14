"""Re-execute the companion notebooks and check that they still reproduce.

The book quotes numbers printed by these notebooks, so a notebook whose output
drifts (a library upgrade, an edited cell) silently invalidates the text. This
script executes each notebook in a scratch copy of the folder layout, so nothing
in the repository is overwritten, and compares every code cell's text output
with the output committed in the notebook:

  * words must match exactly (whitespace-normalized);
  * numbers must agree within a relative tolerance (default 1e-6), so harmless
    last-digit platform noise is not reported as drift;
  * images and stderr (warnings carry machine-specific paths) are ignored;
  * cells tagged "nbcheck-skip" (e.g. wall-clock timings) are skipped;
  * a cell that raises an error is always a failure.

Notebooks that download data (yfinance, urllib) are detected automatically and
can be included or excluded with --network / --no-network.

Usage:
    python tools/check_notebooks.py                       # all of notebooks/
    python tools/check_notebooks.py notebooks/chapter9.ipynb
    python tools/check_notebooks.py --no-network          # skip downloading ones
    python tools/check_notebooks.py zh/notebooks_zh       # the Chinese edition
Exit status is 1 if any notebook fails to execute or drifts.
"""
from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import nbformat
from nbclient import NotebookClient

# pyzmq warns under Windows' default Proactor event loop; the warning is harmless.
warnings.filterwarnings("ignore", message="Proactor event loop does not implement add_reader", category=RuntimeWarning)

NETWORK_PATTERN = re.compile(r"\byfinance\b|\byf\.download\b|\burllib\b|\brequests\.get\b|\bucimlrepo\b")
NUMBER = re.compile(r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
SKIP_TAG = "nbcheck-skip"


def uses_network(nb) -> bool:
    # Cells tagged nbcheck-skip (e.g. the Colab-only setup cell, which fetches requirements.txt
    # only when running on Colab) do not count.
    return any(c.cell_type == "code" and NETWORK_PATTERN.search(c.source)
               and SKIP_TAG not in c.get("metadata", {}).get("tags", []) for c in nb.cells)


def cell_text(cell) -> str:
    """Concatenate the comparable (text, stdout) outputs of a code cell."""
    parts = []
    for out in cell.get("outputs", []):
        kind = out.get("output_type")
        if kind == "stream":
            if out.get("name") == "stdout":
                parts.append(out.get("text", ""))
        elif kind in ("execute_result", "display_data"):
            data = out.get("data", {})
            if "text/plain" in data and not any(k.startswith("image/") for k in data):
                parts.append(data["text/plain"])
        elif kind == "error":
            parts.append(f"ERROR {out.get('ename')}: {out.get('evalue')}")
    return "".join(parts)


def tokens(text: str):
    """Split text into ('num', float, raw) and ('txt', str) tokens."""
    pos, out = 0, []
    for m in NUMBER.finditer(text):
        if m.start() > pos:
            out.extend(("txt", w) for w in text[pos:m.start()].split())
        raw = m.group(0)
        try:
            out.append(("num", float(raw.replace(",", "")), raw))
        except ValueError:
            out.append(("txt", raw))
        pos = m.end()
    out.extend(("txt", w) for w in text[pos:].split())
    return out


def same(expected: str, actual: str, rtol: float, atol: float) -> bool:
    a, b = tokens(expected), tokens(actual)
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x[0] != y[0]:
            return False
        if x[0] == "txt":
            if x[1] != y[1]:
                return False
        elif not math.isclose(x[1], y[1], rel_tol=rtol, abs_tol=atol):
            return False
    return True


def first_difference(expected: str, actual: str) -> str:
    el, al = expected.splitlines(), actual.splitlines()
    for i in range(max(len(el), len(al))):
        e = el[i] if i < len(el) else "<missing>"
        a = al[i] if i < len(al) else "<missing>"
        if e.split() != a.split():
            return f"      committed: {e.strip()[:110]}\n      now:       {a.strip()[:110]}"
    return "      (difference in spacing only)"


def stage(nb_path: Path, workdir: Path) -> Path:
    """Copy the notebook and the data folders beside it into a scratch layout:
    <work>/<folder>/notebook.ipynb, <folder>/*_data/, plus empty chapter folders
    that figure exports (savefig("../chapters/...")) can write into."""
    folder = workdir / nb_path.parent.name
    folder.mkdir(parents=True, exist_ok=True)
    for sib in ("chapters", "chapters_zh"):
        (workdir / sib).mkdir(exist_ok=True)
    for d in nb_path.parent.iterdir():
        if d.is_dir() and d.name.endswith("_data") and not (folder / d.name).exists():
            shutil.copytree(d, folder / d.name)
    dest = folder / nb_path.name
    shutil.copy2(nb_path, dest)
    return dest


def check(nb_path: Path, args) -> tuple[str, list[str], float]:
    committed = nbformat.read(nb_path, as_version=4)
    work = Path(tempfile.mkdtemp(prefix="nbcheck_"))
    try:
        staged = stage(nb_path, work)
        nb = nbformat.read(staged, as_version=4)
        client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                                resources={"metadata": {"path": str(staged.parent)}})
        t0 = time.time()
        try:
            client.execute()
        except Exception as exc:  # execution error: report the first line
            return "ERROR", [f"    execution failed: {str(exc).strip().splitlines()[-1][:200]}"], time.time() - t0
        elapsed = time.time() - t0
        problems = []
        old_cells = [c for c in committed.cells if c.cell_type == "code"]
        new_cells = [c for c in nb.cells if c.cell_type == "code"]
        for i, (old, new) in enumerate(zip(old_cells, new_cells)):
            if SKIP_TAG in old.get("metadata", {}).get("tags", []):
                continue
            e, a = cell_text(old), cell_text(new)
            if not same(e, a, args.rtol, args.atol):
                problems.append(f"    code cell {i + 1}:\n{first_difference(e, a)}")
        return ("DRIFT" if problems else "OK"), problems, elapsed
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=["notebooks"], help="notebooks or folders (default: notebooks)")
    net = ap.add_mutually_exclusive_group()
    net.add_argument("--no-network", action="store_true", help="skip notebooks that download data")
    net.add_argument("--network-only", action="store_true", help="run only notebooks that download data")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per cell")
    args = ap.parse_args()

    files = []
    for p in map(Path, args.paths):
        files += sorted(p.glob("*.ipynb"), key=lambda f: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", f.name)]) if p.is_dir() else [p]

    failures = 0
    for f in files:
        net_nb = uses_network(nbformat.read(f, as_version=4))
        if (args.no_network and net_nb) or (args.network_only and not net_nb):
            print(f"SKIP   {f}  (network)" if args.no_network else f"SKIP   {f}")
            continue
        status, problems, secs = check(f, args)
        print(f"{status:<6} {f}  ({secs:.0f}s)")
        for line in problems:
            print(line)
        failures += status != "OK"
        sys.stdout.flush()
    print(f"\n{len(files)} notebooks, {failures} failing")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
