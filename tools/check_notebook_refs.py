"""Check the section numbers that the companion notebooks cite against the chapters.

Notebook markdown cites the book by hard-coded section number ("Section 8.2.2",
"Sections 1.5.1 through 1.5.4"; in the Chinese edition "第8.2.2节",
"第1.5.1至1.5.4节"). Those numbers go stale silently whenever a chapter gains or
loses a section. This script numbers every chapter's headings the way LaTeX does
(\\section, \\notocsection and \\subsection, including the \\weheading and
\\vignetteheading wrappers; starred headings and \\subsubsection are unnumbered) and checks each cited number:

  * MISSING  the number does not exist in the chapter: always an error;
  * SUSPECT  the number exists but the notebook heading that cites it shares no
             significant word with the section title it now points to (English
             notebooks only; review these by hand). Reported, not an error,
             unless --strict is given.

No notebook is executed, so the check takes a second or two.

Usage:
    python tools/check_notebook_refs.py                 # notebooks/ (+ zh/notebooks_zh if present)
    python tools/check_notebook_refs.py --list          # also print every reference and its target
    python tools/check_notebook_refs.py notebooks/chapter8.ipynb
Exit status is 1 if any reference is MISSING (or SUSPECT with --strict).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NUM = r"\d+\.\d+(?:\.\d+)?"
SEP_EN = r"(?:\s*(?:,|and|or|through|to|&|-|–|—)\s*)"
EN_REF = re.compile(rf"(?:Sections?|Secs?\.|§§?)\s*(?:~|\s)*({NUM}(?:{SEP_EN}(?:Sections?\s*)?{NUM})*)")
SEP_ZH = r"(?:\s*(?:、|，|,|和|与|及|至|到|或|-|–|—)\s*)"
ZH_REF = re.compile(rf"第?({NUM}(?:{SEP_ZH}第?{NUM})*)\s*节")
HEADING = re.compile(r"^\s*(?:\\(?:weheading|vignetteheading)\{\\(section|subsection)\}\{[^}]*\}|\\(notocsection|section|subsection))(\*?)\s*(?:\[[^\]]*\])?\s*\{(.*)")
CHAPTER = re.compile(r"^\s*\\chapter\*?\s*\{")
STOP = set("""a an the of and or in on to for with from by as at its is are be vs versus via into
section sections example examples worked chapter notebook numerical model models using use""".split())


def tex_for(chapter: int, zh: bool) -> Path:
    return ROOT / "zh" / "chapters_zh" / f"chapter{chapter}_zh.tex" if zh else ROOT / "chapters" / f"chapter{chapter}.tex"


def balanced(text: str) -> str:
    """Return the brace-balanced prefix of text (a heading title that started after '{')."""
    depth, out = 1, []
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out)


_numbering_cache: dict[Path, dict[str, str]] = {}


def numbering(tex: Path) -> dict[str, str]:
    """Map 'S.M' / 'S.M.K' (without the chapter number) to the heading title."""
    if tex in _numbering_cache:
        return _numbering_cache[tex]
    sec = sub = 0
    table: dict[str, str] = {}
    in_verbatim = False
    for line in tex.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        if re.match(r"\\begin\{(verbatim|lstlisting|minted)\}", stripped):
            in_verbatim = True
        if in_verbatim:
            if re.match(r"\\end\{(verbatim|lstlisting|minted)\}", stripped):
                in_verbatim = False
            continue
        if stripped.startswith("%"):
            continue
        if CHAPTER.match(line):
            sec = sub = 0
            continue
        m = HEADING.match(line)
        if not m or m.group(3) == "*":
            continue
        level = m.group(1) or m.group(2)
        title = balanced(m.group(4))
        if m.group(1):  # \weheading{...}{key}{Title}: group 4 is the title already
            pass
        if level in ("section", "notocsection"):
            sec, sub = sec + 1, 0
            table[f"{sec}"] = title
        else:
            sub += 1
            table[f"{sec}.{sub}"] = title
    _numbering_cache[tex] = table
    return table


def words(text: str) -> set[str]:
    text = re.sub(r"\\[a-zA-Z]+\*?(\{[^}]*\})?", " ", text)
    return {w for w in re.findall(r"[a-z][a-z0-9-]{2,}", text.lower()) if w not in STOP}


def split_numbers(group: str) -> list[str]:
    return re.findall(NUM, group)


def expand(nums: list[str], joined: str) -> list[str]:
    """Expand 'X through Y' / 'X至Y' ranges of sibling subsections into every member."""
    out = list(nums)
    for a, b in re.findall(rf"({NUM})\s*(?:through|to|至|到|-|–)\s*第?({NUM})", joined):
        pa, pb = a.split("."), b.split(".")
        if len(pa) == len(pb) and pa[:-1] == pb[:-1] and int(pa[-1]) < int(pb[-1]):
            out += [".".join(pa[:-1] + [str(k)]) for k in range(int(pa[-1]) + 1, int(pb[-1]))]
    return out


def check_notebook(nb_path: Path, args) -> tuple[int, int, list[str]]:
    zh = nb_path.stem.endswith("_zh")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    missing = suspect = 0
    lines = []
    pattern = ZH_REF if zh else EN_REF
    current_heading = ""
    for ci, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        src = "".join(cell["source"])
        for line in src.splitlines():
            if line.lstrip().startswith("#"):
                current_heading = line.lstrip("# ").strip()
            for m in pattern.finditer(line):
                for ref in expand(split_numbers(m.group(1)), m.group(1)):
                    chap, rest = ref.split(".", 1)
                    tex = tex_for(int(chap), zh)
                    if not tex.exists():
                        lines.append(f"  NOFILE  cell {ci}: {ref}  ({tex.name} not found)")
                        missing += 1
                        continue
                    title = numbering(tex).get(rest)
                    if title is None:
                        missing += 1
                        lines.append(f"  MISSING cell {ci}: {ref} does not exist in {tex.name}   [{line.strip()[:90]}]")
                        continue
                    context = line if line.lstrip().startswith("#") else current_heading + " " + line
                    is_suspect = (not zh) and not (words(context) & words(title))
                    if is_suspect:
                        suspect += 1
                        lines.append(f"  SUSPECT cell {ci}: {ref} -> \"{title[:70]}\"   [{context.strip()[:90]}]")
                    elif args.list:
                        lines.append(f"  ok      cell {ci}: {ref} -> \"{title[:70]}\"   [{context.strip()[:70]}]")
    return missing, suspect, lines


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="notebooks or folders (default: notebooks, zh/notebooks_zh)")
    ap.add_argument("--list", action="store_true", help="print every reference, not only problems")
    ap.add_argument("--strict", action="store_true", help="treat SUSPECT references as errors")
    args = ap.parse_args()
    paths = [Path(p) for p in args.paths] or [p for p in (ROOT / "notebooks", ROOT / "zh" / "notebooks_zh") if p.is_dir()]
    files = []
    for p in paths:
        files += sorted(p.glob("chapter*.ipynb"), key=lambda f: int(re.search(r"\d+", f.name).group())) if p.is_dir() else [p]
    total_missing = total_suspect = 0
    for f in files:
        missing, suspect, lines = check_notebook(f, args)
        total_missing += missing
        total_suspect += suspect
        if lines:
            print(f"{f.relative_to(ROOT) if f.is_absolute() else f}")
            print("\n".join(lines))
    print(f"\n{len(files)} notebooks: {total_missing} missing, {total_suspect} suspect")
    return 1 if total_missing or (args.strict and total_suspect) else 0


if __name__ == "__main__":
    sys.exit(main())
