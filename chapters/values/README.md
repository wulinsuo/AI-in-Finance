# Generated values

The files `ch9.tex`, `ch15.tex` and `ch21.tex` are **generated** by the companion notebooks
`notebooks/chapter9.ipynb`, `chapter15.ipynb` and `chapter21.ipynb`. Each defines one LaTeX macro per
number the chapter quotes from a fitted model (AUCs, PR-AUCs, fitted coefficients, SHAP values, approval
rates, fairness ratios, and the corner coordinates of Chapter 15's ROC and precision-recall figure), for
example

```latex
\newcommand{\vChNineMonoAUCCon}{0.8757}
```

and the chapter uses `\vChNineMonoAUCCon` where it used to type `0.8757`. Each chapter loads its file
right after `\begin{document}` with `\input{\subfix{values/chN}}`; `\subfix` (from the `subfiles`
package) makes the same path work in a standalone chapter build (run from `chapters/`) and in the
combined book (run from the repository root).

Do not edit these files by hand. Re-run the notebook instead: its `book_value(...)` cells collect the
values as they are computed, and its last cell rewrites the file. If a library upgrade or an edited
cell changes a number, the change shows up as a diff in this folder and in the rebuilt PDF, rather
than silently contradicting the text. The values are rounded half-up to the precision the text prints
(where the text rounds a value the notebook prints to four or six decimals, the notebook rounds the
same way, and says so in a comment).

The Chinese edition has the same mechanism in its own repository (`zh/chapters_zh/values/`, written by
the Chinese notebooks), with the same macro names.
