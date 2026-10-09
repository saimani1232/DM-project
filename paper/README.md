# IEEE Conference Paper (LaTeX)

`main.tex` is the paper in IEEE conference format (`IEEEtran`, two columns, 9 pages), adapted to the course rules: sections Abstract, Keywords, Introduction, Methodology, Results and Discussion, Conclusion, References; Times New Roman 12 pt; 1.15 line spacing; figures and tables numbered in sequence and cited in the text; references numbered in order of first citation. All figures are in `figures/`. The paper has no separate `.bib` file because the references are inside `main.tex`.

**`IEEE_Paper_ABS_FDM.pdf`** is the compiled paper. It was compiled locally with MiKTeX (pdfLaTeX) with no errors and no overfull lines.

## Compile on Overleaf (recommended, nothing to install)
1. Go to https://www.overleaf.com and sign in (a free account is enough).
2. **New Project → Upload Project** → choose `IEEE_paper_overleaf.zip` (in this folder).
3. Overleaf opens `main.tex` and compiles it automatically (Menu → Compiler: **pdfLaTeX**).
4. Edit the author block (search for `Author One`) with the team's names, department, institution and e-mails.
5. Download the PDF from the **Download PDF** button.

## Compile locally
MiKTeX is installed on this PC (`%LOCALAPPDATA%\Programs\MiKTeX`). From the `paper` folder, run this twice (the second run fixes cross-references):
```
pdflatex main.tex
pdflatex main.tex
```

## Regenerate the figures
After re-running the pipeline (`python run_project.py`), run:
```
python paper/make_paper_figures.py
```

## Before submitting
- Replace the author placeholders.
- If the conference requires it, add the copyright notice line the conference provides (IEEE conferences send a `\IEEEpubid{...}` line).
- At 12 pt with 1.15 spacing (course rule) the paper is 9 pages. If it is later submitted to a real IEEE conference, remove `12pt` from `\documentclass`, the `\setstretch{1.15}` line and the two `\normalsize` lines to return to the standard 10 pt IEEE layout (about 6 pages).
