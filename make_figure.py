#!/usr/bin/env python3
"""Figure 1: share of arcs labelled vs pruning threshold (unrestricted regime), three weight panels."""
import json
import subprocess

fig = json.load(open("fig_threshold.json"))
W = [("high_physical", "High physical $(0.8,0.1,0.1)$"),
     ("balanced", "Balanced $(0.4,0.3,0.3)$"),
     ("high_humanistic", "High humanistic $(0.1,0.1,0.8)$")]
STY = {"prune_only": ("dashed, mark=triangle*, mark options={solid}", "Prune-only (not exact)"),
       "screened": ("densely dotted, mark=square*, mark options={solid}", "Screened pricing"),
       "proposed": ("solid, very thick, mark=*", "OLA")}

parts = [r"""\documentclass[border=2pt]{standalone}
\usepackage{amsmath}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{calc}
\pgfplotsset{compat=1.18}
\begin{document}
\begin{tikzpicture}
\begin{groupplot}[group style={group size=3 by 1, horizontal sep=1.1cm, y descriptions at=edge left},
  width=5.6cm, height=5.2cm, ymode=log, ymin=0.5, ymax=150,
  xmin=-3, xmax=63, xtick={0,12,18,24,30,45,60},
  xlabel={$T_{\text{prune}}$ (min)}, tick label style={font=\scriptsize}, label style={font=\small},
  title style={font=\small}, grid=major, grid style={dashed, black!12},
  ytick={1,2,5,10,20,50,100}, yticklabels={1,2,5,10,20,50,100}]
"""]
for k, (w, title) in enumerate(W):
    opts = f"title={{{title}}}"
    if k == 0:
        opts += ", ylabel={Arcs labelled (\\% of $E$)}"
    if k == 2:
        opts += ", legend to name=grouplegend, legend columns=4, legend style={font=\\scriptsize, draw=black!40, /tikz/every even column/.append style={column sep=8pt}}"
    parts.append(f"\\nextgroupplot[{opts}]\n")
    parts.append("\\addplot[black!60, thin, dashdotted, domain=-3:63, samples=2] {100};\n")
    if k == 2:
        parts.append("\\addlegendentry{Two-stage pricing}\n")
    for m in ["prune_only", "screened", "proposed"]:
        pts = fig.get(f"{w}:{m}", [])
        if not pts:
            continue
        coords = " ".join(f"({t},{v:.3f})" for t, v in pts)
        parts.append(f"\\addplot[black, {STY[m][0]}, mark size=1.6pt] coordinates {{{coords}}};\n")
        if k == 2:
            parts.append(f"\\addlegendentry{{{STY[m][1]}}}\n")
parts.append(r"""\end{groupplot}
\node[anchor=north] at ($(group c2r1.south)+(0,-1.1cm)$) {\pgfplotslegendfromname{grouplegend}};
\end{tikzpicture}
\end{document}
""")
open("Figure_1.tex", "w").write("".join(parts))
subprocess.run(["pdflatex", "-interaction=nonstopmode", "Figure_1.tex"], stdout=subprocess.DEVNULL)
print("Figure_1.pdf written")
