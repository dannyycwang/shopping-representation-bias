"""Compile the standalone table fragments for visual QA; never edit the paper."""
from core import *
import subprocess, shutil, re

def main():
    folder=HERE/'qa/layout_preview';folder.mkdir(exist_ok=True)
    content=r'''\documentclass[10pt]{article}
\usepackage[letterpaper,margin=0.7in]{geometry}
\usepackage{booktabs,graphicx}
\pagestyle{empty}
\begin{document}
\input{../../paper/table1.tex}
\input{../../paper/table2.tex}
\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{../../paper/figure1.pdf}
\caption{Target-encoder change in Inclusion@20 by nested attempt budget.
Intervals resample query clusters, preserving all targets and conditions.
All pairs, including retained A, are included.}
\end{figure}
\clearpage
\end{document}
'''
    (folder/'preview.tex').write_text(content,encoding='utf8')
    result=subprocess.run([shutil.which('pdflatex'),'-interaction=nonstopmode','-halt-on-error','preview.tex'],cwd=folder,capture_output=True)
    (folder/'build_stdout.log').write_bytes(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout.decode('utf8',errors='replace')[-3000:]
    log=(folder/'preview.log').read_text(encoding='utf8',errors='replace')
    overfull=re.findall(r'Overfull[^\n]+',log);assert not overfull,overfull
    result=subprocess.run([shutil.which('pdftoppm'),'-scale-to','1800','-png','preview.pdf','preview'],cwd=folder,capture_output=True)
    assert result.returncode==0,result.stderr.decode('utf8',errors='replace')
    pngs=sorted(p.name for p in folder.glob('preview-*.png'))
    dump('qa/layout_build.json',dict(passed=True,overfull_boxes=overfull,pages=len(pngs),pngs=pngs,
        figure_pdf_sha256=sha(HERE/'paper/figure1.pdf'),table_sources={n:sha(HERE/'paper'/n) for n in ['table1.tex','table2.tex']},
        visual_review='PNGs require direct inspection after this script; compile success alone is insufficient'))
    print('LAYOUT PREVIEW READY',pngs,flush=True)

if __name__=='__main__':main()
