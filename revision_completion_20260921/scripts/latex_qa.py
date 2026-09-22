"""Compile the new chapters under acmart without rebuilding the old paper."""
from common import *
import subprocess, shutil
from pypdf import PdfReader

def run(cmd):
    p=subprocess.run(cmd,cwd=HERE,capture_output=True)
    if p.returncode:print(p.stdout.decode('utf8',errors='replace')[-5000:]);raise RuntimeError(cmd)
    return p

def main():
    shutil.copyfile(ROOT/'paper_www2027/references.bib',HERE/'qa/references.bib')
    main=r'''\documentclass[sigconf,nonacm]{acmart}
\setkeys{acmart.cls}{balance=false}
\settopmatter{printacmref=false,printccs=false,printfolios=true}
\usepackage{booktabs,longtable}
\emergencystretch=3em
\title{Isolated proof of revised Chapters 5 and 6}
\author{Author proof}
\begin{document}
\raggedbottom
\twocolumn
\setcounter{section}{4}
\setcounter{figure}{2}
\input{MANUSCRIPT_INSERTIONS.tex}
\clearpage
\appendix
\input{manuscript/appendix_evaluation_details.tex}
\clearpage
\bibliographystyle{ACM-Reference-Format}
\bibliography{qa/references}
\end{document}
'''
    (HERE/'qa/chapters_acm_proof.tex').write_text(main,encoding='utf8')
    cmd=['pdflatex','-interaction=nonstopmode','-halt-on-error','-no-shell-escape','-disable-installer','-output-directory=qa','qa/chapters_acm_proof.tex']
    run(cmd)
    # BibTeX's input names are resolved from package root; output uses qa/ prefix.
    run(['bibtex','qa/chapters_acm_proof']);run(cmd);run(cmd)
    log=(HERE/'qa/chapters_acm_proof.log').read_text(encoding='utf8',errors='replace')
    problems=[x for x in log.splitlines() if any(v in x for v in ['Overfull','Float too large','Missing character','undefined'])]
    pages=len(PdfReader(HERE/'qa/chapters_acm_proof.pdf').pages)
    # All large artifact tables get separate page geometry, never scale-to-fit.
    art=r'''\documentclass[10pt]{article}
\usepackage[paperwidth=8.5in,paperheight=11in,textwidth=7.1in,textheight=9.4in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{booktabs,longtable}
\begin{document}
'''
    for name in ['strategy_states_K20','strategy_states_K100','bge_budget_sweep','cancellation_all_conditions']:
        art+='\\input{tables/'+name+'.tex}\n\\clearpage\n'
    art+='\\end{document}\n';(HERE/'qa/artifact_tables_proof.tex').write_text(art,encoding='utf8')
    artcmd=['pdflatex','-interaction=nonstopmode','-halt-on-error','-no-shell-escape','-disable-installer','-output-directory=qa','qa/artifact_tables_proof.tex']
    run(artcmd);run(artcmd)
    artlog=(HERE/'qa/artifact_tables_proof.log').read_text(encoding='utf8',errors='replace')
    artproblems=[x for x in artlog.splitlines() if any(v in x for v in ['Overfull','Float too large','Missing character','undefined'])]
    dump(HERE/'qa/latex_check.json',dict(acm_chapter_proof_pages=pages,acm_problems=problems,artifact_table_problems=artproblems,complete_latest_manuscript_compiled=False,scope='isolated ACM chapters and selected appendix; no Chapters 1-4 source or final float placement'))
    print('ACM proof',pages,'pages; problems:',problems,'artifact:',artproblems,flush=True)
    for path in (HERE/'figures').glob('*.pdf'):
        run(['pdftoppm','-r','130','-singlefile','-png',str(path),str(HERE/'qa'/path.stem)])
    run(['pdftoppm','-r','110','-png',str(HERE/'qa/chapters_acm_proof.pdf'),str(HERE/'qa/chapters_acm_proof')])
    run(['pdftoppm','-r','110','-png',str(HERE/'qa/artifact_tables_proof.pdf'),str(HERE/'qa/artifact_tables_proof')])

if __name__=='__main__':main()
