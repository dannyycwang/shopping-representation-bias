"""Compile supplied snippets in an isolated manuscript-width proof, not the paper."""
import subprocess
from io_common import HERE, dump, sha

def main():
    qa = HERE / 'qa'
    insertions = (HERE/'LATEX_INSERTIONS.tex').read_text(encoding='utf-8')
    insertions = insertions.replace('../revision_evidence_20260917/figure_revision/figures/', '../figures/')
    preamble = r'''\documentclass[10pt,twocolumn]{article}
\usepackage[paperwidth=8.5in,paperheight=11in,textwidth=7.1in,textheight=9.4in,columnsep=.2in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{graphicx,booktabs,longtable}
\newcommand{\Description}[1]{}
\begin{document}
'''
    proof = preamble + insertions + '\n' + (HERE/'CANCELLATION_MAIN_TABLE.tex').read_text(encoding='utf-8')
    proof += '\n\\clearpage\\onecolumn\n' + (HERE/'CANCELLATION_APPENDIX.tex').read_text(encoding='utf-8')
    proof += '\n\\clearpage\n' + (HERE/'SETUP_SNIPPET.tex').read_text(encoding='utf-8') + '\n\\end{document}\n'
    (qa/'latex_insertion_proof.tex').write_text(proof, encoding='utf-8')
    cmd = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', '-no-shell-escape', '-disable-installer', 'latex_insertion_proof.tex']
    runs = []
    for _ in range(2):
        p = subprocess.run(cmd, cwd=qa, capture_output=True)
        runs.append(p.returncode)
        if p.returncode:
            print(p.stdout.decode('utf-8', errors='replace')[-6000:])
            raise RuntimeError('LaTeX insertion proof failed')
    log = (qa/'latex_insertion_proof.log').read_text(encoding='utf-8', errors='replace')
    overfull = [s for s in log.splitlines() if 'Overfull' in s or 'Float too large' in s or 'Missing character' in s]
    assert not overfull, overfull
    names = ['LATEX_INSERTIONS.tex', 'CANCELLATION_MAIN_TABLE.tex', 'CANCELLATION_APPENDIX.tex', 'SETUP_SNIPPET.tex']
    dump(qa/'latex_check.json', {'status': 'PASS', 'two_pass_exit_codes': runs, 'overfull_or_missing_glyph_warnings': overfull,
                                'text_width_inches': 7.1, 'column_width_inches': 3.45,
                                'sources': {n: sha(HERE/n) for n in names},
                                'scope': 'Isolated article proof of LaTeX syntax, dimensions, and table fit; manuscript not compiled or edited.'})
    print('PASS: all figure insertions, main/appendix tables and shared setup compile; no overfull boxes or missing glyphs.')

if __name__ == '__main__':
    main()
