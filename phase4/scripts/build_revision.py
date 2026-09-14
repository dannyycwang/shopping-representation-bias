"""Assemble the completed evidence and compile a draft for visual verification."""
from pathlib import Path
import subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'phase4/results';S=ROOT/'phase4/scripts';PAPER=ROOT/'paper_www2027'
while not (OUT/'queue_completed.json').exists():time.sleep(2)
for script in ['complete_evidence.py','build_presentation.py','write_manuscript.py','write_results.py','write_appendix.py']:
    print('BUILD',script,flush=True)
    subprocess.run([sys.executable,str(S/script)],cwd=ROOT,check=True)
for command in [['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],['bibtex','main'],['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex']]:
    with (OUT/'latex_build.log').open('a',encoding='utf8') as f:subprocess.run(command,cwd=PAPER,stdout=f,stderr=subprocess.STDOUT,check=True)
print('Compiled draft; visual/page/citation verification remains.',flush=True)
