"""Documentation, hashes, and a portable analysis/report delivery bundle."""
from common import *
import subprocess, zipfile, datetime

def main():
    status=read(HERE/'qa/validation.json');assert status['status']=='PASS'
    latex=read(HERE/'qa/latex_check.json')
    tracked=set(subprocess.check_output(['git','ls-files'],cwd=ROOT).decode().splitlines())
    manifest=read(DATA/'source_manifest.json');known={x['path'] for x in manifest['files']}
    extra=read(DATA/'additional_cost_sources.json')['files']
    for name in ['phase2/scripts/run_dense.py','phase4/scripts/time_retrieval.py','paper_www2027/references.bib']:
        extra.append(dict(path=name,sha256=sha(ROOT/name)))
    for e in extra:
        if e['path'] in known:continue
        p=ROOT/e['path'];entry=dict(path=e['path'],local_exists=True,git_tracked=e['path'] in tracked,availability='local_file',bytes=p.stat().st_size,sha256=e['sha256'],git_distribution='payload_tracked' if e['path'] in tracked else 'metadata_only' if p.with_suffix('.json').relative_to(ROOT).as_posix() in tracked else 'not_tracked')
        if p.suffix=='.npy':
            a=np.load(p,mmap_mode='r');entry.update(shape=list(a.shape),dtype=str(a.dtype),array_bytes=a.nbytes)
        manifest['files'].append(entry);known.add(e['path'])
    dump(DATA/'source_manifest.json',manifest)
    (HERE/'COMPLETION_STATUS.md').write_text(f'''# Completion status\n\nP0--P3 numerical work is complete. Final whole-paper layout is incomplete because the complete LaTeX source corresponding to PDF version (6) was not supplied. Both newly supplied chapter drafts were used and revised in separate copies.\n\n| Priority | Status | Delivered evidence |\n|---|---|---|\n| P0 | Complete | `SOURCE_AUDIT.md`, frozen `PROTOCOL.json`, `{len(manifest['files'])}` source records with SHA256, payload/Git status, revisions, array dtype/shape, ordered IDs and primary supports. Actual HEAD 892ffc9aca25b275d861aee424559b90592d632f; tracked tree matches b73e52a. |\n| P1 | Complete | Four dataset/encoder cells x seven full-catalog FP32 RRF conditions. All 28 reconstructed highest-label rank sets exactly match historical raw ranks; four C0 hybrid full score matrices/ranks/top1000 match Phase IV. New ranks/bits/per-query data and matched contrasts are saved. No model encoding. |\n| P2 | Complete | Four-cell joint strategy table, every pure canonical rule, all 18 canonical conditions in separate complete tables, Set-Mean, M2 centroid/max, BM25 and hybrid; BGE M4/7 fixed sweep. Common seed paired intervals, all membership partitions, cancellation, persistent states, costs and distinct views. All 20 reconstructed aggregation/control score matrices equal saved scores exactly. |\n| P3 | Complete | Six dataset/encoder cells, both interventions, full/fitting separately, six K values; 144 support/cutoff conditions with query-macro and pair-micro results. Paired K changes, all-target severity, conditional weighting/denominators, worst-rank and range ECDFs, and source-order-directed losses. Fixed support and monotonic persistent-state checks pass. |\n| Figures | Complete | Seven vector PDF/SVG figures with plot CSV, caption, ACM Description and command. Fig. 3 reused byte-identically; Fig. 4 keeps all numbers/panels and changes only legend wording. All seven visually inspected at supplied 7.1-inch width; text at least 8 pt after the small ACM width reduction. |\n| Chapters 5/6 | Complete as new copies | `manuscript/chapter5_experimental_setup.tex`, `chapter6_results.tex`, optional appendix and `MANUSCRIPT_INSERTIONS.tex`. Source-order point table replaced by joint mean-Recall/VI table. Original package retained under `inputs/`. |\n| Local compilation | Complete | Real acmart two-column chapter/appendix proof: {latex['acm_chapter_proof_pages']} isolated pages. Chapters 5/6 and deferred main floats occupy pages 1--4; selected appendix pages 5--7 and real bibliography page 8. Separate artifact-table proof also passes. No overflow, oversized floats, missing glyphs or unresolved references. This is not an eight-page full-paper result. |\n| Whole manuscript 8/12-page check | Incomplete | PDF (6) has 12 pages, but matching complete source is absent. That existing count does not certify the revised full paper. Latest Chapters 1--4, discussion/conclusion, references and final float placement must be compiled together. No font reduction or forced full-paper placement was used to claim compliance. |\n| P4 | Not run (out of scope) | No added schedules, rewriting track, training or graded metrics. |\n\nThe only matched cell/cutoff with higher mean Recall and higher VI point estimates is ESCI/BGE at K100; both CIs contain zero. The asserted general/statistically established effectiveness--instability trade-off is not supported. See `CLAIM_VERDICTS.md` for separate descriptive and inferential judgments.\n\nNo source inputs needed for P0--P3 remain missing. Whole-manuscript source is the outstanding input. Existing Phase II--VI results, the September 17 package, original chapter ZIP and all eight pre-existing tracked edits are preserved.\n''',encoding='utf8')
    (HERE/'REPRODUCE.md').write_text(r'''# Reproduction

Run from the repository root with the local Python 3.10 runtime and NumPy 1.26.4, pandas 2.3.3, PyArrow 24.0.0, Matplotlib 3.10.9, threadpoolctl, pypdf, SciPy and fontTools. Exact versions used for core execution are in `qa/runtime.json`; inherited pinned embedding implementations are in the source manifest. Four CPU BLAS threads are fixed. No GPU inference, training, network model download or paid API is used.

```powershell
$env:PYTHONIOENCODING='utf-8'
python revision_completion_20260921/scripts/audit_sources.py
python revision_completion_20260921/scripts/hybrid.py
python revision_completion_20260921/scripts/analyze.py joint
python revision_completion_20260921/scripts/analyze.py diagnostics
python revision_completion_20260921/scripts/controls_and_cost.py
python revision_completion_20260921/scripts/figures.py
python revision_completion_20260921/scripts/report.py
python revision_completion_20260921/scripts/latex_qa.py
python revision_completion_20260921/scripts/validate.py
python revision_completion_20260921/scripts/finalize.py
```

The audit accepts `--chapter-package PATH` and `--reference-pdf PATH` if the supplied external inputs moved. The initial request attachment is also hashed at its original local path. Do not replace frozen historical artifacts with different model revisions or current downloads. Locally present NPY caches are often represented only by their JSON metadata in Git; consult `data/source_manifest.json` before attempting full numerical reproduction from a fresh clone. Missing payloads must be restored by hash; highest-label ranks or top1000 alone cannot reconstruct full-ranking RRF. No canonical inference is rerun: all 18 rank conditions are read through their September 17 source records.

The delivery ZIP contains the newly derived pair/query tables, bootstrap multiplicity matrices and their sorted query IDs, all plot CSVs, ranks and top1000 catalog indices, reports, scripts and the supplied chapter inputs. It does not duplicate all historical multi-GB embedding caches. Array top1000 indices map to `data/{dataset}_catalog_ordering.csv`; query IDs are saved with each array. Every new pair artifact includes complete Hq, including absent-in-both pairs.

`figures.py` reads inherited September 17 figure data/fonts plus the new diagnostic tables. It uses Matplotlib, not generative imagery. All figure sources, captions, accessible descriptions and commands are in `figures/`. The copied Fig. 3 is byte-identical. Fig. 4 numeric plot CSVs are byte-identical to their historical source CSVs. The two cutoff intervention groups and severity/ECDF plots remain optional appendix/artifact figures; no new main Fig. 5 is imposed.

`latex_qa.py` requires local MiKTeX/acmart, BibTeX and Poppler; automatic package installation and shell escape are disabled. It uses real WANDS/ESCI bibliography entries from the repository, not validation stubs. The proof uses ACM two-column geometry, natural font sizes, and `emergencystretch=3em` to permit clean line breaks; consider that preamble setting when integrating. Model names and dataset/encoder slashes have explicit allowable breaks. The proof's page breaks are local test boundaries, not proposed whole-paper float placements. Updated chapters use `tables/` and `figures/` relative to the manuscript root. Preserve author-controlled Chapters 1--4.

After rendering, visually inspect all seven figures and all pages of the chapter and table proofs. The delivered record is `qa/VISUAL_QA.md`. `validate.py` verifies historical source hashes, package input hashes, primary supports, all K-membership/rank matches, integer membership identities, state partitions, fixed fitting support, monotonic persistent states, vector output and LaTeX diagnostics. `finalize.py` refreshes output hashes and packages results, excluding scratch PNGs/logs and the ZIP itself. No historical `write_results.py` or `write_manuscript.py` is called.
''',encoding='utf8')
    (HERE/'README.md').write_text('''# Chapter 5/6 completion, 2026-09-21\n\nStart with `COMPLETION_STATUS.md`, `CLAIM_VERDICTS.md`, and `tables/strategy_joint_main.tex`. Updated author drafts are in `manuscript/`; original supplied drafts are in `inputs/chapter56_revision/`. `MANUSCRIPT_INSERTIONS.tex` includes the updated chapters. Data definitions are in `data/DATA_DICTIONARY.md`; source audit, protocol and reproducibility instructions are at package root.\n\nP0--P3 are computed and checked. The full-paper page limit remains unverified pending complete LaTeX corresponding to PDF (6). No Chapters 1--4 or historical outputs were modified.\n''',encoding='utf8')
    (HERE/'.gitignore').write_text('__pycache__/\nqa/*.png\nqa/*.aux\nqa/*.log\nqa/*.out\nqa/*.bbl\nqa/*.blg\n*.zip\n',encoding='utf8')
    # Optional figure placement snippets retain captions and accessibility descriptions.
    def esc(s):return s.replace('%',r'\%').replace('&',r'\&').replace('_',r'\_').replace('<=',r'$\leq$').replace('>',r'$>$')
    snippets=[]
    for name,r in read(HERE/'figures/captions.json').items():
        snippets.append('\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{figures/'+name+'.pdf}\n\\caption{'+esc(r['caption'])+'}\n\\Description{'+esc(r['ACM_Description'])+'}\n\\end{figure*}\n')
    (HERE/'FIGURE_INSERTIONS.tex').write_text('% Optional insertions; do not duplicate main figures already included by chapter6.\n\n'+'\n'.join(snippets),encoding='utf8')
    sources=read(DATA/'source_manifest.json')['files']
    dump(HERE/'SOURCE_HASHES.json',{e['path']:e.get('sha256') for e in sources})
    exclude={'OUTPUT_HASHES.json','revision_completion_20260921.zip','DELIVERY.json'}
    files=[p for p in HERE.rglob('*') if p.is_file() and p.name not in exclude and '__pycache__' not in p.parts and not (p.parent.name=='qa' and p.suffix in ['.png','.aux','.log','.out','.bbl','.blg'])]
    dump(HERE/'OUTPUT_HASHES.json',{p.relative_to(HERE).as_posix():sha(p) for p in sorted(files)})
    bundle=HERE/'revision_completion_20260921.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[HERE/'OUTPUT_HASHES.json']:z.write(p,'revision_completion_20260921/'+p.relative_to(HERE).as_posix())
    dump(HERE/'DELIVERY.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive=bundle.name,bytes=bundle.stat().st_size,sha256=sha(bundle),files=len(files)+1,validation=status['status']))
    print('DELIVERY',bundle.stat().st_size,'bytes',len(files)+1,'files',flush=True)

if __name__=='__main__':main()
