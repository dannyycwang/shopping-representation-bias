"""Final checks for the scoped evidence package; no inference or historical writes."""
from common import *
import re

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--saved-only',action='store_true');args=ap.parse_args()
    checks=[]
    def checked(name,detail):checks.append(dict(check=name,status='PASS',detail=detail))
    start=read(HERE/'qa/starting_state.json')
    revision=source_revision()
    for path,wanted in start['preexisting_files'].items():assert sha(ROOT/path)==wanted,path
    inventory=pd.read_csv(HERE/'data/source_inventory.csv').drop_duplicates('path')
    for r in inventory.itertuples():
        path=Path(r.path);path=path if path.is_absolute() else ROOT/path
        assert sha(path)==r.sha256,r.path
    checked('historical source and pre-existing edit preservation',dict(**revision,sources=len(inventory),preexisting_edits=len(start['preexisting_files'])))
    p0=read(HERE/'qa/p0_checks.json');assert all(r['status']=='PASS' for r in p0)
    graded=read(HERE/'qa/graded_recomputation.json');assert sum(r['rows'] for r in graded)==49227
    assert max(r['max_existing_error'] for r in graded)<1e-14
    checked('P0 and saved graded ranks',dict(p0_checks=len(p0),graded_conditions=len(graded),graded_records=49227))
    sources=pd.read_csv(HERE/'data/method_rank_provenance.csv');assert len(sources)==122
    for r in sources.itertuples():assert sha(ROOT/r.rank_source)==r.rank_sha256,r.rank_source
    assert not sources.historical_hash_status.eq('MISMATCH').any()
    checked('all method rank sources','122 saved conditions mapped to exact payload hashes and available embedding sidecars; missing runtime fields remain explicit')
    inc=pd.read_parquet(HERE/'data/inclusion_per_query.parquet')
    keys=['dataset','model','intervention','target_support','K']
    assert inc.groupby(keys).ngroups==48 and not inc.duplicated(keys+['query_id']).any()
    assert np.allclose(inc.persistent_inclusion+inc.VI+inc.persistent_omission,1,atol=1e-15,rtol=0)
    checked('complete inclusion companion','48 evaluated setting/support/cutoff cells; partition and unique query rows')
    mem=pd.read_parquet(HERE/'data/membership_per_query.parquet')
    assert len(mem)==48420
    assert np.array_equal(mem.exact_replacement,(mem.gain_count==mem.loss_count)&(mem.gain_count>0))
    assert np.allclose(mem.delta_recall,(mem.gain_count-mem.loss_count)/mem.n_highest,atol=1e-15,rtol=0)
    for r in mem.itertuples():
        gained=set(json.loads(r.gained_ids));lost=set(json.loads(r.lost_ids))
        assert gained.isdisjoint(lost) and len(gained)==r.gain_count and len(lost)==r.loss_count
    checked('membership accounting','48,420 records; disjoint identities, exact positive integer replacement, net identity')
    ctrl=pd.read_parquet(HERE/'data/controls_per_query.parquet');fixed=ctrl[ctrl.invariant_by_construction]
    assert (fixed.VI==0).all()
    assert np.allclose(fixed.Recall,fixed.persistent_inclusion,atol=1e-15,rtol=0)
    assert np.allclose(fixed.persistent_omission,1-fixed.Recall,atol=1e-15,rtol=0)
    assert ctrl.groupby(['dataset','model','method','schedule','K']).ngroups==152
    tr=read(HERE/'qa/transition_checks.json');assert len(tr)==48
    assert all(r['six_cells'] and r['row_sums'] and r['column_sums'] for r in tr)
    checked('fixed controls and transitions','152 method/reference/cutoff cells; 48 complete six-cell partitions')
    case=read(HERE/'data/illustrative_case_selection.json');candidates=pd.read_csv(HERE/'data/illustrative_case_candidates.csv')
    chosen=candidates.sort_values(['selection_hash','query_id']).iloc[0]
    assert int(chosen.query_id)==case['query_id'] and chosen.selection_hash==case['selection_hash']
    assert case['gain_count']==case['loss_count']>0 and case['source_numerator']==case['alternative_numerator']
    assert case['intervention']=='catalog_wide' and case['catalog_size']==42994
    example=pd.read_csv(HERE/'data/illustrative_replacement_case.csv',dtype={'product_id':str})
    query=mem[mem.dataset.eq('wands')&mem.model.eq('bge_base')&mem.method.eq('raw')&mem.K.eq(20)&mem.schedule.eq('C1')&mem.query_id.eq(case['query_id'])].iloc[0]
    assert case['source_numerator']==query.source_count and case['alternative_numerator']==query.alternative_count and case['denominator']==query.n_highest
    for kind,column in [('gained','gained_ids'),('lost','lost_ids')]:
        part=example[example.change.eq(kind)]
        assert set(part.product_id)==set(json.loads(query[column]))
        assert ((part.rank_C0>20)&(part.rank_C1<=20)).all() if kind=='gained' else ((part.rank_C0<=20)&(part.rank_C1>20)).all()
    assert (example.label=='Exact').all() and (example.intervention=='catalog_wide').all()
    checked('factual case selection',dict(query_id=case['query_id'],rule=case['selection_rule'],eligible=len(candidates)))
    ledger=pd.read_csv(HERE/'section6_claim_ledger.csv',keep_default_na=False)
    assert (ledger.manuscript_value==ledger.formatted_recomputation).all()
    flagged=ledger[ledger.status!='PASS'];assert len(flagged)==4
    assert (flagged.claim=='historical expanded-family VI').all()
    if not args.saved_only:assert flagged.resolution.str.startswith('resolved in proposed revision').all()
    checked('claim reconciliation',dict(rows=len(ledger),matched_numeric_displays=len(ledger),historical_provenance_corrections=len(flagged)))
    if args.saved_only:
        dump(HERE/'qa/saved_package_validation.json',dict(time_utc=now(),status='PASS',checks=checks,scope='Saved evidence only; not a harmonized or final-package completion record.'))
        print('SAVED PACKAGE VALIDATION PASS',len(checks),'checks',flush=True)
        return
    harmonized=read(HERE/'qa/harmonized_validation.json');assert len(harmonized)==4
    for model in ['minilm','bge_base']:
        folder=HERE/'harmonized'/model;profile=read(folder/'profile.json');plan=read(folder/'plan.json');done=read(folder/'complete.json')
        assert plan['profile']==done['profile']==profile['fingerprint']
        assert sha(folder/'inputs.parquet')==plan['inputs_sha256'] and sha(folder/'aliases.parquet')==plan['aliases_sha256']
        assert sha(folder/'vectors.npy')==done['vectors_sha256']
        aliases=pd.read_parquet(folder/'aliases.parquet');unique=pd.read_parquet(folder/'inputs.parquet')
        assert aliases.groupby('input_sha256').vector_index.nunique().max()==1
        assert len(unique)==done['unique_vectors'] and unique.vector_index.is_unique
        for ds,nq,ncat in [('wands',128,42994),('esci',499,10076)]:
            a=aliases[aliases.dataset.eq(ds)]
            assert len(a[a.role.eq('query')])==nq and len(a[a.role.eq('catalog')])==ncat
            c0=a[a.role.eq('target')&a.schedule.eq('C0')].merge(a[a.role.eq('catalog')],on='item_id',suffixes=('_target','_catalog'),validate='one_to_one')
            assert (c0.vector_index_target==c0.vector_index_catalog).all()
        probes=read(folder/'repeated_input_checks.json');assert len(probes)>=24
        # Preserve any real execution discrepancy as an explicit limitation, never force agreement.
        repeats_identical=all(r['bitwise_identical'] for r in probes)
        changed_membership=sum(bool(r.get('decision20_changed',False) or r.get('decision100_changed',False)) for r in probes)
        assert repeats_identical and changed_membership==0,(model,'Repeated execution is not identical; qualify the supplement before delivery.')
        checked(model+' harmonized provenance and repeat checks',dict(unique_vectors=len(unique),aliases=len(aliases),probes=len(probes),exact_numeric_equality=repeats_identical,changed_repeat_membership=changed_membership))
    assert all(r['all_families_nested'] and r['repeated_inputs_exactly_equal'] for r in harmonized)
    checked('expanded families','Fixed full/common-fitting support, nested states and exact repeated-input checks passed')
    tex=(HERE/'manuscript/section6_revised.tex').read_text(encoding='utf8')
    assert tex.count('\\input{tables/transition_single.tex}')==1
    assert 'EXPANSION_SENTENCE' not in tex and 'pending harmonized' not in tex
    log=(HERE/'qa/section6_acm_proof.log').read_text(encoding='utf8',errors='replace')
    for bad in ['Overfull','There were undefined references','Reference `','Fatal error']:
        assert bad not in log,('LaTeX proof',bad)
    assert re.search(r'Output written on .*',log)
    pdfchecks=[]
    for pdf in sorted((HERE/'figures').glob('*.pdf')):
        # Some Windows Poppler binaries misdecode non-ASCII absolute paths.
        info=subprocess.check_output(['pdfinfo',pdf.name],cwd=pdf.parent,text=True,encoding='utf8',errors='replace')
        assert re.search(r'Pages:\s+1',info)
        size=re.search(r'Page size:\s+([\d.]+) x ([\d.]+)',info)
        assert size and abs(float(size[1])-504)<.01
        images=subprocess.check_output(['pdfimages','-list',pdf.name],cwd=pdf.parent,text=True,encoding='utf8',errors='replace')
        assert not any(re.match(r'\s*1\s+\d+\s+',line) for line in images.splitlines()),'Raster content in '+pdf.name
        fonts=subprocess.check_output(['pdffonts',pdf.name],cwd=pdf.parent,text=True,encoding='utf8',errors='replace')
        assert ' yes ' in fonts,'Missing embedded font in '+pdf.name
        assert pdf.with_suffix('.png').exists()
        pdfchecks.append(dict(file=pdf.name,width_inches=7,pages=1,vector=True,embedded_fonts=True))
    assert len(pdfchecks)==5
    visual=read(HERE/'qa/visual_review.json');assert visual['all_final_assets_reviewed'] and visual['two_column_proof_reviewed']
    checked('publication assets',pdfchecks)
    checked('editorial handoff and ACM proof','Single transition table, resolved local references, no overfull boxes; full manuscript integration remains explicitly unavailable')
    dump(HERE/'qa/final_validation.json',dict(time_utc=now(),status='PASS',checks=checks,limitations=['The matching full manuscript source is unavailable; its integrated page count/float placement remains an author check.','Primary historical execution metadata has documented gaps; the new supplement does not retroactively erase them.']))
    manifest=[]
    for path in sorted(HERE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.name in ['output_manifest.csv','vectors.npy','section6_review_package.zip','package.json']:continue
        if path.suffix in ['.log','.aux','.out'] or (path.parent==HERE/'qa' and path.name.startswith('proof-')):continue
        manifest.append(dict(path=path.relative_to(HERE).as_posix(),sha256=sha(path),bytes=path.stat().st_size))
    csv(pd.DataFrame(manifest),HERE/'output_manifest.csv')
    print('FINAL VALIDATION PASS',len(checks),'checks;',len(manifest),'manifest entries',flush=True)

if __name__=='__main__':main()
