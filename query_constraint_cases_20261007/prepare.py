"""Freeze queries, schema rules and available comparisons before inspecting directions."""
import collections, importlib.metadata, re, sys
from common import *
from query_design import DESIGN, DEFERRED, PROTOCOL

def main():
    freeze=HERE/'protocol.json'
    if freeze.exists():
        assert sha(freeze)==(HERE/'protocol.sha256').read_text().strip()
        print('Frozen query design already exists; refusing to adapt it to results.');return
    protected=set(p for p in (ROOT/'paper_www2027').rglob('*') if p.is_file())
    protected.update(p for p in (ROOT/'option_coverage_20261007').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    protected.update(ROOT/p for p in git('diff','--name-only').splitlines() if (ROOT/p).is_file())
    dump('qa/preexisting_state.json',dict(commit=git('rev-parse','HEAD'),branch=git('branch','--show-current'),status=git('status','--short'),
         files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in sorted(protected)]))
    products,q,j=load();split=read('phase4/config/splits.json')
    ids=sorted(set(split['wands_heldout'])&set(j[j.label.eq('Exact')].query_id))
    assert ids==read('revision_graded_20260922/PROTOCOL.json')['populations']['wands']['eligible_ids']
    assert set(DESIGN)<=set(ids) and set(DEFERRED)<=set(ids)
    raw=pd.read_csv(src('data/raw/product.csv'),sep='\t',keep_default_na=False).set_index('product_id')
    keys=collections.Counter();keyvals=collections.defaultdict(collections.Counter)
    for p in products:
        original=raw.loc[p['product_id'],'product_features']
        assert p['attributes']==(original.split('|') if original else [])
        for a in p['attributes']:
            k,sep,v=a.partition(':')
            if sep:keys[k.strip()]+=1;keyvals[k.strip()][v.strip()]+=1
    dump('data/schema_census.json',dict(catalog_products=len(products),keys=[dict(key=k,occurrences=v,examples=keyvals[k].most_common(5)) for k,v in keys.most_common()]))
    constraints=[];queries=[]
    for r in q[q.query_id.isin(ids)].itertuples():
        qid=int(r.query_id);query=r.query;head,design,other=DESIGN.get(qid,('',[],[]))
        queryrows=[]
        for n,(span,kind,target,scope,ambiguous) in enumerate(design):
            assert span in query,(qid,span)
            start=query.index(span)
            c=dict(query_id=qid,query=query,constraint_id=f'{qid}:{kind}:{n}',query_span=span,span_start=start,span_end=start+len(span),
                   constraint_type=kind,normalized_value=target,modifier=head,scope=scope,query_ambiguous=ambiguous,
                   included=True,inclusion_reason='Literal query span with fixed field mapping; ambiguous spans remain exploratory C only' if ambiguous else 'Literal attribute modifier',
                   exclusion_reason='',other_explicit_clauses=other,scope_narrowed=kind=='material' and scope in ['frame','upholstery','vase','art surface'],known_seed=qid in [367,292])
            constraints.append(c);queryrows.append(c)
        uses=[]
        for use in ['outdoor','indoor']:
            if re.search(r'\b'+use+r'\b',query):uses.append(use)
        queries.append(dict(query_id=qid,query=query,product_head=head,constraints=[c['constraint_id'] for c in queryrows],
                            use_clauses=uses,other_explicit_clauses=other,known_seed=qid in [367,292],
                            exclusion_reason='' if design else DEFERRED.get(qid,'No unambiguous explicit color/material/shape/dimension modifier in the bounded families; no inference from style or product names')))
        if not design:
            constraints.append(dict(query_id=qid,query=query,constraint_id=f'{qid}:none',query_span='',span_start=None,span_end=None,
                  constraint_type='deferred' if qid in DEFERRED else 'none',normalized_value='',modifier='',scope='',query_ambiguous=False,
                  included=False,inclusion_reason='',exclusion_reason=queries[-1]['exclusion_reason'],other_explicit_clauses=[],scope_narrowed=False,known_seed=False))
        elif other:
            # These literal restrictions survive the design; joint matching cannot ignore them.
            constraints.append(dict(query_id=qid,query=query,constraint_id=f'{qid}:unassessed',query_span='; '.join(other),span_start=None,span_end=None,
                  constraint_type='unassessed_context',normalized_value='',modifier=head,scope='',query_ambiguous=True,included=False,inclusion_reason='',
                  exclusion_reason='Retained restriction; outside reliable mapped clause audit',other_explicit_clauses=other,scope_narrowed=False,known_seed=qid in [367,292]))
    cf=pd.DataFrame(constraints);cf['other_explicit_clauses']=cf.other_explicit_clauses.map(json.dumps)
    cf.to_csv(HERE/'query_constraints.csv',index=False)
    dump('data/queries.json',queries)
    cfg=read('phase2/config/phase2.json')
    profiles=pd.read_csv(src('revision_evidence_20260917/data/encoder_profiles_and_rank_sources.csv')).query("dataset=='wands' and model in ['bge_base','minilm']")
    executions=[];runs=[]
    for row in profiles.to_dict('records'):
        paths=[f"phase2/results/phase2_top50/wands_{row['model']}_native_{row['schedule']}.parquet",row['rank_source'],row['product_embedding'],row['query_embedding']]
        absent=[p for p in paths if not (ROOT/p).exists()]
        for path in paths[:2]:
            if (ROOT/path).exists():src(path)
        pm=read(str(Path(row['product_embedding']).with_suffix('.json')));qm=read(str(Path(row['query_embedding']).with_suffix('.json')))
        assert pm['model']==qm['model'] and pm['profile']==qm['profile']
        executions.append(dict(model=row['model'],schedule=row['schedule'],product_metadata=pm,query_metadata=qm,
            model_revision=pm['model']['revision'],tokenizer_revision=pm['model']['revision'],tokenizer_revision_basis='encoder constructor passes the model revision to AutoTokenizer',
            configured_batch=cfg['encoding']['batch_size_'+row['model']],actual_per_batch_partition='unknown: no per-batch telemetry saved',
            historical_oom_subdivision='unknown',gpu_model='unknown from per-artifact metadata',device=pm.get('device','unknown'),
            similarity='dot product of L2-normalized float32 embeddings',tie_rule='descending score then stable original catalog index',
            padding='dynamic longest in each configured batch',truncation=True,max_length=pm['model']['max_tokens'],
            product_embedding=row['product_embedding'],query_embedding=row['query_embedding'],qrel_source='phase2/data/processed/wands_judgments.csv',
            template='phase2/src/representations.py::_plain; title,class,category,description,pipe-joined attributes, newline-separated nonempty sections'))
        runs.append(dict(model=row['model'],schedule=row['schedule'],top_path=paths[0],pairs_path=paths[1],product_embedding=paths[2],query_embedding=paths[3],
                         available_rankings=not any(p in absent for p in paths[:2]),cache_available=not any(p in absent for p in paths[2:]),missing=absent))
    comparisons=[]
    for model in ['bge_base','minilm']:
        for schedule in ['C1','C2s1','C2s2','C2s3','C2s4','C2s5']:
            pair=[r for r in runs if r['model']==model and r['schedule'] in ['C0',schedule]]
            comparisons.append(dict(comparison=model+'_C0_'+schedule,model=model,before='C0',after=schedule,primary=model=='bge_base' and schedule=='C1',available=all(r['available_rankings'] for r in pair)))
    PROTOCOL.update(eligible_query_ids=ids,eligible_queries=len(ids),query_design=queries,query_constraint_sha256=sha(HERE/'query_constraints.csv'),
        query_design_source_sha256=sha(HERE/'query_design.py'),comparisons=comparisons,execution_profiles=executions,
        actual_starting_commit=git('rev-parse','HEAD'),historical_reference_tree_matches=git('rev-parse','HEAD^{tree}')==git('rev-parse',PROTOCOL['historical_reference_commit']+'^{tree}'),
        frozen_before_directional_counts=True,raw_product_attribute_mismatches=0)
    dump('protocol.json',PROTOCOL);(HERE/'protocol.sha256').write_text(sha(HERE/'protocol.json')+'\n')
    dump('data/runs.json',runs)
    for path in ['option_coverage_20261007/REPORT.md','option_coverage_20261007/CASE_AUDIT.md','option_coverage_20261007/core.py',
      'option_coverage_20261007/data/validated_top20.jsonl','option_coverage_20261007/data/input_manifest.json','option_coverage_20261007/data/provenance_checks.json',
      'phase2/src/representations.py','phase2/src/encoding.py','phase2/src/evaluation.py','phase2/scripts/prepare_data.py','phase2/scripts/run_dense.py',
      'revision_graded_20260922/data/wands_per_query.parquet','data/raw/label.csv','data/raw/query.csv']:src(path)
    dump('input_manifest.json',dict(starting_commit=git('rev-parse','HEAD'),historical_reference_commit=PROTOCOL['historical_reference_commit'],
         inputs=inventory(),python=sys.version,packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','pyarrow','transformers','tokenizers','torch']},
         missing_artifacts=[p for r in runs for p in r['missing']]))
    print('Frozen',len(ids),'queries,',len(DESIGN),'with attribute clauses,',len([c for c in constraints if c['included']]),'clauses;',sum(c['available'] for c in comparisons),'available comparisons.')

if __name__=='__main__':main()
