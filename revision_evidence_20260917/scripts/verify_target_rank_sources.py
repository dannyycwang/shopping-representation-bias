"""Independent checks of inherited target ranks and old-entry replacement semantics."""
from common import *
import ast

def original_rank_function():
    path=ROOT/'phase3/scripts/analyze_target_only_permutations.py'
    tree=ast.parse(path.read_text(encoding='utf8'))
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='rank_target')
    scope={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope['rank_target']

def main():
    fn=original_rank_function();rng=np.random.default_rng(SEED);checks=0
    for size in [1,2,3,10,100]:
      for repeat in range(100):
        base=rng.integers(-3,4,size=size).astype('f4');values=rng.integers(-4,5,size=size).astype('f4');indices=np.arange(size)
        actual=fn(base,indices,values)
        for index,value,rank in zip(indices,values,actual):
            changed=base.copy();changed[index]=value
            expected=np.flatnonzero(np.argsort(-changed,kind='stable')==index)[0]+1
            assert rank==expected
            checks+=1
    rows=[]
    for ds in ['wands','esci']:
      p=products(ds);pi={str(x['product_id']):i for i,x in enumerate(p)}
      q=pd.read_csv(ROOT/f'phase2/data/processed/{ds}_queries.csv');qi={int(x):i for i,x in enumerate(q.query_id)}
      for model in ['minilm','bge_base','gte_modernbert']:
        prov=next(x for x in PROVENANCE['sources'] if x['dataset']==ds and x['model']==model)
        qv=np.load(ROOT/prov['query_embedding']);pv=np.load(ROOT/prov['product_embedding'])
        scores=np.asarray(qv,dtype='f4')@np.asarray(pv,dtype='f4').T
        target=pairread(ROOT/f'phase3/results/target_only_permutations/{ds}_{model}_pairs.parquet').set_index(['query_id','product_id']).sort_index()
        for s in STEMS:
            path=ROOT/f'phase2/results/phase2_pair_ranks/{ds}_{model}_native_{s}.parquet'
            raw=pairread(path).set_index(['query_id','product_id']).loc[target.index].reset_index()
            expected=target[s].to_numpy();actual=np.zeros(len(raw),dtype=int)
            max_score_error=0.
            for qid,g in raw.groupby('query_id',sort=False):
                indices=np.array([pi[x] for x in g.product_id]);base=scores[qi[int(qid)]]
                values=g.score.to_numpy(dtype='f4')
                if s=='C0':max_score_error=max(max_score_error,float(np.max(np.abs(base[indices]-values))));values=base[indices]
                actual[g.index]=fn(base,indices,values)
            mismatches=int(np.sum(actual!=expected))
            rows.append(dict(dataset=ds,model=model,schedule=s,pairs=len(raw),reconstructed_rank_mismatches=mismatches,
              max_rank_difference=int(np.max(np.abs(actual-expected))),max_C0_score_error=max_score_error,
              rank_source=path.relative_to(ROOT).as_posix(),fixed_competitors='raw C0',original_entry_removed=True))
        print('RECONSTRUCTED TARGET CHECK',ds,model,flush=True)
    f=pd.DataFrame(rows);f.to_csv(DATA/'target_rank_reconstruction_audit.csv',index=False)
    dump(DATA/'target_rank_algorithm_tests.json',dict(brute_force_tied_score_replacements=checks,all_passed=True,
      saved_function_source=source(ROOT/'phase3/scripts/analyze_target_only_permutations.py'),
      reconstructed_mismatches=int(f.reconstructed_rank_mismatches.sum()),
      interpretation='Saved ranks remain authoritative; any reconstruction mismatch is disclosed, never silently substituted.',
      reconstruction_sources=[source(ROOT/x[k]) for x in PROVENANCE['sources'] for k in ['query_embedding','product_embedding']]))

if __name__=='__main__':main()
