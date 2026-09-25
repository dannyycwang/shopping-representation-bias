"""Make every saved method/rank condition and its hashes directly inspectable."""
from common import *

def main():
    initial=read(GRADED/'data/conditions.json')
    keycols=['dataset','model','method','schedule']
    lookup={tuple(c[k] for k in keycols):c for c in initial}
    observed=pd.read_parquet(GRADED/'data/per_query.parquet',columns=keycols+['rank_source']).drop_duplicates()
    conditions=[]
    for r in observed.to_dict('records'):
        existing=lookup.get(tuple(r[k] for k in keycols))
        c=dict(existing) if existing else {k:r[k] for k in keycols}
        if existing:assert c['pairs']==r['rank_source']
        c['pairs']=r['rank_source'];c['manifest']='data/conditions.json' if existing else 'data/per_query.parquet';conditions.append(c)
    frozen={r['path']:dict(r,manifest='revision_graded_20260922/data/input_manifest.json') for r in read(GRADED/'data/input_manifest.json')['files']}
    for r in read(GRADED/'OUTPUT_HASHES.json'):
        frozen.setdefault('revision_graded_20260922/'+r['path'],dict(r,manifest='revision_graded_20260922/OUTPUT_HASHES.json'))
    manifests={name:sha(GRADED/name) for name in ['data/conditions.json','data/per_query.parquet']}
    records=[]
    for c in conditions:
        path=ROOT/c['pairs'];actual=sha(path);expected=frozen.get(c['pairs'],{}).get('sha256','')
        record={k:c[k] for k in ['dataset','model','method','schedule']}
        record.update(rank_source=c['pairs'],rank_sha256=actual,bytes=path.stat().st_size,historical_manifest_sha256=expected,historical_hash_source=frozen.get(c['pairs'],{}).get('manifest','unavailable'),historical_hash_status='MATCH' if expected==actual else ('MISMATCH' if expected else 'not in earlier input/output manifests; current source hash recorded'),condition_manifest='revision_graded_20260922/'+c['manifest'],condition_manifest_sha256=manifests[c['manifest']],condition_record=json.dumps(c,ensure_ascii=False),graded_query_companion='data/effectiveness_per_query.parquet')
        embedding=c.get('embedding')
        record.update(embedding_source=embedding or 'not specified by this condition record',embedding_sha256='',embedding_metadata='',embedding_metadata_sha256='',recorded_model_revision='',recorded_precision='',recorded_runtime_versions='unavailable in condition record; not inferred')
        if embedding:
            ep=ROOT/embedding;record['embedding_sha256']=sha(ep)
            side=ep.with_suffix('.json')
            if side.exists():
                meta=read(side);record.update(embedding_metadata=side.relative_to(ROOT).as_posix(),embedding_metadata_sha256=sha(side),recorded_model_revision=meta.get('model',{}).get('revision','unavailable'),recorded_precision=meta.get('precision','unavailable'),recorded_runtime_versions=json.dumps(meta['versions']) if meta.get('versions') else 'unavailable in embedding sidecar')
        records.append(record)
    frame=pd.DataFrame(records);csv(frame,HERE/'data/method_rank_provenance.csv')
    dump(HERE/'qa/method_rank_provenance.json',dict(time_utc=now(),conditions=len(frame),historical_hash_status_counts=frame.historical_hash_status.value_counts().to_dict(),exact_source_manifest=sha(GRADED/'data/conditions.json'),note='Current full source hashes plus available sidecars; unrecorded runtime fields remain unavailable. New graded-derived rank outputs need not occur in the earlier input-only manifest.'))
    assert not frame.historical_hash_status.eq('MISMATCH').any(),frame[frame.historical_hash_status.eq('MISMATCH')].to_dict('records')
    print(frame.historical_hash_status.value_counts().to_string())

if __name__=='__main__':main()
