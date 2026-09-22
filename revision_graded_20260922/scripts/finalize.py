"""Hash deliverables and create an archive without changing historical inputs."""
from common import *
from datetime import datetime,timezone
import zipfile

def main():
    verify_protocol();validation=read(HERE/'qa/validation.json')
    assert validation['all_122_conditions_have_exact_frozen_support'] and validation['preexisting_working_changes_preserved']
    assert validation['input_hash_records_preserved']==469
    states=pd.read_csv(HERE/'tables/canonical_hybrid_state_contrasts.csv')
    assert len(states)==720
    # Every fixed-versus-fixed VI difference is structurally zero.
    fixed=states[(states.metric=='VI')&~states.reference.eq('raw_hybrid')]
    assert np.all(fixed[['delta','ci_low','ci_high']].to_numpy()==0)
    for records in [read(HERE/'data/input_manifest.json')['preexisting_files'],read(HERE/'qa/preexisting_tracked_changes.json')]:
        for path,digest in records.items():assert sha(ROOT/path)==digest
    (HERE/'qa/final_git_status.txt').write_text(git('status','--short','--branch')+'\n',encoding='utf8')
    exclude={'OUTPUT_HASHES.json','DELIVERY.json',HERE.name+'.zip',HERE.name+'.zip.sha256'}
    files=[p for p in HERE.rglob('*') if p.is_file() and p.name not in exclude and '__pycache__' not in p.parts]
    outputs=[dict(path=p.relative_to(HERE).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]
    dump(HERE/'OUTPUT_HASHES.json',outputs)
    dump(HERE/'DELIVERY.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),status='P0/P1 complete; optional P2 deferred',conditions=122,per_query_records=49227,
        populations=read(HERE/'PROTOCOL.json')['populations'],missing_required_payloads=[],encoding_performed=False,training_performed=False,historical_files_modified=False,manuscript_integrated=False,
        protocol_sha256=sha(HERE/'PROTOCOL.json'),output_manifest_sha256=sha(HERE/'OUTPUT_HASHES.json'),output_files=len(outputs),validation=validation,
        bootstrap=dict(draws=DRAWS,seed=SEED,unit='query cluster',interval='uncorrected 95% percentile'),
        start_here=['README.md','GRADED_METRIC_AUDIT.md','CLAIM_VERDICTS.md','MANUSCRIPT_INSERTIONS.md','REPRODUCE.md']))
    archive=HERE/(HERE.name+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files+[HERE/'OUTPUT_HASHES.json',HERE/'DELIVERY.json']):z.write(p,arcname=HERE.name+'/'+p.relative_to(HERE).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
    print('DELIVERY',len(outputs),'files;',archive.stat().st_size,'archive bytes;',sha(archive),flush=True)

if __name__=='__main__':main()
