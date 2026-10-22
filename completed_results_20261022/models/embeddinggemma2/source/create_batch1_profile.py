"""Create an isolated batch=1 retry after the recorded CUDA OOM, without overwrites."""
from common import *
import shutil

def main():
    original=protocol();destination=HERE/'profiles/batch1_20261017';assert not destination.exists(),'Never overwrite an existing profile'
    destination.mkdir(parents=True);(destination/'scripts').mkdir();(destination/'audit').mkdir();(destination/'tables').mkdir()
    for p in (HERE/'scripts').glob('*.py'):
        text=p.read_text(encoding='utf8')
        if p.name=='common.py':text=text.replace('ROOT = HERE.parent','ROOT = Path(__file__).resolve().parents[4]')
        if p.name=='supervise.py':text=text.replace('revision_recent_models_20261016/scripts/','revision_recent_models_20261016/profiles/batch1_20261017/scripts/')
        (destination/'scripts'/p.name).write_text(text,encoding='utf8')
    shutil.copytree(HERE/'tests',destination/'tests',ignore=shutil.ignore_patterns('__pycache__'))
    for p in (HERE/'audit').glob('*'):
        if p.is_file() and (p.name in ['preflight.json','preexisting_changes.json','initial_git_status.txt','attachments.json','implementation_sources.json','download_qwen3_4b.json','download_embeddinggemma2.json','historical_data_hash_reconciliation.json'] or p.suffix=='.csv'):
            shutil.copyfile(p,destination/'audit'/p.name)
    shutil.copytree(HERE/'audit/models',destination/'audit/models')
    for name in ['SOURCE_MAP.md','MODEL_AUDIT.md','PROTOCOL.md','requirements.txt']:
        shutil.copyfile(HERE/name,destination/name)
    # Qwen's adapter bytes, execution profile and technical evidence are unchanged.
    shutil.copytree(HERE/'smoke/qwen3_4b',destination/'smoke/qwen3_4b')
    model=original['models']['embeddinggemma2']
    original['models']['embeddinggemma2']={**model,'batching':{**model['batching'],'base':1,'limits':[[model['max_length'],1]]}}
    original['frozen_utc']=now();original['parent_protocol_sha256']=sha(HERE/'PROTOCOL.json')
    original['execution_deviation']=dict(id='batch1_20261017',reason='Original EmbeddingGemma 2/ESCI CUDA out-of-memory after 896 completed batches; lower batch before considering length/backend changes',
        evidence=source(HERE/'audit/embeddinggemma2_esci_encode.log'),technical_memory_probe=source(HERE/'audit/memory_probe_sdpa_1_False.json'),
        changes={'embeddinggemma2.batch_size':'all inputs = 1'},unchanged=['models and real revisions','native lengths','BF16','SDPA','pooling and projection','prefixes','full catalogs','queries and labels','schedules and seeds','all evaluation and analysis rules'],
        formal_scores_seen=False,cache_policy='Fresh queries and every catalog view; no original profile vectors reused',qwen_technical_evidence='Reused only because Qwen profile fingerprint and adapter bytes are identical; no formal query/catalog vectors reused')
    original['source_files'] += [source(p) for p in (destination/'scripts').glob('*.py')]+[source(p) for p in (destination/'tests').glob('*.py')]+[source(HERE/'PROTOCOL.json')]
    dump(destination/'PROTOCOL.json',original);(destination/'PROTOCOL.sha256').write_text(sha(destination/'PROTOCOL.json')+'\n')
    dump(destination/'PROFILE_DEVIATION.json',original['execution_deviation'])
    dump(HERE/'audit/active_profile.json',dict(profile='batch1_20261017',path=str(destination),status='technical_validation_pending',parent_cache_preserved=True,created_utc=now()))
    command='revision_recent_models_20261016/profiles/batch1_20261017/scripts/'
    (destination/'RUN_COMMANDS.md').write_text('# Isolated batch=1 retry\n\nUse the original revision venv and run from repository root. No vectors are shared with the original failed Gemma profile.\n\n```powershell\n$RecentPy = (Resolve-Path revision_recent_models_20261016/.venv/Scripts/python.exe).Path\n$env:PYTHONIOENCODING = \'utf-8\'\n$env:PYTHONDONTWRITEBYTECODE = \'1\'\n& $RecentPy '+command+'smoke.py --model embeddinggemma2\n& $RecentPy -u '+command+'supervise.py\n```\n\nThe supervisor runs one GPU model at a time, with stage logs and complete flags. All formal rankings and summaries remain unavailable until complete catalog checkpoints pass. Parent RUN_COMMANDS.md documents dependencies and the initial failed attempt.\n',encoding='utf8')
    print(destination,flush=True)

if __name__=='__main__':main()
