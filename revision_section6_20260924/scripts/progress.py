"""Read-only, resumable inference progress; estimates are not benchmark timings."""
from common import *

for model in ['minilm','bge_base']:
    folder=HERE/'harmonized'/model
    if (folder/'complete.json').exists():
        done=read(folder/'complete.json')
        print(model,'COMPLETE',done['unique_vectors'],'unique inputs;',done['changed_decision_checks']+24,'repeat checks; identical =',done['all_repeated_inputs_identical'])
        continue
    if not (folder/'inputs.parquet').exists():
        print(model,'preparing inputs or not started')
        continue
    inputs=pd.read_parquet(folder/'inputs.parquet',columns=['bucket'])
    blocks=[read(path) for path in folder.glob('block_*.json')]
    n=sum(b['rows'] for b in blocks)
    print(model,f'{n}/{len(inputs)} ({100*n/len(inputs):.1f}%) encoded')
    if any(b['bucket']==512 for b in blocks):
        recent=[b for b in blocks if b['bucket']==512][-8:]
        rate=sum(b['seconds'] for b in recent)/sum(b['rows'] for b in recent)
        print('approximate remaining encoding minutes:',round((len(inputs)-n)*rate/60,1),'(recent 512-token blocks; excludes final scoring/QA)')
