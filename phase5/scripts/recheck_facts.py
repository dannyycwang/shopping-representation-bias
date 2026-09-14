from common5 import *
import datetime
src=OUT/'generations.jsonl';rows=[];maps={d:record_map(d) for d in json.loads((CONFIG/'sample.json').read_text())['datasets']}
for line in src.open(encoding='utf8'):
    x=json.loads(line);r=maps[x['dataset']][x['product_id']];source=canonical_text(r) if x['stem']=='canonical' else raw_text(r,x['stem']);x['fact_check']=fact_check(r,source,x['output'],x['finish_reason']);rows.append(x)
tmp=OUT/'generations.rechecked.jsonl'
with tmp.open('w',encoding='utf8') as f:
    for x in rows:f.write(json.dumps(x,ensure_ascii=False)+'\n')
tmp.replace(src)
(OUT/'fact_recheck.json').write_text(json.dumps({'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':len(rows),'checker':'phase5 common5.py frozen rules; recomputed after generation to include source conflict and coverage fields'},indent=2),encoding='utf8')
print('rechecked',len(rows))
