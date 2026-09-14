from common5 import *
import datetime
path=CONFIG/'sample.json'
if path.exists():raise SystemExit('Frozen sample already exists; refusing overwrite')
splits=json.loads((P4/'config/splits.json').read_text())
result={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selection':'highest-label-eligible query IDs ordered by SHA256 phase5-sample:{dataset}:{query_id}; first 8','datasets':{},'noise_products':[],'model':{'id':MODEL_ID,'revision':MODEL_REV,'dtype':'float16'},'primary_decoding':{'do_sample':False,'max_new_tokens':512,'seed':20260907},'noise_decoding':{'do_sample':True,'temperature':.7,'top_p':.9,'max_new_tokens':512,'seeds':[20260981,20260982,20260983]},'methods':list(prompt_sources()),'stems':STEMS,'primary_contrasts':['G1-Raw','CG1-G1','CG1-C','G2-Raw','CG2-G2','CG2-C']}
all_products=[]
for ds,candidates in [('wands',splits['wands_heldout']),('esci',splits['esci_old'])]:
    p,q,j=load(ds);highest='Exact' if ds=='wands' else 'E';eligible=set(map(int,j[j.label.eq(highest)].query_id.unique()));ordered=[x for x in stable_ids(candidates,f'phase5-sample:{ds}') if x in eligible];ids=ordered[:8]
    pairs=j[j.query_id.isin(ids)&j.label.eq(highest)][['query_id','product_id','label']].copy();products=sorted(set(map(str,pairs.product_id)))
    source=P2/'data/processed'/f'{ds}_products.jsonl.gz'
    result['datasets'][ds]={'query_ids':ids,'eligible_queries':len(ids),'query_product_pairs':len(pairs),'unique_products':len(products),'product_ids':products,'products_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    all_products += [(ds,pid) for pid in products]
result['noise_products']=[{'dataset':d,'product_id':p} for d,p in sorted(all_products,key=lambda x:sha(f'phase5-noise:{x[0]}:{x[1]}'))[:8]]
path.write_text(json.dumps(result,indent=2),encoding='utf8')
protocol=ROOT/'OPTIMIZATION_ROBUSTNESS_PROTOCOL.md';(CONFIG/'protocol.sha256').write_text(hashlib.sha256(protocol.read_bytes()).hexdigest())
print(json.dumps({d:{k:v for k,v in x.items() if k not in ['product_ids','products_sha256']} for d,x in result['datasets'].items()},indent=2))
