from common import *
import datetime
config=P4/'config';p,q,j=load('wands');_,eq,_=load('esci')
order=sorted(map(int,q.query_id),key=lambda x:hashlib.sha256(f'phase3-dev:{x}'.encode()).hexdigest());dev=order[:len(order)//5]
raw=P2/'data/esci_repo/shopping_queries_dataset/shopping_queries_dataset_examples.parquet'
examples=pd.read_parquet(raw,filters=[('product_locale','==','us'),('small_version','==',1),('split','==','test')],columns=['query_id'])
unused=set(map(int,examples.query_id))-set(map(int,eq.query_id));fresh=sorted(unused,key=lambda x:hashlib.sha256(f'20260907-prospective:{x}'.encode()).hexdigest())[:100]
record={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wands_dev':dev,'wands_heldout':[int(x) for x in q.query_id if x not in dev],'esci_old':list(map(int,eq.query_id)),'esci_new_fixed':fresh,'new_query_status':'Absent from Phase II/III processed query IDs; prior dataset access known, not globally pristine study','rules':RULES,'view_seeds':SEEDS}
path=config/'splits.json'
assert not path.exists(),'Do not overwrite frozen IDs'
path.write_text(json.dumps(record,indent=2));(config/'protocol.sha256').write_text(hashlib.sha256((config/'EXPERIMENT_PROTOCOL.frozen.md').read_bytes()).hexdigest())
print('Frozen',len(dev),'development,',len(q)-len(dev),'held-out and',len(fresh),'new ESCI query IDs')
