from common import *
import pyarrow.dataset as ds
from threadpoolctl import threadpool_limits
threadpool_limits(4)

def clean(v):return '' if pd.isna(v) else str(v)
def main():
    data=P4/'data';data.mkdir(exist_ok=True);pold,oldq,_=load('esci');split=json.loads((P4/'config/splits.json').read_text());newids=split['esci_new_fixed'];assert not set(newids)&set(oldq.query_id)
    base=P2/'data/esci_repo/shopping_queries_dataset'
    ex=pd.read_parquet(base/'shopping_queries_dataset_examples.parquet',filters=[('product_locale','==','us'),('small_version','==',1),('split','==','test')]);ex=ex[ex.query_id.isin(newids)];ids=sorted(set(ex.product_id)|{x['product_id'] for x in pold})
    tab=ds.dataset(base/'shopping_queries_dataset_products.parquet',format='parquet').to_table(filter=(ds.field('product_locale')=='us')&ds.field('product_id').isin(ids)).to_pandas();assert not tab.product_id.duplicated().any();products=tab.set_index('product_id');records=[]
    for pid in ids:
        row=products.loc[pid];a=[clean(row[k]) for k in ['product_brand','product_color','product_bullet_point'] if clean(row[k])];records.append({'product_id':pid,'title':clean(row.product_title),'class':'','category':'','description':clean(row.product_description),'attributes':a,'attribute_separator':'\n','section_order':['title','description','attributes']})
    with gzip.open(data/'esci_new_products.jsonl.gz','wt',encoding='utf8') as f:
        for p in records:f.write(json.dumps(p,ensure_ascii=False)+'\n')
    q=ex[['query_id','query']].drop_duplicates().sort_values('query_id');q['query_type']='unclassified';q.to_csv(data/'esci_new_queries.csv',index=False)
    j=ex[['query_id','product_id','esci_label']].rename(columns={'esci_label':'label'});assert not j.duplicated(['query_id','product_id']).any();j['grade']=j.label.map({'E':1,'S':.1,'C':.01,'I':0});j.to_csv(data/'esci_new_judgments.csv',index=False)
    (data/'catalog_manifest.json').write_text(json.dumps({'queries':len(q),'products':len(records),'judgments':len(j),'query_ids':list(map(int,q.query_id)),'catalog':'union of historical ESCI catalog plus all judged products for frozen 100 new queries','product_disjoint':False},indent=2))
    folder=OUT/'esci_new';folder.mkdir(exist_ok=True);spec=next(x for x in CFG['models'] if x['key']=='bge_base');enc=DenseEncoder(spec,CFG,OUT/'embeddings')
    from common import _plain
    np.save(folder/'bge_base_queries.npy',encode(enc,q['query'].tolist(),'esci_new_bge_base_queries'));np.save(folder/'bge_base_products.npy',encode(enc,[_plain(x) for x in records],'esci_new_bge_base_original'));enc.close();print('NEW QUERY CATALOG',len(q),len(records),flush=True)
if __name__=='__main__':main()
