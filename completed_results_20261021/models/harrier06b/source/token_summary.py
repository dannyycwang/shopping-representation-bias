"""Export complete token audits independently of unfinished vector checkpoints."""
from common import *

def main():
    design=protocol();summaries=[]
    for model,spec in design['models'].items():
        for ds in ['wands','esci']:
            base=HERE/'runs'/model/ds/'raw'
            if not (base/'plan.json').exists():continue
            plan=read(base/'plan.json');lengths=np.load(base/'unique_lengths.npy');counts=lengths[np.asarray(plan['aliases'])]
            products,queries,j=support(ds);n=len(products);assert len(counts)==7*n
            counts=counts.reshape(7,n).T
            f=pd.DataFrame(counts,columns=['tokens_'+s for s in STEMS]);f.insert(0,'product_id',[str(p['product_id']) for p in products])
            f['max_tokens_seven']=counts.max(1);f['all_seven_fitting']=f.max_tokens_seven<=spec['max_length'];f['effective_max_length']=spec['max_length'];f['native_supported_length']=spec['native_supported_length']
            path=HERE/f'audit/{model}_{ds}_token_audit.parquet';f.to_parquet(path,index=False)
            h=j[j.label.eq(highest(ds))].merge(f[['product_id','all_seven_fitting']],on='product_id',validate='many_to_one')
            summaries.append(dict(model=model,dataset=ds,status='complete_token_audit_only',catalog_products=n,
                max_untruncated_tokens=int(counts.max()),all_seven_fitting_products=int(f.all_seven_fitting.sum()),
                competitors_truncated_under_C0=int((counts[:,0]>spec['max_length']).sum()),
                highest_pairs=len(h),fitting_highest_pairs=int(h.all_seven_fitting.sum()),fitting_pair_fraction=float(h.all_seven_fitting.mean()),
                eligible_queries=len(queries),fitting_queries=int(h[h.all_seven_fitting].query_id.nunique()),
                effective_max_length=spec['max_length'],native_supported_length=spec['native_supported_length'],
                input_plan_sha256=sha(base/'plan.json'),lengths_sha256=sha(base/'unique_lengths.npy'),output_sha256=sha(path)))
    pd.DataFrame(summaries).to_csv(HERE/'tables/token_fit_summary.csv',index=False)
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':main()
