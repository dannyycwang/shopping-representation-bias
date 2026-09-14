from common import *
from threadpoolctl import threadpool_limits
threadpool_limits(4)
for ds in ['wands','esci']:
    p,q,j=load(ds);qv,pv=original_cache(ds,'bge_base');enc=DenseEncoder(next(x for x in CFG['models'] if x['key']=='bge_base'),CFG,OUT/'embeddings');views=[]
    for i in range(1,8):views.append(encode(enc,texts(p,f'view{i}'),f'{ds}_bge_base_view{i}'))
    c=norm(np.mean(views,axis=0));mx=np.maximum.reduce([qv@v.T for v in views])
    for method,score in [('centroid_7',qv@c.T),('max_7',mx)]:evaluate(score,p,q,j,ds,f'bge_base_{method}')
    enc.close();print('SATURATION COMPLETE',ds,flush=True)
