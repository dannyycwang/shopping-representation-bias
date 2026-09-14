"""Prepare the fixed main-experiment extreme cases for source inspection.

Can run while the repetition control is still encoding: that control is excluded
from the main qualitative sample. analyze.py later verifies/recreates selection.
"""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
all_frames=[]
for condition in ['R2','R3','R4','R5','R8']:
    for retriever in ['BM25','Dense','Hybrid']:
        a=pd.read_csv(ROOT/f'results/raw/R0_{retriever}_exact.csv')
        b=pd.read_csv(ROOT/f'results/raw/{condition}_{retriever}_exact.csv')
        z=a.merge(b,on=['query_id','product_id'],suffixes=('_original','_alternative'),validate='one_to_one')
        z['representation']=condition; z['retriever']=retriever
        z['rank_gain']=z.rank_original-z.rank_alternative
        all_frames.append(z)
selected=pd.concat(all_frames,ignore_index=True)
products=pd.read_csv(ROOT/'data/raw/product.csv',sep='\t').fillna('').set_index('product_id')
queries=pd.read_csv(ROOT/'data/raw/query.csv',sep='\t').set_index('query_id')['query']
examples=[]
for direction in ['positive','negative']:
    if direction=='positive':
        eligible=selected[(selected.rank_original>20)&(selected.rank_alternative<=20)].sort_values(['rank_gain','query_id','product_id'],ascending=[False,True,True])
    else:
        eligible=selected[(selected.rank_original<=20)&(selected.rank_alternative>20)].sort_values(['rank_gain','query_id','product_id'],ascending=[True,True,True])
    frame=eligible.drop_duplicates(['query_id','product_id']).head(20).copy(); frame['direction']=direction; examples.append(frame)
    with (ROOT/f'results/qualitative/review_pack_{direction}.jsonl').open('w',encoding='utf-8') as f:
        for index,row in enumerate(frame.to_dict('records'),1):
            product=products.loc[row['product_id']].to_dict()
            f.write(json.dumps({'number':index,**row,'query':queries.loc[row['query_id']],**product},ensure_ascii=False)+'\n')
pd.concat(examples).to_csv(ROOT/'results/qualitative/selected_examples.csv',index=False)
print('40 main-experiment cases prepared.')
