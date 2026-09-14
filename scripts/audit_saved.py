import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import pandas as pd
from src.representations import audit,FIELDS,build

p=pd.read_csv(ROOT/'data/raw/product.csv',sep='\t').fillna('').set_index('product_id')
records=p.to_dict('index'); rows=[]
for path in (ROOT/'data/representations').glob('*.jsonl'):
    if path.stem=='source_map': continue
    for line in path.open(encoding='utf-8'):
        item=json.loads(line); original=records[item['product_id']]
        result=audit(original,item['text']); result['matches_deterministic_generator']=item['text']==build(original,path.stem)
        if not result['pass'] or not result['matches_deterministic_generator']: raise ValueError((path.stem,item['product_id'],result))
        rows.append({'representation':path.stem,'product_id':item['product_id'],**result})
    print(path.stem,'verified',flush=True)
pd.DataFrame(rows).to_csv(ROOT/'results/tables/fidelity.csv',index=False)
