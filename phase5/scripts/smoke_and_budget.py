from common5 import *
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

splits=json.loads((P4/'config/splits.json').read_text())
budget={}
for ds,ids in [('wands',splits['wands_heldout']),('esci',splits['esci_old'])]:
    p,q,j=load(ds);highest='Exact' if ds=='wands' else 'E';counts=j[j.label.eq(highest)].groupby('query_id').size()
    ordered=stable_ids(ids,f'phase5-sample:{ds}')
    budget[ds]={'first_30':[[x,int(counts.get(x,0))] for x in ordered[:30]],'cumulative':{str(n):int(sum(counts.get(x,0) for x in ordered[:n])) for n in [5,6,8,10,12,20,30]}}
(OUT/'sample_budget.json').write_text(json.dumps(budget,indent=2),encoding='utf8')

tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REV)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=MODEL_REV,torch_dtype=torch.float16,device_map='cuda').eval()
prompts=prompt_sources();p,q,j=load('wands');dev=stable_ids(splits['wands_dev'],'phase5-smoke:wands')[:2];highest=j[(j.query_id.isin(dev))&j.label.eq('Exact')]
rm={str(x['product_id']):x for x in p};rows=[]
for pid in list(dict.fromkeys(map(str,highest.product_id)))[:2]:
    rec=rm[pid]
    for method,template in prompts.items():
        source=raw_text(rec,'C0');user=template.format(description=source);messages=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':user}]
        text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);inputs=tok(text,return_tensors='pt').to('cuda');torch.manual_seed(20260907);start=time.time()
        with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=256,do_sample=False,pad_token_id=tok.eos_token_id)
        answer=tok.decode(out[0,inputs.input_ids.shape[1]:],skip_special_tokens=True).strip();fc=fact_check(rec,source,answer,'length' if out.shape[1]-inputs.input_ids.shape[1]>=256 else 'ok')
        rows.append({'query_ids':dev,'product_id':pid,'method':method,'input_tokens':int(inputs.input_ids.shape[1]),'output_tokens':int(out.shape[1]-inputs.input_ids.shape[1]),'seconds':time.time()-start,'output':answer,'fact_check':fc})
(OUT/'development_smoke.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps({'budget':budget,'smoke':[{'product_id':r['product_id'],'method':r['method'],'input_tokens':r['input_tokens'],'output_tokens':r['output_tokens'],'seconds':r['seconds'],'fact_pass':r['fact_check']['pass']} for r in rows]},indent=2))
