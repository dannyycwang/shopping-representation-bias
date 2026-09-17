"""Encoder-specific, untruncated all-order token counts and pure-order audit."""
from common import *
from common import _random_order
from collections import Counter
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'

def main():
    from transformers import AutoTokenizer
    tokrows=[]; preservation=[]; canonical=[]; sources=[]
    for ds in ['wands','esci']:
        p=products(ds); ids=[str(x['product_id']) for x in p]
        # Validate source serialization and every frozen transformation, product by product.
        distinct=[set() for _ in p]
        for stem,condition in zip(STEMS,CFG['primary_variant_family']):
            r=reps(ds,stem)
            assert [str(x['product_id']) for x in r]==ids
            for i,(x,y) in enumerate(zip(p,r)):
                expected=build(x,condition,CFG['attribute_permutation_seeds'])
                assert expected==y['text'],(ds,stem,ids[i],'raw serialization')
                distinct[i].add(hashlib.sha256(expected.encode()).digest())
            sources.append(source(ROOT/f'phase2/data/representations/{ds}/{stem}.jsonl.gz'))
        for x,seen in zip(p,distinct):
            attrs=x['attributes']; inputs=[attrs,list(reversed(attrs))]+[_random_order(attrs,str(x['product_id']),s) for s in CFG['attribute_permutation_seeds']]
            for rule in RULES:
                expected=canonical_text(x,rule)
                assert Counter(canonical_attrs(x,rule))==Counter(attrs)
                for incoming in inputs:
                    assert Counter(incoming)==Counter(attrs)
                    assert canonical_text({**x,'attributes':incoming},rule)==expected
            preservation.append(dict(dataset=ds,product_id=str(x['product_id']),attribute_count=len(attrs),
              duplicate_entry_occurrences=len(attrs)-len(set(attrs)),distinct_serializations_seven=len(seen),
              raw_seven_exact=True,all_three_canonical_rules_invariant=True))
        for rule in RULES:
            texts=[canonical_text(x,rule) for x in p]
            digest=hashlib.sha256('\0'.join(texts).encode()).hexdigest()
            canonical.append(dict(dataset=ds,rule=rule,products=len(p),incoming_schedules_verified=7,
                                  all_text_equal=True,duplicates_preserved=True,text_source_sha256=digest))
        for spec in CFG['models']:
            tok=AutoTokenizer.from_pretrained(spec['name'],revision=spec['revision'],local_files_only=True)
            # Actual historical product and query prefixes are empty (encoder source and config).
            lengths=np.zeros((len(p),7),dtype=np.int32)
            for n,s in enumerate(STEMS):
                rr=reps(ds,s)
                for start in range(0,len(rr),256):
                    text=[x['text'] for x in rr[start:start+256]]
                    got=tok(text,add_special_tokens=True,truncation=False,padding=False,verbose=False)['input_ids']
                    lengths[start:start+len(text),n]=[len(x) for x in got]
                print('TOKENS',ds,spec['key'],s,flush=True)
            df=pd.DataFrame(lengths,columns=[f'tokens_{s}' for s in STEMS])
            df.insert(0,'product_id',ids);df.insert(0,'model',spec['key']);df.insert(0,'dataset',ds)
            df['max_tokens_seven']=lengths.max(1);df['tokenizer_cap']=spec['max_tokens']
            df['fully_fits']=df.max_tokens_seven<=spec['max_tokens'];df['special_token_overhead']=tok.num_special_tokens_to_add(pair=False)
            tokrows.append(df)
            cache=Path.home()/'.cache/huggingface/hub'/('models--'+spec['name'].replace('/','--'))/'snapshots'/spec['revision']
            for name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','sentence_bert_config.json']:
                f=cache/name
                if f.exists():sources.append(dict(path=str(f),sha256=sha(f),bytes=f.stat().st_size,role='pinned tokenizer/config'))
    lengths=pd.concat(tokrows,ignore_index=True)
    lengths.to_csv(DATA/'all_seven_token_lengths.csv',index=False)
    pd.DataFrame(preservation).to_csv(DATA/'serialization_integrity.csv',index=False)
    pd.DataFrame(canonical).to_csv(DATA/'canonical_serialization_audit.csv',index=False)
    pd.DataFrame(lengths.groupby(['dataset','model']).agg(products=('product_id','size'),fully_fitting_products=('fully_fits','sum'),max_observed_tokens=('max_tokens_seven','max'))).reset_index().to_csv(DATA/'fitting_catalog_counts.csv',index=False)
    case=lengths[(lengths.dataset=='wands')&(lengths.model=='minilm')&(lengths.product_id=='34536')].iloc[0]
    target=pairread(ROOT/'phase3/results/target_only_permutations/wands_minilm_pairs.parquet')
    row=target[(target.query_id==162)&(target.product_id=='34536')].iloc[0]
    assert row.C2s4==5 and row.C2s1==1527
    dump(DATA/'introduction_case_audit.json',dict(dataset='wands',query_id=162,product_id='34536',query='turquoise chair',highest_label='Exact',
      model=next(x for x in CFG['models'] if x['key']=='minilm'),profile='native',
      target_ranks={s:int(row[s]) for s in STEMS},full_catalog_ranks={s:int(row['whole_'+s]) for s in STEMS},
      token_counts_including_special={s:int(case['tokens_'+s]) for s in STEMS},fully_fits_all_orders=bool(case.fully_fits),
      cap=256,product_prefix='',special_token_overhead=int(case.special_token_overhead),
      competitors='Full 42,994-product raw C0 catalog, target old entry removed independently.',
      source=source(ROOT/'phase3/results/target_only_permutations/wands_minilm_pairs.parquet'),
      interpretation='Illustrative post-hoc target-only case. Do not label this as fully fitting unless fully_fits_all_orders is true.'))
    dump(DATA/'tokenization_sources.json',dict(sources=sources,models=CFG['models'],schedules=STEMS,product_prefix='',query_prefix='',
      truncation=False,add_special_tokens=True,padding=False,filter_scope='Analyzed target pairs only; full competitor catalog retained.',
      historical_BGE_mask='phase4/results/{dataset}_product_features.csv was BGE-only at 512; never applied to MiniLM or GTE here.',
      outputs=[source(DATA/n) for n in ['all_seven_token_lengths.csv','serialization_integrity.csv','canonical_serialization_audit.csv','introduction_case_audit.json']]))

if __name__=='__main__':main()
