"""Existing case audit and at most two deterministic additional examples."""
import gzip
import json
import pandas as pd
from analyze import HERE, ROOT, load_records, dump, save_records
from core import selection_hash


def main():
    ms=[x for x in load_records("data/membership.jsonl") if x["comparison"]=="raw_C0_vs_C1"]
    fs=[x for x in load_records("data/query_fields.jsonl") if x["comparison"]=="raw_C0_vs_C1"]
    with gzip.open(HERE/"data/product_fields.jsonl.gz","rt",encoding="utf-8") as f:ex={(str(x['product_id']),x['field']):x for x in map(json.loads,f)}
    with gzip.open(ROOT/"phase2/data/processed/wands_products.jsonl.gz","rt",encoding="utf-8") as f:products={str(x['product_id']):x for x in map(json.loads,f)}
    existing=next(x for x in ms if x["query"].casefold()=="anti fatigue mat")
    selected=[dict(category="existing_anti_fatigue_mat",query_id=existing["query_id"],illustrating_field="shape")]
    candidates=[]
    for category,metric in [("additional_cancellation_option_loss","any_lost"),("additional_cancellation_no_field_option_change","unchanged_options")]:
        eligible=[r for r in fs if r['cancellation'] and r['eligible'] and r[metric] and r['query_id'] not in {s['query_id'] for s in selected}]
        qids=sorted({r['query_id'] for r in eligible},key=lambda q:selection_hash(f"20261007:case:{q}"))
        candidates.append(dict(category=category,query_ids_in_hash_order=qids))
        if qids:
            qid=qids[0]
            field=next(f for f in ['material','shape','color'] if any(r['query_id']==qid and r['field']==f for r in eligible))
            selected.append(dict(category=category,query_id=qid,illustrating_field=field))
    dump("data/case_selection.json",dict(rule="SHA256('20261007:case:' + query_id), ascending; existing query 231 and already selected query IDs excluded. Field priority material, shape, color. Categories are field-specific, not a claim of completeness across fields.",selected=selected,candidates=candidates))
    pair={s:pd.read_parquet(ROOT/f'phase2/results/phase2_pair_ranks/wands_bge_base_native_{s}.parquet') for s in ['C0','C1']}
    for p in pair.values():p.product_id=p.product_id.astype(str)
    records=[]
    md=['# Case audit', '', 'All cases use WANDS / BGE native / catalog-wide C0 versus C1 / K=20 / Exact judgments. Additional cases use the frozen query-ID hash and are field-specific. They were not chosen for maximum effect. Full source attributes, all relevant candidate IDs and ranks are in `data/cases.jsonl`; candidate pools and hash selection are in `data/case_selection.json`.', '']
    for s in selected:
        m=next(x for x in ms if x['query_id']==s['query_id']);qid=m['query_id']
        fields=[x for x in fs if x['query_id']==qid]
        union=sorted(set(m['before'])|set(m['after']))
        catalog=[]
        for pid in union:
            ranks={name:int(p[(p.query_id==qid)&p.product_id.eq(pid)].iloc[0]['rank']) for name,p in pair.items()}
            catalog.append(dict(product=products[pid],ranks=ranks,exact_judgment=True,in_before=pid in m['before'],in_after=pid in m['after'],
                                extracted_fields={field:ex[pid,field] for field in ['material','shape','color']}))
        records.append(dict(selection=s,membership=m,query_fields=fields,source_products=catalog))
        md += [f"## Query {qid}: {m['query']}", '', f"Selection: {s['category']}; illustrating field: **{s['illustrating_field']}**.", '',
               f"Relevant candidates {m['n_before']} -> {m['n_after']}; lost/gained products {m['n_lost_products']}/{m['n_gained_products']}; Exact pool {m['highest_n']}. Recall@20 {m['recall_before']:.9f} -> {m['recall_after']:.9f}. Full-ranking nDCG@20 {m['ndcg_before']:.9f} -> {m['ndcg_after']:.9f} (3/1/0 gains; equality is reported only where observed).", '',
               f"Before IDs: {', '.join(m['before'])}.", '', f"After IDs: {', '.join(m['after'])}.", '',
               f"Departing IDs: {', '.join(m['lost_products'])}. Entering IDs: {', '.join(m['gained_products'])}.", '',
               '| Field | Strict eligible | Before options | After options | Lost | Gained |', '|---|---|---|---|---|---|']
        for r in fields:
            if r['eligible']:
                md.append(f"| {r['field']} | yes | {', '.join(r['options_before'])} | {', '.join(r['options_after'])} | {', '.join(r['lost_options']) or 'none'} | {', '.join(r['gained_options']) or 'none'} |")
            else:md.append(f"| {r['field']} | no: {', '.join(r['exclusion_reasons'])} | unknown | unknown | not assessed | not assessed |")
        md += ['', '| Product ID | Source title | C0 rank | C1 rank | Status |', '|---|---|---:|---:|---|']
        for p in catalog:
            pid=str(p['product']['product_id'])
            status='departing' if pid in m['lost_products'] else 'entering' if pid in m['gained_products'] else 'retained'
            title=p['product']['title'].replace('|',' / ')
            md.append(f"| {pid} | {title} | {p['ranks']['C0']} | {p['ranks']['C1']} | {status} |")
        md.append('')
        if s['category']=='existing_anti_fatigue_mat':
            anette=next(p for p in catalog if 'anette salon' in p['product']['title'].casefold())
            supplier={}
            for field in ['material','shape','color']:
                vals=anette['extracted_fields'][field]['values']
                supplier[field]={v:[pid for pid in m['after'] if ex[pid,field]['status']=='valid' and v in ex[pid,field]['values']] for v in vals}
            dump('data/anti_fatigue_remaining_suppliers.json',supplier)
            md += ["The departing Anette Salon product is ID 19170, recorded as `material : foam`, `shape : semi-circle`, `color : black`. The full relevant Top-20 comparison verifies loss of the recorded **semi-circle shape**: all after-products have valid shape metadata and none supplies it. Foam remains supplied by " + ', '.join(supplier['material']['foam']) + ". Black remains explicitly supplied by " + ', '.join(supplier['color']['black']) + ".", '',
                   "Color is strictly ineligible: ID 20172 has conflicting colors, and departing ID 20175 has no generic color key. The known black suppliers establish that black is still represented, but incomplete color metadata do not establish any complete color-set comparison. Both full nDCG@20 values are 1 (within floating representation) because all 20 retrieved candidates are Exact; this is a verified fact about this case, not a cancellation requirement.", '',
                   "Thus the existing two-row substitution illustration alone was insufficient, but this full-list audit supports a narrow recorded-shape disappearance for query 231. It does not establish a utility loss, physical preference satisfaction, or loss of foam/black options.", '']
    save_records('data/cases.jsonl',records)
    (HERE/'CASE_AUDIT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps(selected,indent=2))


if __name__=='__main__':main()
