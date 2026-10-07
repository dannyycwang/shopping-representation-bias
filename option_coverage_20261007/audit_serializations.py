"""Verify schedule/text provenance without encoding or retrieving anything."""
import gzip
import hashlib
import importlib.util
import json
import pandas as pd
from analyze import ROOT, HERE, source, read, dump, sha, INPUTS


def main():
    spec=importlib.util.spec_from_file_location("frozen_representation",source("phase2/src/representations.py"))
    reps=importlib.util.module_from_spec(spec);spec.loader.exec_module(reps)
    cfg=read("phase2/config/phase2.json")
    with gzip.open(source("phase2/data/processed/wands_products.jsonl.gz"),"rt",encoding="utf-8") as f:products=list(map(json.loads,f))
    profiles=pd.read_csv(source("revision_evidence_20260917/data/encoder_profiles_and_rank_sources.csv")).query("dataset=='wands' and model=='bge_base'").set_index("schedule")
    audit=[]
    for condition in cfg["primary_variant_family"]:
        schedule=reps.slug(condition)
        path=source(f"phase2/data/representations/wands/{schedule}.jsonl.gz")
        digest=hashlib.sha256();n=0
        with gzip.open(path,"rt",encoding="utf-8") as f:
            for n,line in enumerate(f,1):
                digest.update(line.encode("utf-8"));row=json.loads(line);p=products[n-1]
                assert row["product_id"]==p["product_id"]
                assert row["text"]==reps.build(p,condition,cfg["attribute_permutation_seeds"])
        assert n==len(products)
        meta=read(str(__import__('pathlib').Path(profiles.loc[schedule,"product_embedding"]).with_suffix(".json")))
        assert digest.hexdigest()==meta["source_sha256"]
        audit.append(dict(schedule=schedule,products=n,source_sha256=digest.hexdigest(),exact_text_and_order_match=True))
        print('Verified raw text',schedule,flush=True)
    for rule in ["lexical_ascending","lexical_descending"]:
        s=read(f"revision_evidence_20260917/experiments/wands_bge_base_{rule}_source.json")
        digest=hashlib.sha256()
        for i,p in enumerate(products):
            attrs=sorted(p["attributes"],key=lambda v:(v.casefold(),v),reverse=rule=="lexical_descending")
            if i:digest.update(b"\0")
            digest.update(reps._plain(p,attrs).encode("utf-8"))
        assert digest.hexdigest()==s["text_source_sha256"]
        audit.append(dict(schedule=rule,products=len(products),source_sha256=digest.hexdigest(),exact_text_and_order_match=True))
    q=pd.read_csv(source("phase2/data/processed/wands_queries.csv"))
    qm=read(str(__import__('pathlib').Path(profiles.iloc[0].query_embedding).with_suffix(".json")))
    assert hashlib.sha256("\0".join(q['query'].fillna('').astype(str)).encode()).hexdigest()==qm['source_sha256']
    for path in ["phase2/scripts/run_dense.py","phase2/src/evaluation.py"]:source(path)
    dump("data/serialization_audit.json",dict(schedules=audit,query_source_matches=True,
         input_hashes=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(INPUTS)],
         inference_performed=False,retrieval_performed=False))


if __name__=="__main__": main()
