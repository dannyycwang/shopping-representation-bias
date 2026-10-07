"""Bounded post-hoc analysis of saved full-catalog Top-20 lists only."""
import argparse
import collections
import gzip
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from core import membership, options, validate_ranking

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
INPUTS = set()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""): h.update(block)
    return h.hexdigest()


def source(path):
    path=ROOT/path
    INPUTS.add(path)
    if not path.exists(): raise FileNotFoundError(path)
    return path


def read(path): return json.loads(source(path).read_text(encoding="utf-8"))
def dump(path, obj):
    path=HERE/path
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+"\n",encoding="utf-8")


def save_records(path, rows):
    with (HERE/path).open("w",encoding="utf-8") as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")


def load_records(path):
    with (HERE/path).open(encoding="utf-8") as f:return list(map(json.loads,f))


def git(*args):return subprocess.check_output(["git",*args],cwd=ROOT).decode("utf-8").strip()


def snapshot():
    path=DATA/"preexisting_state.json"
    if path.exists(): return
    changed=git("diff","--name-only").splitlines()
    # Include untracked manuscript work; every file already present remains protected.
    protected={ROOT/p for p in changed if (ROOT/p).is_file()}
    protected.update(p for p in (ROOT/"paper_www2027").rglob("*") if p.is_file())
    dump("data/preexisting_state.json",dict(commit=git("rev-parse","HEAD"),branch=git("branch","--show-current"),
         status=git("status","--short"), files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in sorted(protected)]))


def verify():
    snapshot()
    protocol=read(CFG["support_protocol"])
    assert sha(source(CFG["support_protocol"]))==source("revision_graded_20260922/PROTOCOL.sha256").read_text().strip()
    population=protocol["populations"]["wands"]
    ids=population["eligible_ids"]
    with gzip.open(source(CFG["catalog"]),"rt",encoding="utf-8") as f: products=list(map(json.loads,f))
    catalog=[str(x["product_id"]) for x in products]
    assert len(catalog)==len(set(catalog))==population["catalog_size"]
    q=pd.read_csv(source(CFG["queries"]))
    j=pd.read_csv(source(CFG["qrels"]),dtype={"product_id":str})
    assert not j.duplicated(["query_id","product_id"]).any()
    assert set(j.product_id)<=set(catalog) and set(j.query_id)<=set(q.query_id)
    raw=pd.read_csv(source("data/raw/label.csv"),sep="\t",dtype={"product_id":str})
    conflict=raw.groupby(["query_id","product_id"]).label.nunique()
    bad=set(conflict[conflict>1].index)
    clean=raw[~raw.set_index(["query_id","product_id"]).index.isin(bad)].drop_duplicates(["query_id","product_id"])
    cols=["query_id","product_id","label"]
    pd.testing.assert_frame_equal(clean[cols].sort_values(cols[:2]).reset_index(drop=True),j[cols].sort_values(cols[:2]).reset_index(drop=True))
    split=read(CFG["splits"])
    derived=sorted(set(split["wands_heldout"])&set(j.loc[j.label.eq("Exact"),"query_id"]))
    assert ids==derived
    assert j[j.query_id.isin(ids)&j.label.eq("Exact")].shape[0]==population["highest_label_pairs"]
    ca=pd.read_csv(source(CFG["catalog_axis"]),dtype={"catalog_id":str})
    qa=pd.read_csv(source(CFG["query_axis"]))
    assert ca["index"].tolist()==list(range(len(catalog))) and ca.catalog_id.tolist()==catalog
    assert qa.queries_id.tolist()==q.query_id.tolist()
    highest={int(k):set(g.product_id) for k,g in j[j.label.eq("Exact")].groupby("query_id")}
    labels={int(k):dict(zip(g.product_id,g.label)) for k,g in j.groupby("query_id")}
    conditions=[x for x in read(CFG["conditions"]) if x["dataset"]=="wands" and x["model"]=="bge_base"]
    saved=pd.read_parquet(source(CFG["saved_metrics"]))
    saved=saved[saved.model.eq("bge_base")]
    profiles=pd.read_csv(source("revision_evidence_20260917/data/encoder_profiles_and_rank_sources.csv")).query("dataset=='wands' and model=='bge_base'")
    assert profiles.query_embedding.nunique()==1
    raws=[]
    for row in profiles.to_dict("records"):
        meta=read(str(Path(row["product_embedding"]).with_suffix(".json")))
        raws.append(meta)
    for key in ["model","profile","implementation","precision","seed"]:
        assert all(x[key]==raws[0][key] for x in raws)
    query_meta=read(str(Path(profiles.iloc[0].query_embedding).with_suffix(".json")))
    assert query_meta["model"]==raws[0]["model"] and query_meta["profile"]==raws[0]["profile"]
    canonical_sources={rule:read(f"revision_evidence_20260917/experiments/wands_bge_base_{rule}_source.json") for rule in CFG["canonical_rules"]}
    metas=[read(s["product_embedding_metadata"]["relative_path"]) for s in canonical_sources.values()]
    for key in ["model","profile","implementation","precision","seed","attention_implementation"]:
        assert metas[0][key]==metas[1][key]
    for s in canonical_sources.values():
        assert s["query_ids"]==q.query_id.tolist() and s["catalog_size"]==len(catalog)
        assert s["query_embedding"]["relative_path"]==profiles.iloc[0].query_embedding
        # An absent optional canonical rank file is reported by the run inventory below.
        if (ROOT/s["rank_source"]["relative_path"]).exists():
            assert sha(source(s["rank_source"]["relative_path"]))==s["rank_source"]["sha256"]
    addendum=read("revision_evidence_20260917/BGE_EXECUTION_ADDENDUM.json")
    hybrid_sources=read("revision_graded_20260922/data/canonical_hybrid_sources.json")
    hs=[x for x in hybrid_sources if x["dataset"]=="wands" and x["model"]=="bge_base" and x["method"].endswith(tuple(CFG["canonical_rules"]))]
    assert len(hs)==2 and all(x["full_catalog"] for x in hs)
    assert hs[0]["bm25_full_rank_sha256"]==hs[1]["bm25_full_rank_sha256"]
    assert hs[0]["query_embedding_sha256"]==hs[1]["query_embedding_sha256"]==canonical_sources["lexical_ascending"]["query_embedding"]["sha256"]
    for path in ["phase2/config/phase2.json","phase2/scripts/prepare_data.py","phase2/src/representations.py",
                 "revision_graded_20260922/scripts/evaluate.py","revision_graded_20260922/scripts/common.py",
                 "revision_evidence_20260917/experiments/EXECUTION_NOTES.md","data/raw/product.csv",
                 "revision_evidence_20260917/CANCELLATION_TABLE.md","revision_graded_20260922/qa/canonical_input_invariance.csv"]:
        source(path)
    runs={}; runinfo=[];missing=[];checkrows=[]
    plan=[]
    for schedule in CFG["raw_schedules"]:
        c=next(x for x in conditions if x["method"]=="raw" and x["schedule"]==schedule)
        plan.append((schedule,"raw",schedule,c["top"],c["pairs"]))
    for rule in CFG["canonical_rules"]:
        c=next(x for x in conditions if x["method"]==rule)
        plan.append((rule,rule,"fixed",f"revision_graded_20260922/data/wands_bge_base_{rule}_top1000.npz",c["pairs"]))
        m="canonical_hybrid_"+rule
        plan.append((m,m,"fixed",f"revision_graded_20260922/data/wands_bge_base_{m}_top1000.npz",f"revision_graded_20260922/data/wands_bge_base_{m}_pairs.parquet"))
    for name,method,schedule,topfile,pairfile in plan:
        absent=[p for p in [topfile,pairfile] if not (ROOT/p).exists()]
        if absent:
            missing.append(dict(run=name,missing_artifacts=absent));continue
        pairs=pd.read_parquet(source(pairfile));pairs.product_id=pairs.product_id.astype(str)
        assert not pairs.duplicated(["query_id","product_id"]).any()
        pd.testing.assert_frame_equal(pairs[cols].sort_values(cols[:2]).reset_index(drop=True),j[cols].sort_values(cols[:2]).reset_index(drop=True))
        if topfile.endswith(".parquet"):
            top=pd.read_parquet(source(topfile));top.product_id=top.product_id.astype(str)
            assert set(top.query_id)==set(q.query_id)
            assert not top.duplicated(["query_id","product_id"]).any()
            for _,g in top.groupby("query_id"):
                assert sorted(g["rank"])==list(range(1,len(g)+1))
            top=top[top["rank"]<=CFG["k"]]
            rankings={int(k):g.sort_values("rank").product_id.tolist() for k,g in top.groupby("query_id") if k in ids}
        else:
            z=np.load(source(topfile),allow_pickle=False)
            assert z["query_ids"].tolist()==ids
            ix=z["catalog_indices"]
            assert ix.shape[1]>=CFG["k"] and ix.min()>=0 and ix.max()<len(catalog)
            rankings={int(qid):[catalog[int(i)] for i in ix[n,:CFG["k"]]] for n,qid in enumerate(ids)}
        metrics=saved[saved.method.eq(method)&saved.schedule.eq(schedule)].set_index("query_id")
        assert set(metrics.index)==set(ids)
        judged_top=pairs[pairs["rank"]<=CFG["k"]].groupby("query_id")
        for qid in ids:
            ranking=rankings[qid];validate_ranking(ranking,catalog,CFG["k"])
            actual={p:i+1 for i,p in enumerate(ranking) if p in labels[qid]}
            expected=dict(zip(judged_top.get_group(qid).product_id,judged_top.get_group(qid)["rank"])) if qid in judged_top.groups else {}
            assert actual==expected,(name,qid,"judged rank disagreement")
            included=set(ranking)&highest[qid]
            recall=len(included)/len(highest[qid])
            assert abs(recall-metrics.loc[qid,"Recall@20"])<1e-14
            gains={"Exact":3.,"Partial":1.,"Irrelevant":0.}
            discount=1/np.log2(np.arange(2,CFG["k"]+2))
            dcg=sum(gains.get(labels[qid].get(p),0)*discount[i] for i,p in enumerate(ranking))
            ideal=sorted((gains[v] for v in labels[qid].values()),reverse=True)[:CFG["k"]]
            ndcg=dcg/float(np.dot(ideal,discount[:len(ideal)]))
            assert abs(ndcg-metrics.loc[qid,"nDCG@20"])<1e-14
            checkrows.append(dict(run=name,query_id=qid,recall=recall,ndcg=ndcg,top20=ranking))
        if name.startswith("canonical_hybrid_"):
            h=next(x for x in hs if x["method"]==name)
            assert sha(source(pairfile))==h["pairs_sha256"]
        runs[name]=rankings
        runinfo.append(dict(run=name,method=method,schedule=schedule,top_path=topfile,pairs_path=pairfile,queries=len(rankings),k=CFG["k"],catalog_size=len(catalog)))
    comparisons=[(f"raw_C0_vs_{s}","C0",s) for s in CFG["raw_schedules"][1:]]
    comparisons += [("canonical_dense","lexical_ascending","lexical_descending"),("canonical_hybrid","canonical_hybrid_lexical_ascending","canonical_hybrid_lexical_descending")]
    mrows=[]
    score={(x["run"],x["query_id"]):x for x in checkrows}
    for comp,before,after in comparisons:
        if before not in runs or after not in runs:
            missing.append(dict(comparison=comp,reason="required saved ranking unavailable"));continue
        for qid in ids:
            row=membership(runs[before][qid],runs[after][qid],highest[qid])
            row.update(comparison=comp,query_id=qid,query=q.set_index("query_id").loc[qid,"query"],before_run=before,after_run=after,
                       highest_n=len(highest[qid]),recall_before=score[before,qid]["recall"],recall_after=score[after,qid]["recall"],
                       ndcg_before=score[before,qid]["ndcg"],ndcg_after=score[after,qid]["ndcg"])
            row["delta_recall"]=row["recall_after"]-row["recall_before"]
            row["delta_ndcg"]=row["ndcg_after"]-row["ndcg_before"]
            mrows.append(row)
    save_records("data/membership.jsonl",mrows)
    save_records("data/validated_top20.jsonl",checkrows)
    dump("data/available_runs.json",runinfo)
    primary=[x for x in mrows if x["comparison"]=="raw_C0_vs_C1"]
    anchors=dict(evaluation_queries=len(primary),primary_cancellation_queries=sum(x["cancellation"] for x in primary))
    provenance=dict(actual_anchors=anchors,expected_check_only=CFG["anchors_for_checks_only"],anchors_match=anchors==CFG["anchors_for_checks_only"],
        catalog_size=len(catalog),heldout_queries=len(split["wands_heldout"]),no_exact_heldout=len(split["wands_heldout"])-len(ids),
        eligible_queries=len(ids),highest_pairs=population["highest_label_pairs"],raw_conflicting_qrel_pairs=len(bad),
        raw_duplicate_rows_removed=len(raw)-sum(raw.set_index(cols[:2]).index.isin(bad))-len(clean),
        unknown_top20_ids=0,duplicate_top20_ids=0,incomplete_top20_lists=0,judged_rank_mismatches=0,
        recall_saved_mismatches=0,ndcg_saved_mismatches=0,validated_query_run_records=len(checkrows),
        raw_profile=raws[0],canonical_profiles=metas,
        canonical_execution_caveat="Ascending inherited configured ceiling 48; descending effective ceiling 16 under frozen addendum. Scientific settings, encoder implementation, precision, query vectors and catalog match; numerical forward execution is not identical. Strict batch-matched saved canonical pair unavailable. Comparisons describe these fixed artifacts, not isolated causal sorting effects.",
        batching_addendum=addendum,missing_artifacts=missing,
        not_saved_or_unavailable=["Strict identical-batch ascending/descending canonical rankings (both dense and derived hybrid); not required for the descriptive saved-artifact contrast, required to isolate sorting from execution roundoff."],
        tie_rule="descending stored float32 scores then ascending original catalog index; saved ranks authoritative",
        gain_mapping={"Exact":3,"Partial":1,"Irrelevant":0},unjudged_policy="Unknown relevance; zero gain only for full-rank nDCG calculation",
        bootstrap_seed=CFG["seed"],bootstrap_draws=CFG["bootstrap_draws"])
    dump("data/provenance_checks.json",provenance)
    dump("data/input_manifest.json",dict(commit=git("rev-parse","HEAD"),branch=git("branch","--show-current"),
         runtime=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__),
         configuration=CFG,inputs=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(INPUTS)],
         missing_artifacts=missing))
    print(json.dumps({k:provenance[k] for k in ["actual_anchors","anchors_match","validated_query_run_records","missing_artifacts","canonical_execution_caveat"]},indent=2))
    for comp,g in pd.DataFrame(mrows).groupby("comparison"):
        print(comp,len(g),'changed',g.membership_changed.sum(),'cancellation',g.cancellation.sum())


def boot(values, seed=CFG["seed"]):
    a=np.asarray(values,dtype=float)
    if a.ndim==1:a=a[:,None]
    if not len(a):return None,None,None
    rng=np.random.default_rng(seed)
    draws=np.concatenate([a[rng.integers(0,len(a),(min(250,CFG["bootstrap_draws"]-i),len(a)))].mean(axis=1)
                          for i in range(0,CFG["bootstrap_draws"],250)],axis=0)
    lo,hi=np.quantile(draws,[.025,.975],axis=0)
    return a.mean(axis=0),lo,hi


def analysis():
    manifest=json.loads((DATA/"input_manifest.json").read_text(encoding="utf-8"))
    for x in manifest["inputs"]:assert sha(ROOT/x["path"])==x["sha256"],x["path"]
    freeze=json.loads((DATA/"extraction_freeze.json").read_text())
    assert sha(HERE/"extraction_spec_v1.json")==freeze["spec_sha256"]
    audit=json.loads((DATA/"attribute_audit_review.json").read_text(encoding="utf-8"))
    assert audit["reviewed_records"]==90 and audit["extraction_errors"]==0
    with gzip.open(DATA/"product_fields.jsonl.gz","rt",encoding="utf-8") as f: extraction=list(map(json.loads,f))
    fields=list(json.loads((HERE/"extraction_spec_v1.json").read_text())["fields"])
    byfield={field:{str(x["product_id"]):x for x in extraction if x["field"]==field} for field in fields}
    ms=load_records("data/membership.jsonl")
    rows=[]
    for m in ms:
        for field in fields:
            o=options(m["before"],m["after"],byfield[field])
            reverse=options(m["after"],m["before"],byfield[field])
            assert reverse["eligible"]==o["eligible"]
            if o["eligible"]:
                assert reverse["lost_options"]==o["gained_options"]
                assert m["membership_changed"] or not o["option_changed"]
            rows.append(dict(comparison=m["comparison"],query_id=m["query_id"],field=field,cancellation=m["cancellation"],
                             membership_changed=m["membership_changed"],n_before=m["n_before"],n_after=m["n_after"],**o))
    save_records("data/query_fields.jsonl",rows)
    save_records("data/exclusions.jsonl",[r for r in rows if not r["eligible"]])
    metrics=["option_changed","any_lost","any_gained","unchanged_options","n_lost","n_gained","n_options_before","n_options_after","net_options"]
    summaries=[]; exclusion_summary=[]; coverage=[]
    for comp in sorted({r["comparison"] for r in rows}):
        for field in fields:
            for subgroup in ["all","cancellation"]:
                group=[r for r in rows if r["comparison"]==comp and r["field"]==field and (subgroup=="all" or r["cancellation"])]
                eligible=[r for r in group if r["eligible"]]
                means,lo,hi=boot([[r[m] for m in metrics] for r in eligible])
                result=dict(comparison=comp,field=field,subgroup=subgroup,total_evaluation_queries=sum(m["comparison"]==comp for m in ms),
                            total_cancellation_queries=sum(m["comparison"]==comp and m["cancellation"] for m in ms),
                            subgroup_total=len(group),eligible_n=len(eligible),excluded_n=len(group)-len(eligible),
                            tiny_support=len(eligible)<CFG["tiny_support_threshold"])
                for i,metric in enumerate(metrics):
                    result[metric+"_mean"]=float(means[i]) if means is not None else None
                    result[metric+"_ci_low"]=float(lo[i]) if lo is not None else None
                    result[metric+"_ci_high"]=float(hi[i]) if hi is not None else None
                    result[metric+"_degenerate"]=bool(lo[i]==hi[i]) if lo is not None else None
                    if metric in metrics[:4]:result[metric+"_count"]=sum(r[metric] for r in eligible)
                summaries.append(result)
                combos=collections.Counter(";".join(r["exclusion_reasons"]) for r in group if not r["eligible"])
                for reason,n in sorted(combos.items()):
                    exclusion_summary.append(dict(comparison=comp,field=field,subgroup=subgroup,reasons=reason,n=n,denominator=len(group),category="mutually_exclusive_combination"))
                reasons=collections.Counter(x for r in group for x in r["exclusion_reasons"])
                for reason,n in sorted(reasons.items()):
                    exclusion_summary.append(dict(comparison=comp,field=field,subgroup=subgroup,reasons=reason,n=n,denominator=len(group),category="overlapping_reason"))
                for population,subset in [("all",group),("excluded",[r for r in group if not r["eligible"]])]:
                    for side in ["before","after"]:
                        denomin=sum(r["n_"+side] for r in subset)
                        known=sum(r["known_"+side] for r in subset)
                        proportions=[r["known_fraction_"+side] for r in subset if r["known_fraction_"+side] is not None]
                        coverage.append(dict(comparison=comp,field=field,subgroup=subgroup,population=population,setting=side,
                          query_n=len(subset),empty_relevant_queries=sum(r["n_"+side]==0 for r in subset),
                          queries_with_missing=sum(r["missing_"+side]>0 for r in subset),candidate_occurrences=denomin,
                          known_occurrences=known,unknown_occurrences=denomin-known,known_fraction_micro=known/denomin if denomin else None,
                          known_fraction_query_macro=float(np.mean(proportions)) if proportions else None))
    pd.DataFrame(summaries).to_csv(DATA/"option_summary.csv",index=False)
    pd.DataFrame(exclusion_summary).to_csv(DATA/"exclusion_summary.csv",index=False)
    pd.DataFrame(coverage).to_csv(DATA/"coverage_summary.csv",index=False)
    summary=[]
    for comp in sorted({m["comparison"] for m in ms}):
        group=[m for m in ms if m["comparison"]==comp]
        names=["recall_before","recall_after","delta_recall","ndcg_before","ndcg_after","delta_ndcg","n_lost_products","n_gained_products"]
        mean,lo,hi=boot([[r[k] for k in names] for r in group])
        row=dict(comparison=comp,n=len(group),membership_changed=sum(r["membership_changed"] for r in group),
                 unchanged_membership=sum(not r["membership_changed"] for r in group),cancellation=sum(r["cancellation"] for r in group),
                 changed_non_cancellation=sum(r["membership_changed"] and not r["cancellation"] for r in group),
                 recall_increased=sum(r["delta_recall"]>0 for r in group),recall_decreased=sum(r["delta_recall"]<0 for r in group),
                 recall_equal=sum(r["recall_equal"] for r in group),empty_before=sum(r["n_before"]==0 for r in group),empty_after=sum(r["n_after"]==0 for r in group))
        for i,k in enumerate(names): row.update({k+"_mean":mean[i],k+"_ci_low":lo[i],k+"_ci_high":hi[i]})
        summary.append(row)
    pd.DataFrame(summary).to_csv(DATA/"membership_summary.csv",index=False)
    # No pooled query-contrast denominator: a common-support query has a vector of all contrasts.
    joint=[]
    comps=["raw_C0_vs_"+s for s in CFG["raw_schedules"][1:]]
    for field in fields:
        index={(r["comparison"],r["query_id"]):r for r in rows if r["field"]==field}
        common=sorted(set.intersection(*[{q for (c,q),r in index.items() if c==comp and r["eligible"] and r["cancellation"]} for comp in comps]))
        a=np.array([[index[c,q]["any_lost"] for c in comps] for q in common],dtype=float)
        if len(common):
            v=np.column_stack([a,a[:,1:].mean(axis=1),a[:,1:].mean(axis=1)-a[:,0]])
            mean,lo,hi=boot(v)
            for i,label in enumerate(comps+["five_shuffle_mean_per_query","five_shuffle_mean_minus_reversal"]):
                joint.append(dict(field=field,contrast=label,common_query_n=len(common),query_ids=json.dumps(common),mean=mean[i],ci_low=lo[i],ci_high=hi[i],tiny_support=len(common)<20,degenerate=lo[i]==hi[i]))
        else:joint.append(dict(field=field,contrast="all_six_common_complete_cancellation_support",common_query_n=0,query_ids="[]"))
    pd.DataFrame(joint).to_csv(DATA/"joint_query_bootstrap.csv",index=False)
    print(pd.DataFrame(summaries).query("subgroup=='cancellation'")[["comparison","field","subgroup_total","eligible_n","any_lost_count","any_gained_count","unchanged_options_count","n_lost_mean","n_gained_mean"]].to_string(index=False))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["verify","analyze"]);a=p.parse_args()
    verify() if a.stage=="verify" else analysis()
