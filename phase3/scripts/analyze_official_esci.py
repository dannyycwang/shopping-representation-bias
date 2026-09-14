from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
P2 = ROOT / "phase2"
OUT = ROOT / "phase3" / "results"
MODELS = ["minilm", "bge_base", "gte_modernbert"]
STEMS = ["C0", "C1", "C2s1", "C2s2", "C2s3", "C2s4", "C2s5"]


def dcg(values: np.ndarray, k: int) -> float:
    x = np.asarray(values[:k], dtype=float)
    return float(np.sum(x / np.log2(np.arange(2, len(x) + 2)))) if len(x) else 0.0


def main() -> None:
    table_dir = OUT / "phase3_tables"; query_dir = OUT / "phase3_per_query"
    table_dir.mkdir(parents=True, exist_ok=True); query_dir.mkdir(parents=True, exist_ok=True)
    summary=[]
    for model in MODELS:
        frames=[]
        for stem in STEMS:
            path=P2/"results"/"phase2_pair_ranks"/f"esci_{model}_native_{stem}.parquet"
            d=pd.read_parquet(path)[["query_id","product_id","label","grade","score"]]
            # Official formulation: rank only the judged candidates supplied for each query.
            d=d.sort_values(["query_id","score","product_id"],ascending=[True,False,True],kind="stable")
            d[stem]=d.groupby("query_id").cumcount()+1
            frames.append(d[["query_id","product_id","label","grade",stem]])
        wide=frames[0]
        for d in frames[1:]: wide=wide.merge(d,on=["query_id","product_id","label","grade"],validate="one_to_one")
        ranks=wide[STEMS].to_numpy()
        for k in [10,20,50]: wide[f"VI@{k}"]=((ranks<=k).any(1)&(ranks>k).any(1)).astype(float)
        rows=[]
        for qid,g in wide.groupby("query_id",sort=True):
            ideal=np.sort(g.grade.to_numpy(float))[::-1]
            original=g.sort_values(["C0","product_id"],kind="stable")
            e=g[g.label.eq("E")]
            row={"query_id":qid,"candidates":len(g)}
            for k in [10,20,50]:
                denom=dcg(ideal,k)
                row[f"NDCG@{k}"]=dcg(original.grade.to_numpy(float),k)/denom if denom else np.nan
                row[f"ERecall@{k}"]=float((e.C0<=k).mean()) if len(e) else np.nan
                row[f"VI@{k}_query_macro"]=float(e[f"VI@{k}"].mean()) if len(e) else np.nan
            rows.append(row)
        pq=pd.DataFrame(rows); pq.to_csv(query_dir/f"esci_official_pool_{model}.csv",index=False)
        e=wide[wide.label.eq("E")]
        summary.append({"dataset":"ESCI Official Candidate Pool","retriever":model,"queries":len(pq),
                        "median_candidates":float(pq.candidates.median()),"NDCG@10":pq["NDCG@10"].mean(),
                        "NDCG@20":pq["NDCG@20"].mean(),"ERecall@20":pq["ERecall@20"].mean(),
                        "rank_range_median":float((e[STEMS].max(1)-e[STEMS].min(1)).median()),
                        "VI@10_micro":e["VI@10"].mean(),"VI@10_query_macro":pq["VI@10_query_macro"].mean(),
                        "VI@20_micro":e["VI@20"].mean(),"VI@20_query_macro":pq["VI@20_query_macro"].mean(),
                        "VI@50_micro":e["VI@50"].mean(),"VI@50_query_macro":pq["VI@50_query_macro"].mean()})
        wide.to_parquet(OUT/"phase3_audit"/f"esci_official_pool_{model}_pair_ranks.parquet",index=False)
    result=pd.DataFrame(summary)
    result.to_csv(table_dir/"esci_official_candidate_pool.csv",index=False)
    print(result.to_string(index=False))


if __name__=="__main__": main()

