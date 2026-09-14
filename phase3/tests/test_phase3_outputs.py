from pathlib import Path

import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[2]


def test_ecommerce_retriever_has_nontrivial_instability_on_both_datasets():
    d=pd.read_csv(ROOT/"phase3/results/phase3_tables/ecommerce_retriever_sensitivity.csv")
    assert set(d.dataset)=={"wands","esci"}
    assert (d["VI@20_micro"]>.03).all()
    assert (d["VI@20_macro_ci_low"]>.01).all()


def test_reranker_outputs_and_candidate_loss():
    rows=[]
    for dataset in ["wands","esci"]:
        row=pd.read_csv(ROOT/f"phase3/results/phase3_tables/{dataset}_reranker_pipeline.csv").iloc[0]
        rows.append(row)
        assert row["Irrecoverable@100_micro_all_highest"]>0
        assert row["common_pairs_rerank_VI@20"]<row["common_pairs_dense_VI@20"]
    assert rows[0]["Irrecoverable@100_micro_all_highest"]>rows[1]["Irrecoverable@100_micro_all_highest"]


def test_invariant_scores_and_alignment():
    d=pd.read_csv(ROOT/"phase3/results/phase3_tables/invariant_methods.csv")
    assert len(d)==8
    assert (d.filter(regex=r"^VI@").to_numpy()==0).all()
    assert d.max_abs_score_diff.max()<1e-6
    pareto=pd.read_csv(ROOT/"phase3/results/phase3_tables/robustness_relevance_pareto.csv")
    chosen=pareto[pareto.method.eq("set_mean")]
    assert set(chosen.dataset)=={"wands","esci"}
    assert chosen.aligned_success.all()
