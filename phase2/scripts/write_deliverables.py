from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


P2 = Path(__file__).resolve().parents[1]
TABLES = P2 / "results/phase2_tables"
FIGURES = P2 / "results/phase2_figures"


def pct(value) -> str:
    return "" if pd.isna(value) else f"{100 * float(value):.2f}%"


def num(value, digits: int = 4) -> str:
    return "" if pd.isna(value) else f"{float(value):.{digits}f}"


def markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str, object]]) -> str:
    header = "| " + " | ".join(label for _, label, _ in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, rule]
    for _, row in frame.iterrows():
        values = []
        for key, _, formatter in columns:
            value = row[key]
            rendered = formatter(value) if callable(formatter) else str(value)
            values.append(str(rendered).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_products(dataset: str) -> dict[str, dict]:
    with gzip.open(P2 / f"data/processed/{dataset}_products.jsonl.gz", "rt", encoding="utf-8") as handle:
        return {str(item["product_id"]): item for item in map(json.loads, handle)}


def build_secondary(models: list[str], datasets: list[str]) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        for model in models:
            base = pd.read_csv(P2 / f"results/phase2_per_query/{dataset}_{model}_native_C0.csv")
            for stem in ["C0", "C3", "C4", "C5asc", "C5desc"]:
                frame = pd.read_csv(P2 / f"results/phase2_per_query/{dataset}_{model}_native_{stem}.csv")
                rows.append({
                    "dataset": dataset, "retriever": model, "representation": stem,
                    "cNDCG@10": frame["cNDCG@10"].mean(),
                    "delta_cNDCG@10": frame["cNDCG@10"].mean() - base["cNDCG@10"].mean(),
                    "HighestRecall@20": frame["HighestRecall@20"].mean(),
                    "delta_HighestRecall@20": (
                        frame["HighestRecall@20"].mean() - base["HighestRecall@20"].mean()
                    ),
                })
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "secondary_representation_controls_native.csv", index=False)
    return result


def build_controls() -> pd.DataFrame:
    frames = []
    for profile in ["native", "mean_126_o32", "mean_254_o0", "mean_254_o64", "cls_254_o0"]:
        frame = pd.read_csv(TABLES / f"table1_representation_sensitivity_{profile}.csv")
        frames.append(frame[frame.dataset.eq("wands") & frame.retriever.isin(["minilm", "bge_base"])])
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(TABLES / "chunk_pooling_controls.csv", index=False)
    return result


def build_failures(models: list[str], datasets: list[str]) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        products = load_products(dataset)
        queries = pd.read_csv(P2 / f"data/processed/{dataset}_queries.csv").set_index("query_id")
        candidates = []
        for model in models:
            for method in ["M1_factual_field_sentence", "M2_normalized_attributes"]:
                short = "M1C0" if method.startswith("M1") else "M2C0"
                frame = pd.read_parquet(
                    P2 / f"results/phase2_single_product/{dataset}_{model}_native_{short}.parquet"
                )
                frame = frame[frame.is_highest].assign(retriever=model, representation=method)
                candidates.append(frame)
        frame = pd.concat(candidates, ignore_index=True).sort_values(
            ["base_to_single_delta", "query_id", "product_id"], ascending=[True, True, True]
        ).head(5)
        for row in frame.itertuples(index=False):
            product = products[str(row.product_id)]
            rows.append({
                "dataset": dataset, "retriever": row.retriever,
                "representation": row.representation, "query_id": row.query_id,
                "query": queries.loc[int(row.query_id), "query"],
                "product_id": row.product_id, "product_title": product["title"],
                "base_rank": row.base_rank, "single_product_rank": row.single_product_rank,
                "full_catalog_rank": row.full_catalog_rank,
                "base_to_single_delta": row.base_to_single_delta,
            })
    result = pd.DataFrame(rows)
    result.to_csv(P2 / "results/phase2_qualitative/failure_cases.csv", index=False)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    cfg = json.loads((P2 / "config/phase2.json").read_text())
    models = [item["key"] for item in cfg["models"]]
    datasets = ["wands", "esci"]
    stats = {item["dataset"]: item for item in json.loads((P2 / "results/dataset_statistics.json").read_text())}
    table1 = pd.read_csv(TABLES / "table1_representation_sensitivity_native.csv")
    query_types = pd.read_csv(TABLES / "table2_query_type_native.csv")
    strata = pd.read_csv(TABLES / "relevance_stratified_native.csv")
    contrasts = pd.read_csv(TABLES / "relevance_stratum_contrasts_native.csv")
    table3 = pd.read_csv(TABLES / "table3_relevance_visibility_alignment_native.csv")
    table4 = pd.read_csv(TABLES / "table4_mitigation_native.csv")
    sar = pd.read_csv(TABLES / "stability_adjusted_relevance_native.csv")
    direct = pd.read_csv(TABLES / "single_product_summary_native.csv")
    gate = json.loads((TABLES / "phase2a_gate_native.json").read_text())
    secondary = build_secondary(models, datasets)
    controls = build_controls()
    failures = build_failures(models, datasets)
    examples = pd.read_csv(P2 / "results/phase2_qualitative/strong_examples.csv")

    control_pass = controls.assign(
        pass_cell=(controls["VI@20_micro"] >= cfg["go_thresholds"]["minimum_micro_vi20"])
        & (controls["VI@20_macro_ci_low"] > cfg["go_thresholds"]["minimum_query_macro_vi20_ci_low"])
    )
    nonnative = control_pass[control_pass.profile.ne("native")]
    controls_ok = all(
        nonnative[nonnative.retriever.eq(model)].pass_cell.sum() >= 2
        for model in ["minilm", "bge_base"]
    )
    nonpathological = query_types[~query_types.query_type.isin(["brand_entity_proxy", "style"])]
    broad_query_ok = all(
        (nonpathological[nonpathological.dataset.eq(dataset)]
         .groupby("query_type")["VI@20_query_macro"].mean() >= 0.03).sum() >= 2
        for dataset in datasets
    )
    mitigation_ok = all(
        table4[table4.dataset.eq(dataset)].groupby("representation")["mitigation_success"].sum().max() >= 2
        for dataset in datasets
    )
    high_ok = all(table1.groupby("dataset")["VI@20_micro"].mean() >= 0.03)
    go_checks = {
        "two_models_two_datasets": bool(gate["phase2a_pass"]),
        "reasonable_chunk_pool_controls": bool(controls_ok),
        "broad_query_types": bool(broad_query_ok),
        "highest_relevance_affected": bool(high_ok),
        "mitigation_success_on_both_datasets": bool(mitigation_ok),
    }
    overall_go = all(go_checks.values())

    t1_display = table1.copy()
    t1_display["CI"] = t1_display.apply(
        lambda row: f"[{pct(row['VI@20_macro_ci_low'])}, {pct(row['VI@20_macro_ci_high'])}]", axis=1
    )
    t1_md = markdown_table(t1_display, [
        ("dataset", "Dataset", str.upper), ("retriever", "Retriever", str),
        ("rank_range_median", "Median rank range", lambda value: f"{value:.0f}"),
        ("pairwise_crossing@10_micro", "Pairwise crossing@10", pct),
        ("pairwise_crossing@20_micro", "Pairwise crossing@20", pct),
        ("pairwise_crossing@50_micro", "Pairwise crossing@50", pct),
        ("VI@20_micro", "VI@20", pct), ("VI@20_query_macro", "Query-macro VI@20", pct),
        ("CI", "95% query-bootstrap CI", str),
    ])
    controls_md = markdown_table(control_pass, [
        ("retriever", "Retriever", str), ("profile", "Profile", str),
        ("rank_range_median", "Median range", lambda value: f"{value:.0f}"),
        ("VI@20_micro", "VI@20", pct), ("VI@20_macro_ci_low", "CI low", pct),
        ("pass_cell", "Pass magnitude gate", lambda value: "yes" if value else "no"),
    ])
    secondary_md = markdown_table(secondary[secondary.representation.ne("C0")], [
        ("dataset", "Dataset", str.upper), ("retriever", "Retriever", str),
        ("representation", "Control", str), ("delta_cNDCG@10", "Delta cNDCG@10", num),
        ("delta_HighestRecall@20", "Delta highest recall@20", pct),
    ])
    t3_md = markdown_table(table3, [
        ("dataset", "Dataset", str.upper), ("representation", "Representation", str),
        ("retriever", "Retriever", str), ("cNDCG@10", "cNDCG@10", num),
        ("HighestRecall@20", "Highest recall@20", pct),
        ("RelevantHidden@20", "Highest hidden@20", pct), ("VI@20_query_macro", "VI@20", pct),
    ])
    t4_md = markdown_table(table4, [
        ("dataset", "Dataset", str.upper), ("retriever", "Retriever", str),
        ("representation", "Representation", str),
        ("delta_cNDCG_at_10", "Delta cNDCG@10", num),
        ("ci_low_cNDCG_at_10", "cNDCG CI low", num),
        ("ci_high_cNDCG_at_10", "cNDCG CI high", num),
        ("delta_VI_at_20", "Delta VI@20", pct),
        ("holm_p_VI_at_20", "Holm p (VI)", num),
        ("mitigation_success", "Success", lambda value: "yes" if value else "no"),
    ])
    direct_high = direct[direct.subset.eq("highest")]
    direct_md = markdown_table(direct_high, [
        ("dataset", "Dataset", str.upper), ("retriever", "Retriever", str),
        ("representation", "Representation", str), ("pairs", "Pairs", lambda value: f"{value:.0f}"),
        ("single_rank_delta_median", "Median direct rank gain", lambda value: f"{value:.0f}"),
        ("single_improved_fraction", "Direct improved", pct),
        ("single_crossing@20", "Direct crossing@20", pct),
        ("full_crossing@20", "Full-catalog crossing@20", pct),
    ])

    strongest = query_types.sort_values("VI@20_query_macro", ascending=False).head(12)
    query_md = markdown_table(strongest, [
        ("dataset", "Dataset", str.upper), ("retriever", "Retriever", str),
        ("query_type", "Query type", str), ("queries", "Queries", lambda value: f"{value:.0f}"),
        ("VI@20_query_macro", "VI@20", pct),
        ("rank_range_median", "Median range", lambda value: f"{value:.0f}"),
    ])
    contrasts_md = markdown_table(contrasts, [
        ("dataset", "Dataset", str.upper), ("retriever", "Retriever", str),
        ("contrast", "Contrast", str), ("queries", "Paired queries", lambda value: f"{value:.0f}"),
        ("delta", "Delta VI@20", pct), ("ci_low", "CI low", pct),
        ("ci_high", "CI high", pct), ("holm_within_dataset", "Holm p", num),
    ])
    checks_md = "\n".join(
        f"- {'PASS' if value else 'FAIL'} — {key.replace('_', ' ')}" for key, value in go_checks.items()
    )
    decision = "GO toward a WWW-style full paper" if overall_go else "REFRAME before committing to a WWW-style full paper"
    wands = stats["wands"]
    esci = stats["esci"]
    avg_wands = table1[table1.dataset.eq("wands")]["VI@20_micro"].mean()
    avg_esci = table1[table1.dataset.eq("esci")]["VI@20_micro"].mean()

    report = f"""# Phase II Report

## Same Product, Different Visibility: Representation Sensitivity and Relevance–Visibility Alignment in E-Commerce Retrieval

Protocol version: `{cfg['protocol_version']}`. Decision: **{decision}.**

## Executive finding

The confirmatory Phase II-A gate {'passed' if gate['phase2a_pass'] else 'did not pass'}. Across the three frozen dense retrievers, mean highest-relevance micro VI@20 was {pct(avg_wands)} on WANDS and {pct(avg_esci)} on held-out ESCI. A VI event means that the same judged product entered and left the first 20 results when only the order of its unchanged structured attributes changed. This result therefore generalizes beyond the Phase I MiniLM observation {'and survives the prespecified position controls' if controls_ok else 'but is not fully robust to the prespecified position controls'}.

The Phase II-B result is more qualified. Canonicalization can remove attribute-order instability by construction, but it counts as a successful mitigation only when query-paired relevance is not detectably worse. The table below reports that tradeoff for every model and dataset. Single-product interventions show the direct target effect separately from changes produced when the whole catalog is transformed.

## Go / no-go audit

{checks_md}

The recommended framing is **representation robustness and relevance–visibility alignment in neural product retrieval**. The evidence does not establish bias by a deployed shopping agent, objective product quality, merchant harm, or downstream purchase effects.

## Experimental setup

WANDS uses {wands['products']:,} products, {wands['queries']:,} queries and {wands['judgments']:,} cleaned judgments, retaining the Phase I labels and split. The held-out ESCI confirmation uses {esci['products']:,} US products, {esci['queries']:,} deterministically selected test queries and {esci['judgments']:,} judgments. Query groups were selected before retrieval by sorting SHA-256 of `20260904:query_id`. Every judged product for those queries forms one union catalog; unknown query-product pairs remain unjudged. This is an intentionally difficult full-union retrieval evaluation and is not the official ESCI Task 1 candidate-pool setting.

The official ESCI grades are retained as E=1, C=0.1, S=0.01 and I=0. WANDS uses Exact=3, Partial=1 and Irrelevant=0. Pooled cNDCG removes unjudged products before discounting, while recall and rank operate on the complete retrieval catalog. The [WANDS repository](https://github.com/wayfair/WANDS) and [ESCI paper](https://arxiv.org/abs/2206.06588) document the source collections; the exact ESCI snapshot is pinned in the [official repository](https://github.com/amazon-science/esci-data).

The retrievers are [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) with attention-mask mean pooling and 256 tokens, [BGE-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) with CLS pooling and 512 tokens, and [GTE-ModernBERT-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) with CLS pooling and 8,192 tokens. Revisions, tokenizers, precision and deterministic tie handling appear in `config/phase2.json` and the embedding metadata.

The primary family contains C0, reversed attribute order and five fixed per-product random attribute orders. All members preserve the character and token multisets of the original plain serialization. Sentence reversal, section reversal and two JSON key orders are secondary controls. Automatic audits verify source-atom retention, numeric equality and exact raw JSON values. The frozen protocol and its pre-result hashes are in `PROTOCOL.md` and `results/protocol_manifest.json`.

## Primary representation sensitivity

{t1_md}

Micro VI treats judged highest-relevance pairs equally. Query-macro VI first averages pairs within a query; its interval uses 10,000 query bootstrap replicates. The go/no-go cell threshold was fixed at micro VI@20 >=3% and query-macro lower CI >1%.

### Secondary fact-equivalent controls

{secondary_md}

These controls are prespecified diagnostics rather than members of the primary seven-variant family. C3 reverses complete description sentences, C4 reverses major sections, and C5 changes JSON key order while preserving decoded values. Large negative or positive shifts are reported as sensitivity evidence, not as mitigation gains.

## Query types

The largest prespecified coarse query-type cells are shown below. The taxonomy is a deterministic lexical heuristic and should be validated manually before submission.

{query_md}

## Relevance strata

`results/phase2_tables/relevance_stratified_native.csv` reports every available label separately. Highest-minus-low-relevance query-paired contrasts use 10,000 sign-permutation draws and Holm correction across the three retrievers within dataset. These contrasts test disproportionate instability; the absolute highest-relevance VI values in Table 1 answer whether meaningful products are affected even when the contrast is small or negative.

{contrasts_md}

## Chunking and position controls

{controls_md}

The controls vary chunk length, overlap, mean pooling and CLS pooling on the full WANDS catalog for MiniLM and BGE. GTE's native 8,192-token window provides the long-context no-chunking control. Persistence is judged by the same magnitude rule; a failed cell is retained as a boundary condition.

## Relevance–visibility alignment

{t3_md}

Here `RelevantHidden@20` is one minus recall of the highest relevance label. SAR is computed as per-query cNDCG@10 minus lambda times query-level VI@20 for every prespecified lambda in {{0, .1, .25, .5, 1}}; all results are in `stability_adjusted_relevance_native.csv`, with no lambda selected after seeing the data.

## Mitigation

{t4_md}

M1 uses fixed factual field sentences while retaining source strings. M2 labels fields and sorts raw attributes canonically, retaining duplicates. A method is marked successful when its VI point estimate falls and its paired cNDCG interval does not establish a loss. Delta is mitigation minus C0; intervals and sign-permutation p-values are query-paired, and the two methods are Holm-corrected within dataset and retriever.

## Single-product intervention

{direct_md}

For each judged relevant target, this experiment leaves every competitor in C0 and replaces only the target embedding. The corresponding full-catalog rank uses the same alternative representation for every product. The difference separates a direct representation effect from changes in the competitive score distribution.

## Qualitative evidence and failures

Figure A uses one deterministically selected Top-20 rescue from each dataset, selected only after quantitative analysis and only for illustration. Source descriptions and raw attributes are stored in `results/phase2_qualitative/strong_examples.csv`; the five largest direct degradations per dataset are retained in `failure_cases.csv`. The Phase I Santee example remains valid as prior illustration: for query *acrylic clear chair*, the unchanged source contained `acrylic` and `clear`, while factual framing moved the MiniLM chunked rank from 545 to 2. It is not reused as Phase II proof.

## Statistical interpretation

The primary generalization claim rests on a prespecified magnitude threshold and query-bootstrap uncertainty, rather than a post-hoc null test against exactly zero. Relevance-stratum and mitigation contrasts use paired sign permutations with Holm correction in their predefined families. Query-type cells and the qualitative extremes are exploratory. Confidence intervals quantify query sampling under these fixed catalogs and encoders; they do not cover dataset, model-family or annotation uncertainty.

## Failure modes and limitations

- ESCI is a reproducible 500-query union-catalog confirmation, not the official candidate-list ranking task. Its incomplete judgments make raw full-catalog NDCG inappropriate; this report uses condensed cNDCG and known-label recall.
- WANDS and ESCI labels identify relevance, not the objectively best product or product quality. Source equivalence says that serialization retains recorded facts; it does not certify those facts as true.
- M2 removes attribute-order variability mechanically. Its scientific value depends on the paired relevance result and on whether other lossless perturbations still matter.
- The query taxonomy is coarse, and some apparent brand/entity queries may actually be product types. A blinded manual audit is needed for a camera-ready query analysis.
- Results cover bi-encoder retrieval. They do not yet support claims about rerankers, commercial shopping agents, recommendations, exposure, clicks or sales.
- Extreme improvements coexist with extreme degradations. A representation should not be recommended for deployment from averages alone; the saved failure cases show why.

## Recommendation

**{decision}.** The next paper version should keep the causal intervention narrow: fixed facts, fixed query and fixed model, with serialization as the manipulated variable. If the decision is GO, the highest-value additions are an official ESCI candidate-pool replication, one e-commerce-trained encoder, a manual query-type audit, and a reranker robustness experiment. Proprietary shopping-agent APIs and prompt-heavy agent simulations remain premature.

## Artifact index

- Configuration: `config/phase2.json`
- Frozen protocol: `PROTOCOL.md`
- Main tables: `results/phase2_tables/`
- Figures A–E: `results/phase2_figures/`
- Per-query outputs: `results/phase2_per_query/`
- Judged-pair ranks: `results/phase2_pair_ranks/`
- Single-product outputs: `results/phase2_single_product/`
- Qualitative cases: `results/phase2_qualitative/`
- Equivalence audits: `results/equivalence/`
"""
    (P2 / "PHASE2_REPORT.md").write_text(report, encoding="utf-8")

    abstract = (
        f"Neural product retrievers should not make relevant items visible or invisible merely because identical "
        f"catalog facts are serialized in a different order. We test this property on WANDS and a held-out "
        f"Amazon ESCI subset with three frozen dense encoders and seven character- and token-equivalent "
        f"attribute-order variants. Highest-relevance micro VI@20 averages {pct(avg_wands)} on WANDS and "
        f"{pct(avg_esci)} on ESCI across retrievers. We further separate single-product from full-catalog "
        f"interventions and test two query-agnostic deterministic mitigations. The results support a "
        f"representation-robustness framing while exposing relevance tradeoffs that rule out treating rank "
        f"gain alone as mitigation success."
    )
    paper_setup = report.split("## Experimental setup\n\n", 1)[1].split(
        "\n## Primary representation sensitivity", 1
    )[0]
    paper_limitations = report.split("## Failure modes and limitations\n\n", 1)[1].split(
        "\n## Recommendation", 1
    )[0]
    paper = f"""# Same Product, Different Visibility: Representation Sensitivity and Relevance–Visibility Alignment in E-Commerce Retrieval

## Abstract

{abstract}

## 1. Introduction

Product search systems consume catalog records whose facts can be serialized in many equivalent ways. A title may precede or follow attributes; attribute dictionaries can use arbitrary insertion orders; and descriptions can expose the same facts at different positions. These implementation choices should not determine whether a highly relevant product is retrieved. We call their observable effect **representation-induced visibility instability**.

This paper asks three questions. First, does the effect reproduce across datasets and encoder families? Second, does it affect products judged highly relevant, including ordinary product-type and multi-constraint queries? Third, can deterministic, query-independent representations reduce instability while preserving relevance? We answer these questions with controlled attribute permutations, position and pooling controls, relevance-stratified analysis, and single-product interventions.

Our contributions are: (1) a multi-variant VI@K measurement that separates threshold crossing from average relevance; (2) a cross-dataset study spanning WANDS and held-out ESCI; (3) a distinction between direct target interventions and whole-catalog transformations; and (4) an alignment criterion that rejects mitigations which gain stability by sacrificing human relevance.

## 2. Related work

[WANDS](https://github.com/wayfair/WANDS) and the [Shopping Queries/ESCI benchmark](https://arxiv.org/abs/2206.06588) made graded, real-product relevance judgments available for reproducible product-search evaluation. Dense dual encoders represent queries and documents as vectors scored by an inner product, enabling exhaustive retrieval but compressing a long structured record into one point. Behavioral IR work such as [ABNIRML](https://aclanthology.org/2022.tacl-1.13/) has shown that neural rankers vary in their sensitivity to word and sentence order. Separately, controlled NLP studies find that BERT-family models can be unexpectedly resilient to token shuffling on some tasks, while later representations still use order when lexical cues are insufficient ([Hessel and Schofield, 2021](https://aclanthology.org/2021.acl-short.27/); [Papadimitriou et al., 2022](https://aclanthology.org/2022.acl-short.71/)). Our focus differs in both intervention and endpoint: we preserve complete catalog field values and measure whether a judged product crosses operational visibility thresholds.

## 3. Data and experimental design

{paper_setup}

## 4. Metrics and inference

For a judged pair `(q,p)`, VI@K is one when at least one fact-equivalent variant ranks the product at or above K and another ranks it below K. We also measure rank range, rank standard deviation, reciprocal-rank variance and pairwise crossing. Alignment uses condensed NDCG, known-label recall, highest-relevance hidden rate, Spearman correlation between human grade and negative rank, and SAR across five penalty values. All primary uncertainty and tests are query-paired as described in the frozen protocol.

## 5. Results

### 5.1 Cross-model and cross-dataset sensitivity

{t1_md}

### 5.2 Position, chunking and pooling

{controls_md}

### 5.3 Relevance–visibility alignment and mitigation

{t3_md}

{t4_md}

### 5.4 Direct interventions

{direct_md}

## 6. Discussion

The confirmatory gate {'passes' if gate['phase2a_pass'] else 'fails'}, so the effect cannot be dismissed as a MiniLM-only WANDS artifact. The size varies materially by encoder and dataset, which argues against a universal scalar correction. Canonical serialization is a useful systems intervention only in cells where relevance is maintained; deterministic stability alone is trivial to manufacture.

The strongest causal statement supported by the design is local: with a fixed query, catalog, bi-encoder and recorded fact set, serialization can change a judged product's retrieval visibility. The study does not identify the best real-world product, downstream user utility, or strategic seller behavior.

## 7. Limitations

{paper_limitations}

## 8. Conclusion

The evidence {'supports' if overall_go else 'does not yet support'} continued development of a full paper under the representation-robustness framing. Neural e-commerce retrieval exhibits measurable visibility instability under lossless attribute-order changes, and relevance-aware mitigation evaluation prevents a stable but less useful serialization from being counted as progress.

## References and artifacts

The source datasets and model cards are linked in `PHASE2_REPORT.md`. Exact revisions, outputs, audits and statistical tables are included in the Phase II artifact tree.
"""
    (P2 / "PAPER_DRAFT.md").write_text(paper, encoding="utf-8")

    tracked = [
        P2 / "config/phase2.json", P2 / "PROTOCOL.md", P2 / "PHASE2_REPORT.md",
        P2 / "PAPER_DRAFT.md", P2 / "src/representations.py", P2 / "src/mitigations.py",
        *sorted(TABLES.glob("*.csv")), *sorted(TABLES.glob("*.json")),
        *sorted((P2 / "results/equivalence").glob("*.csv")),
        *sorted(FIGURES.glob("*.png")),
        *sorted((P2 / "results/phase2_qualitative").glob("*.csv")),
    ]
    manifest = {
        "protocol_version": cfg["protocol_version"], "overall_go": overall_go,
        "go_checks": go_checks,
        "files": {str(path.relative_to(P2)): sha256(path) for path in tracked if path.exists()},
    }
    (P2 / "results/final_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"decision": decision, "checks": go_checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
