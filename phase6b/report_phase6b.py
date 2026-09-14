"""Generate the complete VI-B report from frozen measurements and diagnostics."""
import json
import numpy as np
import pandas as pd

from experiment import ROOT, HERE, OUT, CONFIG, FAMILIES, STATS, sha, dump, now


def tab(frame, columns=None, digits=3):
    return (frame[columns] if columns else frame).to_markdown(index=False, floatfmt=f".{digits}f")


def main():
    cfg = json.loads(CONFIG.read_text(encoding="utf8"))
    lock = json.loads((HERE / "selection.json").read_text())
    audit = json.loads((HERE / "PHASE6B_TRAINING_AUDIT.json").read_text())
    results = pd.read_csv(HERE / "PHASE6B_RESULTS.csv")
    scale = pd.read_csv(HERE / "PHASE6B_SCALE.csv")
    bootstrap = pd.read_csv(HERE / "PHASE6B_BOOTSTRAP.csv")
    seeds = pd.read_csv(HERE / "seed_summary.csv")
    curves = pd.read_csv(HERE / "all_development_curves.csv")
    costs = pd.read_csv(HERE / "cost_summary.csv")
    invariant = json.loads((HERE / "PHASE6B_INVARIANCE.json").read_text())
    initial = pd.read_csv(HERE / "initialization.csv")
    winner_keys = list(lock["family_winners"].values())
    confirm_keys = [k for k, v in lock["selected"].items() if v["seed"] != 42]
    primary_keys = [f"{f}_{t}_s42" for f in FAMILIES for t in ["T0", "T1", "T2"]]
    test = results[(results.Split == "test") & (results.Mode == "catalog")].set_index("Method")
    dev = results[(results.Split == "dev") & (results.Mode == "catalog")].set_index("Method")
    baseline = test.loc["Set-Attention", "Recall@100"]
    base_only = all(lock["selected"][k]["best_epoch"] == 0 for k in primary_keys)
    def difference(key, reference):
        row = bootstrap[(bootstrap.Method == key) & (bootstrap.Reference == reference) & (bootstrap.Split == "test") &
                        (bootstrap.Mode == "catalog") & (bootstrap.Metric == "Recall@100")].iloc[0]
        return f"{row.Delta_pp:+.3f} pp (95% CI {row.CI_low_pp:+.3f} to {row.CI_high_pp:+.3f})"
    # The report does not label an epoch0 identity copy as a new learned solution.
    if base_only:
        decision = "A. PHASE VI FAILURE CONFIRMED"
        rationale = "Every seed42 T0/T1/T2 primary model selected epoch 0. Initialization restores Set-Attention, but learned corrections provide no development-supported gain."
    else:
        strong, competitive, scale_gain = False, False, False
        for family, key in lock["family_winners"].items():
            group = seeds[(seeds.Family == family) & (seeds.Split == "test") & (seeds.Mode == "catalog")].iloc[0]
            dgroup = seeds[(seeds.Family == family) & (seeds.Split == "dev") & (seeds.Mode == "catalog")].iloc[0]
            trained = lock["selected"][key]["best_epoch"] > 0
            strong |= bool(trained and group.N >= 3 and dgroup["Recall@100_min"] > 100 * cfg["dev_baselines"]["Set-Attention_s42"] and
                           group["Recall@100_mean"] >= test.loc["Canonical", "Recall@100"])
            competitive |= bool(trained and dev.loc[key, "Recall@100"] > 100 * cfg["dev_baselines"]["Set-Attention_s42"] and group["Recall@100_mean"] >= baseline)
            x = scale[(scale.Method == FAMILIES[family]) & (scale.Split == "dev")].sort_values("Training_Fraction")
            scale_gain |= bool((np.diff(x["Recall@100"]) > 0).all())
        decision = ("D. STRONG SOLUTION FOUND" if strong else "C. LEARNED INVARIANT BASELINE IS NOW COMPETITIVE" if competitive else
                    "B. TRAINING REGIME WAS THE MAIN LIMITATION" if scale_gain else "A. PHASE VI FAILURE CONFIRMED")
        rationale = "Joint development, seed, scaling and held-out evidence; no decision from zero VI or test point estimates alone."
    label_distribution = pd.DataFrame([dict(Label=k, **v) for k, v in audit["label_distribution"].items()])
    data_rows = []
    for key, v in audit["datasets"].items():
        counts = np.asarray(list(v["examples_per_query"].values()))
        data_rows.append(dict(Regime=key, Examples=v["examples"], Queries=v["unique_queries"], Unique_positives=v["unique_positive_products"],
                              Unique_negatives=v["unique_negative_products"], Negatives_min=v["negatives_per_example"]["min"],
                              Negatives_mean=v["negatives_per_example"]["mean"], Negatives_max=v["negatives_per_example"]["max"],
                              Examples_per_query_min=int(counts.min()), Examples_per_query_median=float(np.median(counts)),
                              Examples_per_query_max=int(counts.max()), Products_repeated_across_queries=v["repeated_products_across_queries"]))
    diagnostics, invrows, candidate_rows = [], [], []
    for key, selected in lock["selected"].items():
        summary = json.loads((OUT / key / "residual_summary.json").read_text())
        # The coefficient is global. Its exact per-product value is the median,
        # not the FP32 accumulation mean of thousands of equal scalar values.
        row = dict(Method=key, Epoch=selected["best_epoch"], Alpha=summary["alpha"]["median"],
                   Base_norm_mean=summary["base_attribute_norm"]["mean"], Delta_norm_mean=summary["delta_norm"]["mean"],
                   Delta_to_base_median=summary["delta_to_base"]["median"], Delta_to_base_p95=summary["delta_to_base"]["p95"],
                   Scaled_residual_p95=summary["scaled_delta_to_base"]["p95"], Cosine_final_base_mean=summary["final_base_cosine"]["mean"],
                   Cosine_final_base_p05=summary["final_base_cosine"]["p05"])
        diagnostics.append(row)
        inv = invariant["models"][key]
        full = inv["full_catalog_audit"]
        invrows.append(dict(Method=key, Sample_max_L2=inv["l2_max"], Sample_mean_L2=inv["l2_mean"],
                            Sample_max_cosine_difference=inv["cosine_difference_max"], Sample_mean_cosine_difference=inv["cosine_difference_mean"],
                            Full_max_L2=max(x["max_L2"] for x in full), Changed_ranks=sum(x["changed_ranks"] for x in full),
                            Regenerated_VI20=100 * inv["regenerated_metrics"]["VI@20"], Regenerated_VI100=100 * inv["regenerated_metrics"]["VI@100"]))
    for c in lock["candidates"]:
        cv = curves[curves.candidate.eq(c["name"])]
        best_trained = cv[cv.epoch > 0].sort_values(["Recall@100", "epoch"], ascending=[False, True]).iloc[0]
        last = cv.iloc[-1]
        candidate_rows.append(dict(Candidate=c["name"], Selected_epoch=c["best_epoch"], Epochs_run=c["epochs_run"],
                                   Selected_dev_R100=100 * c["dev_recall100"], Best_trained_epoch=int(best_trained.epoch),
                                   Best_trained_dev_R100=100 * best_trained["Recall@100"], Last_dev_R100=100 * last["Recall@100"],
                                   Last_train_loss=last.train_loss, Last_alpha=last.alpha_global,
                                   Last_delta_norm=last.delta_norm, Last_cosine_to_base=last.final_base_cosine,
                                   Max_gradient_norm=float(cv.gradient_max.max())))
    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(HERE / "candidate_failure_diagnostics.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(HERE / "residual_summary.csv", index=False)
    pd.DataFrame(invrows).to_csv(HERE / "invariance_summary.csv", index=False)
    table1 = pd.read_csv(HERE / "TABLE_VI_B1.csv")
    labels = {f"{f}_{t}_s42": f"Residual {'DeepSets' if f == 'D' else 'Set Transformer'} {t}" for f in FAMILIES for t in ["T0", "T1", "T2"]}
    table1.Method = table1.Method.replace(labels)
    init_display = initial.copy()
    for col in ["Recall@20", "Recall@100", "VI@20", "VI@100", "Robust@100", "Never@100"]:
        init_display[col] *= 100
    fit = results[(results.Split == "test") & (results.Mode == "catalog_fully_fitting")]
    target = results[(results.Split == "test") & (results.Mode == "target")]
    ci = bootstrap[(bootstrap.Split == "test") & (bootstrap.Mode == "catalog") & (bootstrap.Metric == "Recall@100")]
    ablation_ci = bootstrap[(bootstrap.Split == "dev") & (bootstrap.Mode == "catalog") & bootstrap.Reference.str.match(r"^[DS]_T")]
    selected_dev_columns = ["Recall@20", "Recall@100", "Robust@100", "Never@100", "Epoch"]
    complete_epoch_count = int(sum(c["epochs_run"] for c in lock["candidates"]))
    alpha_note = "The raw epoch CSV alpha column averages identical FP32 scalar entries and can differ slightly from the actual coefficient through accumulation rounding. `all_development_curves.csv:alpha_global` is reconstructed from each epoch's final per-step scalar (epoch0=.05). Selected per-product alpha quantiles give the same global value. This reporting correction changes no weights, scores, checkpoint choices or frozen training code."
    lines = [
        "# PHASE VI-B — Training scale and value-preserving invariant representations", "",
        f"Completed {now()}. All VI-B outputs are under `phase6b/`; historical Phase VI artifacts remain unchanged.", "",
        "## Executive summary and final decision", "",
        f"**{decision}.** {rationale}", "",
        f"The bounded experiment completed {len(lock['candidates'])} candidate runs and {complete_epoch_count} trained epochs. All {len(lock['selected'])} development-selected checkpoints—including the scale runs and seeds 43/44—are epoch 0. Every selected checkpoint received a separate full-catalog permutation audit.", "",
        f"Residual initialization works: both architectures begin at Set-Attention's {baseline:.3f}% held-out Recall@100, rather than Phase VI's random-output retrieval space. The judged training expansion is 546→3,257 positive examples; T2 uses up to8 explicit negatives, averaging6.395. It expands positive coverage across the same44 contrastively usable queries, not an independently enlarged query population.", "",
        "The distinction between preserved initialization and useful learning is central. A selected epoch0 model is a copy of the existing Set-Attention representation, not a newly learned improvement. All training curves and their degraded trained states are retained alongside the selected checkpoint results. The scale plot exposes this distinction instead of presenting fallback-to-base performance as evidence that more data helps.", "",
        "No new architecture search, circular model, Set Transformer + Field-ID variant, PI-FT, consistency objective or Phase VI-C experiment was run. The main paper was not modified.", "",
        "## 1. Scientific question and frozen history", "",
        "Phase VI showed failure under546 triples while introducing new random output mappings. VI-B tests whether preserving a pretrained invariant value path and expanding judged supervision can recover useful retrieval. It does not test or justify the universal claim that permutation-invariant architectures are ineffective.", "",
        "The current Phase VI report, models, protocol, selected checkpoints, raw evaluation code and training source were inspected before implementation. `historical_sources.json` records the prior delivery and input hashes, including Phase VI checkpoints, candidate logs, reports and ranks. These hashes are checked before training/evaluation and at final delivery. The raw WANDS/BGE track remains separate from labeled PI-FT.", "",
        "## 2. Training-data audit — reported before training", "",
        "There are72 fixed training query IDs and34,203 judged training pairs. Pair counts, unique catalog products and queries differ; they are not interchangeable:", "",
        tab(label_distribution), "",
        f"The historical546 triples were reconstructed exactly from the existing `phase5_mitigation/screen.py:freeze` rule: at most16 Exact positives/query in `positive42` SHA256 order, paired cyclically with explicitly Irrelevant products in `negative42` hash order. Queries without negatives were skipped. The new audit identifies {audit['excluded_positive_pairs']} Exact pairs in {len(audit['excluded_queries'])} training queries with no explicitly Irrelevant judgments; their complete IDs and reasons are saved in `data/excluded_positives.csv`. They cannot form a safe judged-only contrastive example under this protocol.", "",
        "Regime construction:", "",
        "- **T0:** exactly the existing546 triples.",
        "- **T1:** all3,257 eligible Exact query-product pairs, each with one explicit Irrelevant negative. T0 positives retain their original negative, making T0 a true subset of T1 comparisons. There is no cap on eligible positives.",
        "- **T2:** exactly the T1 positive examples, with up to8 distinct explicit Irrelevant products per example. The first negative is the T1 negative. Remaining negatives have deterministic query/positive/negative hash order, fixed across epochs and optimizer seeds.", "",
        tab(pd.DataFrame(data_rows)), "",
        "Full per-query example counts, repeated-product counts and label presentations are in PHASE6B_TRAINING_AUDIT.json. T0/T1/T2 use44 unique training queries; the25% and50% subsets cover38 and41. The entire training split contains72 queries,54 with Exact labels, but only44 have both an Exact positive and an explicit Irrelevant negative. Unjudged and Partial products are never used as negatives, and no development or held-out judgment enters training.", "",
        "Repeated catalog products across queries are preserved with query-specific labels. A product relevant to one query may be an explicit Irrelevant product for another; only the latter judgment authorizes that comparison. There are no unmasked in-batch negatives. Larger T1/T2 also alter examples-per-query weighting relative to the old16-positive cap; this limits causal attribution to sample count alone.", "",
        "## 3. Exact residual representation and architecture reuse", "",
        "Only the original Phase VI S4 DeepSets and S1 Set Transformer branches are reused, including128 hidden dimensions and the original output dimension768. DeepSets retains its existing field-ID component and frozen2,582-entry training vocabulary; no vocabulary expansion accompanies data scaling. The retained Set Transformer has one four-head self-attention block, residual/norm/feed-forward, one PMA-style learned seed and no positional encoding. There is no Set Transformer + Field-ID experiment.", "",
        "The frozen Set-Attention seed42 scorer produces an invariant base from the original BGE field values:", "",
        "```text",
        "b_attr = normalize(sum_i softmax(frozen_scorer(h_i)) * h_i)",
        "delta  = raw 768-dimensional output of the existing invariant branch",
        "alpha  = sigmoid(beta), beta(0) = log(0.05 / 0.95)",
        "v_attr = normalize(b_attr + alpha * delta)",
        "v_final = normalize(0.5 * unchanged_non_attribute + 0.5 * v_attr)",
        "```", "",
        "Only the final linear layer of each learned branch is zero-initialized (weight and bias). Other branch parameters retain the existing random initialization. Thus delta is exactly zero at epoch0, while alpha starts at.05. The base scorer and BGE are frozen. No Original flattened vector is used as a residual base. Field occurrences, including duplicate keys/atoms, remain intact; field-internal text is never rewritten or reordered. The attribute base and residual are both reaggregated in permutation audits.", "",
        "This is an additive residual with an unnormalized learned branch. A small coefficient does not bound `alpha*||delta||`; the branch may grow large during training. That is a limitation of this prespecified formulation, not a reason to introduce another formulation after seeing the results.", "",
        "## 4. Mandatory epoch-0 check", "",
        "Both seed42 architectures were checked before any optimizer step using independent permutations, full-catalog vectors and the archived Set-Attention base. The maximum L2 difference to the archived base was about1.57e-7, well inside1e-5. Zero residual was asserted. The initialization test metrics below were read only for the prespecified identity guard; no parameter or model was selected from them. All subsequent trained-model test evaluation occurred after the development lock.", "",
        tab(init_display, ["family", "split", "epoch", "Recall@100", "VI@20", "VI@100", "Robust@100", "max_L2_to_archived_base"]), "",
        "Every later candidate and confirmation seed also checks the zero residual and base similarity at epoch0. Epoch0 is an eligible safety checkpoint. If training cannot improve dev retrieval, selecting epoch0 is explicitly a training failure/fallback, not evidence of a successful learned residual.", "",
        "## 5. Training, negatives, checkpoint selection and seeds", "",
        "The loss is `logsumexp([q·p+, q·n1, ..., q·nK]/0.05) - q·p+/0.05`. With one negative it equals the previous pairwise softplus objective. K ranges1–8 according to available distinct Irrelevant judgments; missing slots are masked with negative infinity. Product vectors can be computed once per unique product within a batch and gathered for each authorized comparison. This computational deduplication does not collapse attribute occurrences or create extra negatives.", "",
        "Batch16 query-positive examples; AdamW at1e-4 or3e-4 only; weight decay.01; gradient norm clipping at1; maximum20 epochs; patience4. Dev Recall@100 is primary and Robust@100 secondary, with1e-12 numerical improvement tolerance. Ties favor earlier epoch, then fewer examples/lower LR where selecting between regimes. Seed42 first. All encoder/query/cache settings remain pinned: BAAI/bge-base-en-v1.5 at the Phase VI revision, CLS/512-token input, unchanged normalized cached query/field/non-attribute embeddings, FP32 aggregator and dot-product scores, TF32 disabled, stable catalog-index tie handling.", "",
        f"The run comprises {len(lock['candidates'])} candidate runs and {complete_epoch_count} trained epochs, plus epoch0 evaluations. Config frozen at {cfg['frozen_utc']}; development/model/seed decisions locked at {lock['locked_utc']}. Selection SHA256: `{sha(HERE / 'selection.json')}`.", "",
        "A seed42 primary winner is selected per architecture from T0/T1/T2. Nested fraction diagnostics do not replace these primary choices. Seeds43/44 run when that winner reaches Canonical on development. This trigger is permissive here: Set-Attention already has79.584% development Recall@100 versus Canonical78.237%, so an unchanged base can trigger confirmation. Running extra seeds does not establish a learned gain.", "",
        tab(pd.DataFrame([dict(Family=k, **v) for k, v in lock["seed_gates"].items()])), "",
        "## 6. Complete development evidence", "",
        tab(dev.loc[["Original", "Canonical", "Set-Mean", "Set-Attention", "Phase-VI DeepSets", "Phase-VI Set Transformer"] + primary_keys + confirm_keys].reset_index(), ["Method"] + selected_dev_columns), "",
        "Every candidate, including the alternative LR and degraded later epochs:", "",
        tab(candidates, ["Candidate", "Selected_epoch", "Epochs_run", "Selected_dev_R100", "Best_trained_epoch", "Best_trained_dev_R100", "Last_dev_R100"]), "",
        "`all_development_curves.csv` retains each epoch's training loss, Recall@20/100, Robust@100, gradient summaries, base/residual norms and final-to-base cosine. Every checkpoint folder also retains per-step losses, gradient norms and alpha. Selection from only17 Exact-eligible dev queries is coarse and uncertain; a tie at the base is not proof that training preserves every retrieval ranking.", "",
        "## 7. Table VI-B1 — Training regime and residual initialization", "",
        "Held-out percentages; all rows use21,299 Exact pairs,308 eligible queries and42,994 competitors. Negatives/query means distinct negatives in a query-positive example/presentation, not the lifetime unique negative pool. Epoch0 rows are marked explicitly and must be interpreted as the frozen base.", "",
        tab(table1), "",
        "Full primary metrics with Original/Canonical and confirmation seeds:", "",
        tab(test.loc[["Original", "Canonical", "Set-Mean", "Set-Attention", "Phase-VI DeepSets", "Phase-VI Set Transformer"] + primary_keys + confirm_keys].reset_index(),
            ["Method", "Epoch", "Recall@20", "Recall@100", "VI@20", "VI@100", "Robust@100", "Never@100", "worst_schedule@100"]), "",
        "The inherited worst-schedule column is the mean of each query's worst observed schedule. `minimum_macro_schedule100` separately reports the worst aggregate catalog schedule. Recall averages schedule inclusion within pairs, then pairs within queries, then queries equally. VI=Sometimes, Robust=Always, Never=never included. All invariant methods use one cached product vector per item for primary deployment metrics; independent numerical regeneration is reported below.", "",
        "## 8. Table VI-B2 — Supervision scaling", "",
        "Both lines use T2, one fixed full-data-selected learning rate per architecture, seed42 and the same early-stopping rule. Subsets are nested global SHA256 prefixes; counts814/1,628/3,257 are approximate25%/50%/100%. The100% run is reused. No asymptotic extrapolation is made.", "",
        tab(scale[scale.Split == "dev"], ["Method", "Training_Fraction", "Examples", "Epoch", "Recall@100", "Robust@100", "Never@100", "Best_Trained_Dev_R100"]), "",
        "Held-out metrics of those same dev-selected fraction checkpoints:", "",
        tab(scale[scale.Split == "test"], ["Method", "Training_Fraction", "Examples", "Epoch", "Recall@100", "Robust@100", "Never@100"]), "",
        "![Measured supervision scaling](PHASE6B_SCALING.png)", "",
        "The upper panel is the actual selection rule, including epoch0. The lower panel is a clearly labeled descriptive view of the highest development Recall@100 among trained epochs at the same fixed LR. It exposes degradation that a flat base-fallback curve would conceal; those diagnostic epochs are not newly selected for test evaluation. More examples also mean more optimizer steps per epoch and a changed query weighting, so this is not a clean causal estimate of sample count alone.", "",
        "Development paired contrasts for supervision and objective changes:", "",
        tab(ablation_ci, ["Method", "Reference", "Delta_pp", "CI_low_pp", "CI_high_pp"]), "",
        "## 9. Bootstrap uncertainty and training seeds", "",
        "10,000 paired query-bootstrap draws, inherited seed20260963, percentile95% CIs, explicitly aligned query IDs. No multiplicity correction; no equivalence claim from a CI crossing zero. PHASE6B_BOOTSTRAP.csv reports Δ Recall@100, Δ Robust@100 and Δ Never@100 versus the four raw controls and the corresponding Phase VI architecture, plus T1−T0/T2−T1 and scaling contrasts. The full and fully-fitting supports remain separate.", "",
        tab(ci, ["Method", "Reference", "Delta_pp", "CI_low_pp", "CI_high_pp"]), "",
        "Seed summaries (sample SD describes training-seed variation; bootstrap intervals describe query uncertainty conditional on the selected models):", "",
        tab(seeds[(seeds.Mode == "catalog")], ["Family", "Split", "N", "Recall@100_mean", "Recall@100_std", "Recall@100_min", "Recall@100_max"]), "",
        "Seed-mean bootstrap rows average per-query performance across the fixed trained models before resampling queries. They do not ensemble product vectors or treat seeds as additional independent queries. Identical epoch0 seed results are mechanically identical bases, not confirmation of a new learned method.", "",
        "## 10. Common fully-fitting raw support", "",
        "Exactly the existing phase4 `fully_fits` support is reused:6,797 held-out Exact pairs across239 queries; all42,994 competitors remain. The definition tests the seven original raw serialized inputs against the token budget. It is not the labeled PI-FT support, a guarantee that every independently encoded atom is untruncated, or a causal truncation decomposition.", "",
        tab(fit[fit.Method.isin(["Original", "Canonical", "Set-Mean", "Set-Attention", "Phase-VI DeepSets", "Phase-VI Set Transformer"] + primary_keys + confirm_keys)],
            ["Method", "Epoch", "Recall@20", "Recall@100", "VI@20", "VI@100", "Robust@100", "Never@100", "worst_schedule@100"]), "",
        "## 11. Target-only gate and results", "",
        "The target-only gate requires a positive trained epoch, dev Recall@100 strictly above Set-Attention and a passed invariance check. Only primary architecture winners and their confirmation seeds can qualify. Competitors, where evaluated, are each model's own C0 catalog; the existing remove-old-target/reinsert routine preserves stable ties. Inclusion describes independent target interventions, not a common mixed-index Recall.", "",
        ("No model passed this gate. Target-only was therefore not run, as required; no negative diagnostic was substituted." if target.empty else
         tab(target, ["Method", "Inclusion@20", "Inclusion@100", "VI@20", "VI@100", "Robust@100", "Never@100"])), "",
        "## 12. Invariance verification", "",
        "Every candidate epoch uses the same128-product sample and all seven actual Phase VI schedules, rebuilding both frozen base and learned branch. Atom-multiset and serializer checks demonstrate actual reorder rather than cache reuse. Trained nonzero residuals were additionally tested against arbitrary permutations, padding and duplicate handling in unit tests. All selected models receive a full42,994-item independent schedule audit before delivery.", "",
        pd.DataFrame(invrows).to_markdown(index=False, floatfmt=".9g"), "",
        "L2 must remain≤1e-5; nonfinite values or larger changes halt retrieval evaluation. Cosine differences use float64 comparisons of the FP32 output vectors; sample means include C0. Tiny floating-point changes can alter exact ranks near ties without changing inclusion boundaries. Regenerated catalog ranks and any target-only ranks are retained, alongside primary one-vector results; zero cached VI is not presented as bitwise equality.", "",
        "## 13. Residual magnitude and retrieval geometry", "",
        "Selected models' diagnostics cover every catalog product. Alpha is a single global learned coefficient, so its per-product distribution is a point mass. `residual_diagnostics.parquet` retains base norms, raw residual norms, raw/scaled residual-to-base ratios and cosine(final,base); `residual_summary.json` contains min/5th/median/95th/max and mean values.", "",
        tab(pd.DataFrame(diagnostics), digits=6), "",
        "Training-state diagnostics expose behavior hidden by epoch0 fallback:", "",
        tab(candidates, ["Candidate", "Last_train_loss", "Last_alpha", "Last_delta_norm", "Last_cosine_to_base", "Max_gradient_norm"]), "",
        "Large residual norms and falling development recall are consistent with displacement from the pretrained retrieval geometry. The scalar alpha alone can remain near.05 while the unnormalized branch grows enough to dominate its unit base. This is an observed association within training, not a causal proof of the mechanism. Selecting epoch0 preserves geometry exactly by discarding that learned correction.", "",
        alpha_note, "",
        "## 14. Cost, stored artifacts and limitations", "",
        f"Field/query/non-attribute encoding was cached and reused, with zero new BGE encoding. Instrumented stages sum to {costs.seconds.sum():.3f} wall seconds; individual candidate training, dev generation, ranking and full schedule audits remain in cost.jsonl/cost_summary.csv. This includes CPU waiting within measured stages and may include the user's connection interruption; it excludes report-writing gaps and interpreter/import startup, and is not a profiler kernel total or online latency benchmark.", "",
        tab(costs.groupby("stage").seconds.sum().reset_index()), "",
        "The trainable branch sizes are593,921 parameters for residual DeepSets and462,593 for residual Set Transformer, including beta. Both additionally retain98,561 frozen Set-Attention scorer parameters; BGE is unchanged and frozen. Each deployment stores one768-dimensional FP32 vector per product:132,077,568 bytes (125.959 MiB) for42,994 products. Query encoding and online dot-product/ranking cost are unchanged at the same catalog size. Experimental checkpoint/vector audits on disk are not a multi-view inference index.", "",
        "Key limitations:44 contrastively usable training queries,17 Exact-eligible dev queries, historical308-query test, one dataset and one encoder; an unbounded additive residual; query weighting and optimizer-step changes with data scale; a vocabulary fixed to the historical T0 products; fixed explicit-negative lists; early stopping on a coarse metric. No inference here establishes that all invariant architectures are unsuitable, that adding arbitrary quantities of data cannot help, or that a circle/field identity causes a specific mechanism.", "",
        "## 15. Explicit answers to the ten report questions", "",
        "**Q1. Was Phase VI primarily limited by training supervision?** This experiment does not establish that. It removes the546-triple positive cap, but selected performance and the trained-state scale curves do not provide a consistent upward trend. The number of contrastively usable queries remains44, and more data changes optimization exposure and weighting.", "",
        f"**Q2. Does residual/value-preserving initialization materially improve DeepSets?** The selected result versus Phase VI DeepSets is {difference(lock['family_winners']['D'], 'Phase-VI DeepSets')}. It repairs epoch0 retrieval by starting at the known Set-Attention vector. That is initialization recovery, not evidence of a useful learned correction; compare selected/trained epoch columns rather than crediting the base as a new gain.", "",
        f"**Q3. Does it materially improve Set Transformer?** The selected result versus Phase VI Set Transformer is {difference(lock['family_winners']['S'], 'Phase-VI Set Transformer')}. It removes the random-output collapse at initialization and restores the same base. Additional cross-field learning must improve beyond that starting point to support a new method claim; the selected checkpoints and full curves show whether it does.", "",
        "**Q4. Does supervision show a consistent positive scale trend?** No supported positive trend is established by the measured25%/50%/100% curves. A flat selected curve produced by epoch0 fallback is not saturation of a successfully learned model. The trained-only diagnostic is reported explicitly, without extrapolation.", "",
        "**Q5. Does multi-negative training improve effectiveness?** The T2−T1 paired contrasts and development curves do not establish a useful gain beyond the base. Larger negative sets strengthen the objective but can accompany retrieval degradation. This does not prove that every contrastive objective or negative-mining policy would fail.", "",
        "**Q6. Can a structurally invariant model match Set-Attention?** Yes, by construction at epoch0. That equality is not a learned improvement or a statistical equivalence argument. A competitive new learned baseline would require a trained checkpoint with development-supported benefit, rather than selecting the frozen base again.", "",
        "**Q7. Can a model match Canonical?** Matching/exceeding Canonical on development is insufficient here because the inherited Set-Attention base already does. Joint held-out performance and seed evidence in the tables must support the claim; base-level held-out Recall@100 remains62.173%, below Canonical63.237%.", "",
        "**Q8. Can a model match Original with near-zero VI?** Base-level held-out Recall@100 remains below Original63.620%, despite near-zero VI. No strong Original-level learned solution is established by this reinforcement screen.", "",
        "**Q9. Does residual learning preserve pretrained BGE geometry?** The identity initialization does. Unconstrained subsequent residual learning need not: raw delta magnitude can grow while alpha changes only slightly, and development recall/cosine to base can fall together. The selected safety checkpoint can preserve geometry by choosing no correction.", "",
        "**Q10. Should a learned invariant model enter the WWW main paper?** These results do not justify a new main solution claim. Retain Set-Attention as the existing invariant comparator, and use the VI/VI-B negative curves as bounded evidence about training and alignment limitations, preferably in supplementary analysis. No manuscript rewrite or next-phase architecture expansion was performed.", "",
        "## 16. Final decision and reproducibility", "",
        f"**{decision}.** Interpret this as failure to learn a useful additional correction in the tested regime, not confirmation that permutation invariance itself is ineffective. The known base is preserved; the main question is whether new training adds retrieval effectiveness beyond it.", "",
        "Required outputs: PHASE6B_REPORT.md, PHASE6B_RESULTS.csv, PHASE6B_SCALE.csv, PHASE6B_BOOTSTRAP.csv, PHASE6B_INVARIANCE.json, PHASE6B_CONFIG.json, PHASE6B_TRAINING_AUDIT.json and FINAL_DELIVERY_MANIFEST.json. The manifest includes selected checkpoints, all candidate logs, audited source hashes and full catalog residual/rank diagnostics. Primary CSV metrics are percentages; bootstrap deltas are percentage points. Training/initialization logs retain fractions unless their header explicitly says otherwise.", "",
        "```powershell",
        "phase2/.venv/Scripts/python.exe -X utf8 -m pytest phase6b/test_residual.py -q -o cache_dir=phase6b/.pytest_cache",
        "phase2/.venv/Scripts/python.exe -X utf8 phase6b/data_audit.py",
        "phase2/.venv/Scripts/python.exe -X utf8 phase6b/experiment.py prepare",
        "phase2/.venv/Scripts/python.exe -X utf8 phase6b/experiment.py train",
        "phase2/.venv/Scripts/python.exe -X utf8 phase6b/experiment.py evaluate",
        "phase2/.venv/Scripts/python.exe -X utf8 phase6b/analyze_results.py",
        "phase2/.venv/Scripts/python.exe -X utf8 phase6b/report_phase6b.py",
        "phase2/.venv/Scripts/python.exe -X utf8 phase6b/verify_delivery.py",
        "```", "",
        "Completed runs are reused. Frozen config/source hashes reject silent experimental changes. Negative candidates and historical Phase VI evidence remain intact; no additional architecture was invented after the failure.", "",
    ]
    text = "\n".join(lines)
    for a, b in {"under546": "under 546", "are72": "are 72", "and34,203": "and 34,203", "historical546": "historical 546", "most16": "most 16",
                 "existing546": "existing 546", "all3,257": "all 3,257", "to8": "to 8", "averaging6.395": "averaging 6.395", "same44": "same 44",
                 "use44": "use 44", "the25%": "the 25%", "and50%": "and 50%", "cover38": "cover 38", "and41": "and 41", "contains72": "contains 72",
                 "queries,54": "queries, 54", "only44": "only 44", "old16": "old 16", "including128": "including 128", "dimension768": "dimension 768",
                 "frozen2,582": "frozen 2,582", "seed42": "seed 42", "Seed42": "Seed 42", "epoch0": "epoch 0", "at.05": "at .05", "near.05": "near .05",
                 "about1.57": "about 1.57", "inside1e-5": "inside 1e-5", "ranges1": "ranges 1", "Batch16": "Batch 16", "at1e-4": "at 1e-4",
                 "or3e-4": "or 3e-4", "decay.01": "decay .01", "at1;": "at 1;", "maximum20": "maximum 20", "patience4": "patience 4",
                 "with1e-12": "with 1e-12", "Seeds43/44": "Seeds 43/44", "has79.584": "has 79.584", "Canonical78.237": "Canonical 78.237",
                 "only17": "only 17", "use21,299": "use 21,299", "pairs,308": "pairs, 308", "and42,994": "and 42,994", "counts814": "counts 814",
                 "approximate25%": "approximate 25%", "The100%": "The 100%", "seed20260963": "seed 20260963", "percentile95%": "percentile 95%",
                 "reused:6,797": "reused: 6,797", "across239": "across 239", "all42,994": "all 42,994", "same128": "same 128", "full42,994": "full 42,994",
                 "are593,921": "are 593,921", "and462,593": "and 462,593", "retain98,561": "retain 98,561", "one768": "one 768",
                 "product:132,077,568": "product: 132,077,568", "for42,994": "for 42,994", "limitations:44": "limitations: 44", "queries,17": "queries, 17",
                 "historical308": "historical 308", "the546": "the 546", "remains44": "remains 44", "measured25%": "measured 25%", "remains62.173": "remains 62.173",
                 "Canonical63.237": "Canonical 63.237", "Original63.620": "Original 63.620"}.items():
        text = text.replace(a, b)
    (HERE / "PHASE6B_REPORT.md").write_text(text, encoding="utf8")
    dump(HERE / "decision.json", dict(decision=decision, rationale=rationale, generated_utc=now(), selection_sha256=sha(HERE / "selection.json")))
    print(decision)


if __name__ == "__main__":
    main()
