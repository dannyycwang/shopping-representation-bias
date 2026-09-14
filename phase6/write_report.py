"""Build the Phase VI report from actual locked results; never train or select models."""
import json
import numpy as np
import pandas as pd

from run import ROOT, HERE, CONFIG, BASELINES, dump, sha, now
from analyze import method_label


def table(frame, columns=None, digits=3):
    x = frame[columns].copy() if columns else frame.copy()
    return x.to_markdown(index=False, floatfmt=f".{digits}f")


def main():
    cfg = json.loads(CONFIG.read_text(encoding="utf8"))
    sel = json.loads((HERE / "selection.json").read_text())
    results = pd.read_csv(ROOT / "PHASE6_RESULTS.csv")
    cis = pd.read_csv(ROOT / "PHASE6_BOOTSTRAP.csv")
    inv = json.loads((ROOT / "PHASE6_INVARIANCE.json").read_text())
    trials = pd.read_csv(HERE / "development_candidates.csv")
    costs = pd.read_csv(HERE / "cost_summary.csv")
    transitions = pd.read_csv(HERE / "failure_transitions.csv")
    test = results[(results.split == "test") & (results["mode"] == "catalog")].set_index("method")
    dev = results[(results.split == "dev") & (results["mode"] == "catalog")].set_index("method")
    newkeys = list(sel["selected"])
    assert not any(g["passed"] for g in sel["gates"].values()), "Reassess the report interpretation if the dev gate changes"
    assert max(dev.loc[newkeys, "Recall@100"]) < dev.loc["Set-Attention_s42", "Recall@100"]
    decision = "A. KEEP CURRENT DIAGNOSIS PAPER"
    core = ["S1_s42", "S2_s42", "S3_s42"]
    best_core = dev.loc[core, "Recall@100"].idxmax()
    best_test = test.loc[newkeys, "Recall@100"].idxmax()
    primary_ci = cis[(cis.split == "test") & (cis["mode"] == "catalog") & (cis.metric == "Recall@100")]
    invrows, numerical = [], []
    for key, x in inv["models"].items():
        invrows.append(dict(method=key, products=x["products"], changed_order_products=x["products_with_changed_order"],
                            max_L2=x["l2_max"], mean_L2=x["l2_mean"], max_cosine_difference=x["cosine_difference_max"],
                            mean_cosine_difference=x["cosine_difference_mean"]))
        a = x["full_catalog_regeneration"]
        numerical.append(dict(method=key, max_full_catalog_L2=max(t["l2_max"] for t in a),
                              changed_pair_schedule_ranks=sum(t["changed_exact_ranks"] for t in a),
                              changed_R20_inclusions=sum(t["changed_inclusion20"] for t in a),
                              changed_R100_inclusions=sum(t["changed_inclusion100"] for t in a),
                              regenerated_VI20=100 * x["regenerated_test_metrics"]["VI@20"],
                              regenerated_VI100=100 * x["regenerated_test_metrics"]["VI@100"]))
    selection_rows = []
    for key, r in sel["selected"].items():
        selection_rows.append(dict(method=key, learning_rate=r["lr"], selected_epoch=r["best_epoch"], epochs_run=r["epochs_run"],
                                   dev_Recall100=100 * r["dev_recall100"], seed_gate=sel["gates"][r["method"]]["passed"]))
    costrows = []
    for key, r in sel["selected"].items():
        family = r["method"]
        cc = costs[costs["run"].str.startswith(family + "_")]
        row = dict(method=key, parameters=r["parameters"],
                   all_candidates_training_s=float(cc.loc[cc.stage.eq("training"), "seconds"].sum()),
                   dev_vector_generation_s=float(cc.loc[cc.stage.eq("dev_catalog_generation"), "seconds"].sum()),
                   dev_ranking_s=float(cc.loc[cc.stage.eq("dev_ranking"), "seconds"].sum()),
                   final_C0_generation_s=float(cc.loc[cc.stage.eq("final_catalog_generation"), "seconds"].sum()),
                   schedule_audit_s=float(cc.loc[cc.stage.eq("independent_schedule_audit"), "seconds"].sum()))
        costrows.append(row)
    cost_table = pd.DataFrame(costrows)
    cost_table.to_csv(HERE / "offline_costs.csv", index=False)
    candidates = trials[["name", "best_epoch", "epochs_run", "dev_recall100"]].copy()
    candidates["dev_recall100"] *= 100
    fit = results[(results.split == "test") & (results["mode"] == "catalog_fully_fitting")]
    target = results[(results.split == "test") & (results["mode"] == "target")]
    target_fit = results[(results.split == "test") & (results["mode"] == "target_fully_fitting")]
    target_audit = pd.read_csv(HERE / "target_numerical_audit.csv")
    fit_ci = cis[(cis.split == "test") & (cis["mode"] == "catalog_fully_fitting") & (cis.metric == "Recall@100") & cis.reference.isin(["Canonical", "Set-Attention_s42"])]
    allinv = [json.loads(p.read_text()) for p in (HERE / "invariance").glob("*_epoch*.json")]
    changed = pd.read_csv(HERE / "query_changes.csv")
    qtext = pd.read_csv(ROOT / "phase2/data/processed/wands_queries.csv")
    examples = changed[changed.method.eq(best_core)].sort_values("net")
    examples = pd.concat([examples.head(3), examples.tail(3)]).merge(qtext[["query_id", "query"]], on="query_id")
    figure_path = "phase6/PHASE6_SCREENING.png"
    table1 = pd.read_csv(HERE / "TABLE_VI_1.csv")
    source_cache = pd.DataFrame([dict(component="unique attribute atoms" if "unique_attributes" in x["path"] else "fixed non-attribute text",
                                         vectors=x["metadata"]["shape"][0], legacy_encoding_s=x["metadata"]["seconds"],
                                         fraction_truncated=x["metadata"]["fraction_truncated"])
                                for x in cfg["field_caches"]])
    def r(key, metric="Recall@100"):
        return float(test.loc[key, metric])
    def delta(key, ref, subset="catalog"):
        x = cis[(cis.method == key) & (cis.reference == ref) & (cis.split == "test") & (cis["mode"] == subset) & (cis.metric == "Recall@100")].iloc[0]
        return f'{x.delta_pp:+.3f} pp (95% CI {x.ci_low_pp:+.3f} to {x.ci_high_pp:+.3f})'
    params = {k: sel["selected"][k]["parameters"] for k in newkeys}
    lines = [
        "# PHASE VI — Structural permutation-invariant product representations", "",
        f"Completed {now()}. Raw WANDS / pinned frozen BGE. Fresh experimental phase in the existing repository.", "",
        "## 1. Executive summary", "",
        f"**Final decision: {decision}.**", "",
        "The prespecified structural screen failed to recover retrieval effectiveness. All four new architectures passed permutation-invariance checks, but every selected model was below Set-Attention on development Recall@100. No additional training seeds were authorized by the development gate. Negative candidates and later-epoch degradation are retained.", "",
        f"On the 308 Exact-eligible held-out queries, S1/S2/S3/S4 Recall@100 was {r('S1_s42'):.3f}% / {r('S2_s42'):.3f}% / {r('S3_s42'):.3f}% / {r('S4_s42'):.3f}%, compared with Original {r('Original'):.3f}%, Canonical {r('Canonical'):.3f}%, Set-Mean {r('Set-Mean'):.3f}%, and Set-Attention {r('Set-Attention_s42'):.3f}%.", "",
        f"The strongest core architecture on development was {method_label(best_core)}; it was chosen for the negative target-only diagnostic before test evaluation. The highest new test point estimate happened to be {method_label(best_test)} ({r(best_test):.3f}%), a descriptive observation with no follow-on selection. Zero VI coexists with substantial stable misses. This screen supports retaining the diagnosis paper, without a new structural solution claim.", "",
        "## 2. Motivation and authoritative scope", "",
        "Phase V showed that simple symmetric aggregation can remove order sensitivity while losing retrieval-relevant information. Phase VI asks whether cross-field interactions and explicit schema identity recover that effectiveness. The authoritative context inspected before implementation was the current repository, `phase5_mitigation/SET_ATTENTION_REPORT.md`, `phase5_pift/PI_FT_REPORT.md`, the existing Set-Mean/Set-Attention implementations, and current raw WANDS/BGE ranking code, plus the supplied Phase VI instructions. No earlier Codex conversations were consulted and no Phase I–V narrative was reconstructed.", "",
        "The labeled PI-FT report was read to establish the track boundary. None of its labeled serializer, 407-pair training set, fine-tuned encoder, cache, or labeled fully-fitting support enters these comparisons. All inherited baseline ranks were reused; no baseline model was retrained or re-encoded. Prior source hashes are checked again during delivery.", "",
        "## 3. Prior Set-Attention versus true Set Transformer", "",
        "Prior Set-Attention independently maps each frozen 768-dimensional atom through a 768→128→1 tanh scorer, softmaxes scalar field weights, and averages the original BGE field vectors. Its 98,561 trainable parameters cannot explicitly contextualize one field embedding through another field. The shared softmax denominator alone does not make it a self-attention encoder.", "",
        "S1/S2 contain multihead attribute-to-attribute self-attention: each projected field queries all unmasked projected fields, followed by residual connections, layer normalization, and a feed-forward block. A separate learned seed attends over the contextualized set. There is no serialization-position input. This is a small Set Transformer-style encoder with one self-attention block and PMA-style seed pooling, rather than merely an independent scalar field scorer. No claim is made of reproducing every detail of a particular external implementation.", "",
        "## 4. Architecture definitions", "",
        "For every method, the input atom is the complete unchanged native `key:value` string. Token order inside the atom is untouched. Cached BGE independently embeds each atom. Field order is arbitrary; duplicate keys and duplicate identical atoms retain their separate occurrences. Non-attribute title/class/category/description text and query embeddings remain exactly those of the raw controls.", "",
        f"- **S1 — Set Transformer ({params['S1_s42']:,} parameters):** 768→128 input projection; one four-head self-attention block; residual and layer norm; 128→256→128 GELU feed-forward; one learned seed with four-head cross-attention, residual/norm/feed-forward pooling; 128→768 output projection.",
        f"- **S2 — Set Transformer + Field-ID ({params['S2_s42']:,}):** identical common modules and initialization to S1, with a learned 128-dimensional field embedding added after content projection. Field embeddings initialize with normal standard deviation .02.",
        f"- **S3 — Circular Projection ({params['S3_s42']:,}):** concatenate 128-dimensional projected content with cos/sin of the field angle; 130→128→128 GELU per-atom MLP; masked symmetric mean; 128→128→768 output MLP. Angles are trainable and initialized evenly across the sorted training vocabulary, including UNK. This is field-identity geometry, not cyclic serialization.",
        f"- **S4 — Field-aware DeepSets ({params['S4_s42']:,}):** the inexpensive control reuses S3's machinery, replacing the two circular coordinates with an ordinary learned 128-dimensional field identity; 256→128→128 per-atom MLP, masked mean, and the same output MLP.", "",
        f"There are {cfg['field_vocab_size'] - 1:,} known keys plus trainable UNK ID 0. Keys are the trimmed substring before the first colon, with no case folding or semantic key merging. The vocabulary is fitted only to products present in the 546 raw training triples. The catalog has {cfg['catalog_unique_nonempty_field_keys']:,} unique nonempty keys; {cfg['unknown_unique_atoms']:,} unique atom strings map to UNK because their key is absent, empty, or malformed. Content still enters through the full atom embedding. {cfg['duplicate_atom_products']} products contain repeated identical atoms, preserved without deduplication. Repeated keys with distinct values are also preserved.", "",
        "All models output `normalize(0.5 * non_attribute + 0.5 * normalize(attribute_aggregate))`, using the prior convention. Final dimensionality is 768, with exactly one product vector. Empty sets contribute zero attribute signal. No dropout, positional encoding, rewritten facts, extra candidates, multi-view index, BGE fine-tuning, content-vector skip connection, or auxiliary pretraining was added. The new output mapping is randomly initialized and must learn useful alignment to the frozen retrieval space. This capacity/initialization change is part of the tested aggregators and limits mechanistic attribution relative to the value-preserving prior weighted mean. S3-fixed and larger hidden/block settings were deliberately omitted.", "",
        "## 5. Training setup and selection lock", "",
        f"BGE: `{cfg['model']['name']}@{cfg['model']['revision']}`; CLS pooling, 512-token cap, no query instruction prefix, cached unit vectors from the inherited FP16 encoder/FP32 pooling path. New aggregators and scoring use FP32; TF32 is disabled. The frozen source and cache checks include content hashes, model revision, array shape/dtype, and unit norms. The raw field inventory and catalog order are unchanged.", "",
        "The inherited split is 72 training / 24 development / 384 held-out query IDs, of which 54 / 17 / 308 have Exact judgments. Training reuses all 546 Exact-positive/explicit-Irrelevant-negative triples; no additional mining or labeled PI-FT filtering. Query splits are disjoint, but catalog/product identities may overlap, as in the inherited query split. The test is the historical held-out set, not a new independent confirmation set.", "",
        "Seed 42; AdamW with weight decay .01; learning rates .001 and .0003 only; batch 16; gradient norm clipped at 1; maximum 10 epochs; patience 3 completed epochs without a strictly better development Recall@100. Pairwise loss is `softplus((q·negative − q·positive)/0.05)`. Only aggregator parameters train. Each epoch generates the entire catalog and ranks development queries using existing stable scoring. Checkpoints are chosen solely by dev query-macro Recall@100; ties favor the earlier epoch, then the first listed learning rate. No epoch 0 model is a primary candidate.", "",
        f"Protocol/config frozen: `{cfg['frozen_utc']}`. Selection and seed decisions locked: `{sel['locked_utc']}`. Selection SHA256: `{sha(HERE / 'selection.json')}`. New test ranks were computed only after this lock. The eight candidate trials total {int(trials.epochs_run.sum())} completed epochs. All candidates' best checkpoints, epoch losses, development metrics and invariance checks are retained.", "",
        "## 6. Invariance verification", "",
        "Each selected model recomputed representations on 128 deterministic sampled products under C0, reverse C1, and all five existing seeded schedules. Atom multisets and exact correspondence to the inherited serializers were asserted. Input-order hashes demonstrate actual reordered inputs; field IDs travel with atoms. These are seven separate forward passes, not seven reads of one output cache. The same gate also ran before every epoch's development retrieval evaluation.", "",
        pd.DataFrame(invrows).to_markdown(index=False, floatfmt=".9g"), "",
        f"All {len(allinv)} epoch states passed; maximum epoch-level L2 difference was {max(x['l2_max'] for x in allinv):.9g}. The required maximum L2 tolerance is 1e-5. L2 uses the FP32 output differences; cosine comparisons use float64 arithmetic on those FP32 vectors to avoid cancellation in near-unit dot products. Means include C0 self-comparisons (128×7 comparisons).", "",
        "A stronger numerical audit independently rebuilt the entire 42,994-product catalog under the other six schedules for each selected checkpoint. Stable ranking was recomputed rather than assumed. The following counts sum over relevant-pair/schedule comparisons; they are not unique-product counts.", "",
        pd.DataFrame(numerical).to_markdown(index=False, floatfmt=".9g"), "",
        "The primary set-method table follows the prior deployed convention: one cached C0 invariant product vector is used across schedules. The independent numerical audit is retained separately, including any tiny changes in exact ranks. Mathematical permutation invariance does not promise bitwise equality, and floating-point near-ties are not hidden. Synthetic regression tests also cover all 24 permutations of a masked four-slot set, empty sets, duplicates, padding changes, UNK parsing, gradients, stable ties, target reinsertion, and query-macro aggregation. All 13 tests passed.", "",
        "## 7. Development results and seed gate", "",
        table(dev.reset_index(), ["method", "Recall@20", "Recall@100", "Robust@100", "Never@100"]), "",
        "Selected checkpoints:", "", table(pd.DataFrame(selection_rows), digits=6), "",
        "Every learning-rate trial (including negative alternatives):", "", table(candidates, digits=6), "",
        "S1/S2/S3 must strictly exceed the archived Set-Attention dev Recall@100 (79.583913%) and pass invariance before seeds 43/44 can run. None did. S4 was the optional control and also fell short. No additional seeds or architecture settings were run, and test outcomes were not used to reconsider that decision. The development set has only 17 Exact-eligible queries, so small dev differences are unstable; the complete dev bootstrap is in PHASE6_BOOTSTRAP.csv.", "",
        "## 8. Test results — Table VI-1: Structural invariant retrieval", "",
        "Percentages; deltas are percentage points. Params counts additional trainable aggregator parameters, excluding the identical frozen BGE. Every row uses 21,299 Exact query-product pairs, 308 eligible queries, and all 42,994 catalog competitors. Seed 42 for learned methods.", "",
        table(table1), "",
        "Recall@k averages the seven schedule indicators within each relevant pair, then relevant pairs within each query, then queries equally. VI is sometimes-included, Robust is always-included, and Never is never-included under the seven schedules. A zero-VI method can miss many relevant products consistently. All new methods remain below both Original and Canonical on this test, and have no development-supported effectiveness gain.", "",
        "Worst-schedule details (the inherited column is the mean of each query's worst observed schedule; the final column is the worst aggregate catalog schedule):", "",
        table(test.reset_index(), ["method", "worst_schedule@100", "minimum_macro_schedule100"]), "",
        f"![Recall versus robust inclusion screening]({figure_path})", "",
        "The diagonal represents Recall=Robust for invariant inclusion. Being on that diagonal is not sufficient: movement toward lower recall and more Never is an effectiveness loss. This plot reports one seed and does not represent independent confirmation.", "",
        "## 9. Common fully-fitting raw target subset", "",
        "Exactly the inherited `phase4/results/wands_product_features.csv:fully_fits` support is reused: products whose seven original flattened raw schedules fit the BGE token budget. It yields 6,797 held-out Exact pairs across 239 queries. Only targets are filtered; all 42,994 competitors remain. The labeled PI-FT support is excluded. This is a conditional descriptive subset, not a causal truncation decomposition and not a guarantee that every independently encoded field is untruncated.", "",
        table(fit, ["method", "Recall@20", "Recall@100", "VI@20", "VI@100", "Robust@100", "Never@100", "worst_schedule@100"]), "",
        "Relevant paired intervals on the same support:", "",
        table(fit_ci, ["method", "reference", "delta_pp", "ci_low_pp", "ci_high_pp"]), "",
        "The subset comparison must be read jointly with the full catalog results. The stronger fully-fitting performance of the prior set controls is an empirical baseline to preserve, not evidence that structural invariance alone solves truncated retrieval. The new architectures do not establish a reliable recovery of that prior advantage.", "",
        "## 10. Target-only results", "",
        f"No architecture met the development success gate. Following the pre-test protocol, {method_label(best_core)} ({best_core}) was evaluated as a **negative diagnostic**, because it was the highest-development S1/S2/S3 model. This is not a successful-method claim or a selection from test results.", "",
        "Competitors use **each method's own C0 catalog**. A target is independently replaced, with its old C0 entry removed and stable catalog-index ties preserved by the existing reinsertion routine. Primary invariant cached vectors give identical inclusion across schedules. Independently regenerated target ranks are also saved; the C0 reinsertion routine was checked against full-catalog ranks for every evaluated Exact target. Baseline target-only ranks are inherited unchanged.", "",
        table(target, ["method", "Inclusion@20", "Inclusion@100", "VI@20", "VI@100", "Robust@100", "Never@100"]), "",
        "Fully-fitting target-only support:", "",
        table(target_fit, ["method", "Inclusion@20", "Inclusion@100", "VI@20", "VI@100", "Robust@100", "Never@100"]), "",
        "Independent numerical target-only audit (rank-change counts are summed over pairs and schedules):", "",
        table(target_audit), "",
        "Inclusion is the mean of independent target interventions, not common mixed-index Recall. This analysis does not hold all methods against Original's competitors and cannot be interpreted as such an intervention.", "",
        "## 11. Bootstrap uncertainty", "",
        "10,000 paired query-bootstrap draws; inherited RNG seed 20260963; percentile 95% intervals. Query IDs are explicitly aligned before subtraction. The point estimate and all replicates preserve the existing query-macro aggregation. Full, fully-fitting, target-only and development results are stored in PHASE6_BOOTSTRAP.csv. Intervals are descriptive and uncorrected for multiple comparisons. They are conditional on selected checkpoints and omit training-seed and checkpoint-selection uncertainty. No equivalence conclusion follows from an interval covering zero.", "",
        "Full-catalog test Δ Recall@100 against all four controls, plus prespecified architectural comparisons:", "",
        table(primary_ci, ["method", "reference", "delta_pp", "ci_low_pp", "ci_high_pp"]), "",
        "Additional training seeds were not run. Their variation is unknown, not zero; `phase6/seed_summary.csv` leaves sample standard deviation undefined for a single seed. Bootstrap draws do not substitute for training seeds or independent datasets.", "",
        "## 12. Cost and parameters", "",
        "Frozen field encoding was fully reused; **new field-encoding cost was zero**. The following legacy cache timings are metadata from their original generation, not new measurements or complete end-to-end system timings:", "",
        table(source_cache, digits=6), "",
        "Phase VI measured offline wall seconds, with GPU synchronization around training and blocking transfers during generation. Training sums include both learning-rate candidates and all epochs executed until early stopping. Development catalog construction is separate from training. Final C0 generation creates the single deployment vector per product. Schedule audit costs include independent reorder/regeneration and retrieval/target-only analysis and are experimental verification overhead.", "",
        table(cost_table), "",
        f"Total instrumented Phase VI stages: {costs.seconds.sum():.3f}s. This excludes interpreter/import startup, gaps between commands, test execution, final report/plot creation and some setup overhead; it is not a profiler kernel total or an end-to-end latency benchmark. GPU: {cfg['environment']['gpu']}; PyTorch {cfg['environment']['torch']}. Detailed stage accounting remains in `phase6/cost.jsonl` and `cost_summary.csv`.", "",
        f"Online query encoding is unchanged. Every method stores one 768-dimensional FP32 vector per item: {42994 * 768 * 4:,} bytes ({42994 * 768 * 4 / 2**20:.3f} MiB) for the catalog vector matrix, identical to Original. Exact scoring performs the same query-to-catalog dot products and stable ranking, so algorithmic scoring cost is 1.0× Original. No claim of a measured online speedup is made. Extra aggregator parameters are offline-only; vectors per item and candidate slots do not expand. Experimental checkpoints/rank audits on disk are not a multi-vector inference index.", "",
        "## 13. Failure analysis", "",
        "Training loss generally decreased while development retrieval degraded, often sharply, in later epochs. Early stopping preserved the best trained epoch rather than the lowest training loss. S1/S2/S3 selected epoch 1 at lr .001; S4 selected epoch 1 at lr .0003. All alternatives remain in the candidate table and epoch CSVs. The tiny S3 development advantage of lr .001 over lr .0003 was resolved mechanically by the frozen metric, not by looking at test.", "",
        "These models learn new projected output maps from only 546 supervised triples. The prior scalar attention instead preserves the original BGE value vectors directly. The failure is compatible with insufficiently learned output alignment and poor generalization, but this experiment does not isolate those explanations. It cannot establish that cross-field attention, schema identity, or circular coordinates inherently destroy relevance. It also does not establish a causal truncation mechanism. More training data, different initialization or alternative parameterizations remain untested; none was added after seeing the failure.", "",
        "Query-macro inclusion transitions versus Original (percentage points; rescued/newly_missed average over the seven matched schedule indicators):", "",
        table(transitions), "",
        f"Illustrative largest losses and gains for the dev-selected core diagnostic {best_core}; these are post-evaluation descriptive examples, not model-selection criteria:", "",
        table(examples, ["query_id", "query", "rescued", "newly_missed", "net"]), "",
        "All query transitions are retained, including negative examples. The full rank parquets support pair-level inspection without selecting a favorable subset. Stable absence (Never) and lost recall prevent interpreting zero VI as a visibility benefit.", "",
        "## 14. Scientific interpretation — explicit answers", "",
        f"**Q1. Does cross-field self-attention improve over independent field weighting?** No in this screen. S1 versus Set-Attention is {delta('S1_s42', 'Set-Attention_s42')} on test Recall@100, and development Recall@100 is {dev.loc['S1_s42', 'Recall@100']:.3f}% versus {dev.loc['Set-Attention_s42', 'Recall@100']:.3f}%. The comparison changes the attribute aggregator, including its learned output projection, so it is not a causal isolation of attention alone.", "",
        f"**Q2. Does field identity help?** It improves S2 over S1 on development ({dev.loc['S2_s42', 'Recall@100'] - dev.loc['S1_s42', 'Recall@100']:+.3f} pp), but the test difference is {delta('S2_s42', 'S1_s42')}. The directions disagree. There is no generalizable positive field-ID conclusion from this single-seed small-data screen.", "",
        f"**Q3. Does circular geometry add anything beyond ordinary field identity?** S3 versus the ordinary-ID DeepSets control is {delta('S3_s42', 'S4_s42')} on test; development difference is {dev.loc['S3_s42', 'Recall@100'] - dev.loc['S4_s42', 'Recall@100']:+.3f} pp. S3 versus S2 is {delta('S3_s42', 'S2_s42')}, but that comparison changes pooling/attention and parameterization as well. Neither supports an inherently correct circular topology or novelty claim. S3/S4 also differ in identity bottleneck size and parameter count, so even their contrast does not isolate geometry alone.", "",
        f"**Q4. Can a structural architecture match Canonical?** None of these selected models does so convincingly. Their test point estimates are all below Canonical {r('Canonical'):.3f}%; no model passed the development effectiveness gate. Even the highest new test estimate is {delta(best_test, 'Canonical')} versus Canonical. A CI crossing zero, if present, would not establish equivalence.", "",
        f"**Q5. Can one match or exceed Original while keeping VI near zero?** No tested method does. Original Recall@100 is {r('Original'):.3f}%; the largest new test estimate is {r(best_test):.3f}%, with {delta(best_test, 'Original')} against Original. Robust and Never must be examined alongside VI; invariance alone supplies no effectiveness success.", "",
        "**Q6. Does the failure strengthen the diagnosis?** Yes, within this bounded setting: the challenge is not achieving invariance, but preserving retrieval-relevant information under invariance. Strong numerical invariance coexists with failure to recover recall and with stable misses. This is evidence about the tested frozen-BGE architectures and small supervised regime, not a universal impossibility result for invariant encoders.", "",
        "## 15. Recommendation for the WWW paper and delivery", "",
        f"**{decision}.** Retain the paper's representation-sensitivity diagnosis. Keep this fully documented negative screen as supplementary evidence; if space permits, a short comparison can distinguish scalar field weighting from actual cross-field self-attention. Do not promote S3 as novel because it uses a circle, introduce a main structural solution claim, or start Phase VII from these results. No manuscript redesign or paper-file edits were made.", "",
        "The recommendation is based on failed development gates and joint recall/Robust/Never/VI evidence, not merely on invariance. With only 17 eligible development queries, one training seed, one dataset/encoder, a historical test and a very small supervised set, external generalization remains unresolved.", "",
        "Required outputs: `PHASE6_REPORT.md`, `PHASE6_RESULTS.csv`, `PHASE6_BOOTSTRAP.csv`, `PHASE6_INVARIANCE.json`, `PHASE6_CONFIG.json`, and `FINAL_DELIVERY_MANIFEST.json`. Table VI-1 and the screening plot are also saved under `phase6/`. CSV metrics use percentages and deltas use percentage points; target rows use Inclusion columns. The manifest hashes required deliverables, code, checkpoints, complete candidate logs and raw rank artifacts, and records unchanged reused-source hashes.", "",
        "Reproduction from the existing repository (the pinned Python environment is reused):", "",
        "```powershell",
        "phase2/.venv/Scripts/python.exe -X utf8 -m pytest phase6/test_models.py phase6/test_metrics.py -q",
        "phase2/.venv/Scripts/python.exe -X utf8 phase6/run.py prepare",
        "phase2/.venv/Scripts/python.exe -X utf8 phase6/run.py train",
        "phase2/.venv/Scripts/python.exe -X utf8 phase6/run.py evaluate",
        "phase2/.venv/Scripts/python.exe -X utf8 phase6/analyze.py",
        "phase2/.venv/Scripts/python.exe -X utf8 phase6/verify.py",
        "phase2/.venv/Scripts/python.exe -X utf8 phase6/write_report.py",
        "phase2/.venv/Scripts/python.exe -X utf8 phase6/analyze.py manifest",
        "```", "",
        "Completed checkpoints and evaluation outputs are reused. Frozen input/code hashes reject silent changes. Use a separately documented phase or protocol amendment for any future training redesign; do not overwrite negative candidates or reuse this held-out test as a tuning set.", "",
    ]
    (ROOT / "PHASE6_REPORT.md").write_text("\n".join(lines), encoding="utf8")
    dump(HERE / "decision.json", dict(decision=decision, rationale="All core development gates failed; exact invariance did not preserve effectiveness",
                                      report_generated_utc=now(), selection_sha256=sha(HERE / "selection.json")))
    print(decision)


if __name__ == "__main__":
    main()
