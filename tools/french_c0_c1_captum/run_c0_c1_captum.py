"""Run the frozen French-molding C0 -> C1 case with official Captum IG.

Run in the original experiment environment. No model/cache downloads, training,
or changes to existing results. A new output directory is required each time.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import traceback
from audit_support import audit_original_inputs

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_write(path, rows):
    if not rows:
        raise ValueError("Empty result table: " + str(path))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def serialize_with_spans(common, product, schedule):
    indices = list(range(len(product["attributes"])))
    if schedule == "C1":
        indices.reverse()
    text, parts = "", []
    for field, value in common.rep._blocks(
            product, [product["attributes"][i] for i in indices]):
        if text:
            parts.append((len(text), len(text) + 1, "separator", "section separator"))
            text += "\n"
        if field == "attributes":
            for j, occurrence in enumerate(indices):
                if j:
                    sep = product["attribute_separator"]
                    parts.append((len(text), len(text) + len(sep), "separator", "attribute separator"))
                    text += sep
                value = product["attributes"][occurrence]
                parts.append((len(text), len(text) + len(value),
                              f"attr_{occurrence:03d}", value))
                text += value
        else:
            parts.append((len(text), len(text) + len(value), "fixed_" + field, value))
            text += value
    if text != common.serialize(product, schedule):
        raise ValueError("Serializer disagreement: " + schedule)
    return text, parts


def run(args, out):
    import numpy as np
    import torch
    import transformers
    import captum
    from transformers import AutoModel, AutoTokenizer
    from captum.attr import IntegratedGradients

    if not torch.cuda.is_available():
        raise RuntimeError("Use the original CUDA experiment environment; no automatic CPU/profile substitution.")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    repo = args.repo.resolve()
    environment = audit_original_inputs(repo, out, HERE)
    dump(out / 'environment.json', environment)
    fixture = read(HERE / "case_inputs.json")
    helper = repo / "revision_final_strengthening_20260922/scripts/common.py"
    spec = importlib.util.spec_from_file_location("c0_c1_frozen_common", helper)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    common.verify()
    model_spec = common.model_spec("minilm")
    expected_revision = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    if (model_spec["revision"] != expected_revision or model_spec["max_tokens"] != 256
            or model_spec['name'] != 'sentence-transformers/all-MiniLM-L6-v2'
            or model_spec['native_pooling'] != 'attention-mask mean'):
        raise ValueError("Model revision or tokenizer cap differs from the frozen case.")
    products, queries = common.load("wands")
    product_index = next(i for i, p in enumerate(products) if str(p["product_id"]) == "12575")
    query_index = next(i for i, qid in enumerate(queries.query_id) if int(qid) == 359)
    product = products[product_index]
    if str(queries.iloc[query_index]["query"]) != fixture["query"]:
        raise ValueError("Query text mismatch")
    if list(product["attributes"]) != fixture["source_attributes"]:
        raise ValueError("Source attribute contents/order mismatch")
    if len(products) != 42994:
        raise ValueError("Catalog membership differs from the frozen case")

    provenance = common.provenance()
    sources = provenance[(provenance.dataset == "wands") &
                         (provenance.model == "minilm")].set_index("schedule")
    qpath = repo / str(sources.loc["C0", "query_embedding"])
    qcache = np.load(qpath, mmap_mode="r")
    q = np.array(qcache[query_index], dtype=np.float32, copy=True)
    if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-5):
        raise ValueError("Cached query is not normalized")
    qv = torch.from_numpy(q).cuda()
    p0path = repo / str(sources.loc["C0", "product_embedding"])
    p0 = np.load(p0path, mmap_mode="r")
    if qcache.shape != (480, 384) or p0.shape != (42994, 384):
        raise ValueError('Original cache dimensions changed')
    # Match boundary.py: the original full-query FP32 GEMM, not a new GEMV.
    # This only scores existing caches; it does not encode queries or rerun a benchmark.
    base_scores = (qcache @ p0.T)[query_index]
    indices = np.arange(len(products))
    competitors = indices[indices != product_index]
    order = competitors[np.lexsort((competitors, -base_scores[competitors]))]
    threshold_index = int(order[19])
    threshold = float(base_scores[threshold_index])
    native = {v["schedule"]: v for v in fixture["native_records"]}
    cache_checks = []
    paths = {"query": qpath, "C0_products": p0path, "common": helper}
    target_path = repo / 'phase3/results/target_only_permutations/wands_minilm_pairs.parquet'
    target_frame = common.pairread(target_path)
    target_row = target_frame[(target_frame.query_id == 359) &
                              (target_frame.product_id == '12575')].iloc[0]
    paths['target_only_ranks'] = target_path
    for schedule in ("C0", "C1"):
        old = native[schedule]
        if product_index != old["catalog_index"]:
            raise ValueError("Stable catalog order differs from the frozen case")
        path = repo / str(sources.loc[schedule, "product_embedding"])
        paths[schedule + "_products"] = path
        matrix = np.load(path, mmap_mode="r")
        cache_dot_score = float(q @ matrix[product_index])
        rank_path = repo / str(sources.loc[schedule, 'rank_source'])
        saved_pairs = common.pairread(rank_path)
        saved_pair = saved_pairs[(saved_pairs.query_id == 359) &
                                  (saved_pairs.product_id == '12575')].iloc[0]
        # C1 uses the original saved target score with every C0 competitor fixed.
        score = float(base_scores[product_index]) if schedule == 'C0' else float(saved_pair.score)
        paths[schedule + '_rank_source'] = rank_path
        rank = 1 + int(np.sum((base_scores[competitors] > score) |
                             ((base_scores[competitors] == score) & (competitors < product_index))))
        ok = (rank == old["saved_rank"] == int(target_row[schedule])
              and threshold_index == old["competitor_threshold_index"]
              and threshold == old["competitor_threshold_score"]
              and score == old["score"] == float(saved_pair.score)
              and abs(cache_dot_score - old['score']) <= 2e-6)
        cache_checks.append(dict(schedule=schedule, reconstructed_score=score,
                                 cache_dot_score=cache_dot_score,
                                 reconstructed_rank=rank, threshold=threshold, passed=bool(ok),
                                 competitor_count=len(competitors),
                                 tie_break='descending saved FP32 score, ascending catalog index; no epsilon'))
        if not ok:
            dump(out / "cache_checks.json", cache_checks)
            raise ValueError("Frozen cache/rank mismatch; do not relabel a different execution profile.")
    dump(out / "cache_checks.json", cache_checks)
    np.save(out / 'fixed_query_embedding.npy', q)
    csv_write(out / 'fixed_competitors.csv', [dict(catalog_index=int(i),
              product_id=str(products[i]['product_id']), C0_score=float(base_scores[i]))
              for i in competitors])
    csv_write(out / 'attribute_sequences.csv', [dict(schedule=s, position=j,
              entry_id=f'attr_{i:03d}', entry_text=product['attributes'][i])
              for s in ('C0', 'C1') for j, i in enumerate(
                  range(10) if s == 'C0' else reversed(range(10)))])

    tokenizer = AutoTokenizer.from_pretrained(model_spec["name"], revision=expected_revision,
                                              local_files_only=True)
    model = AutoModel.from_pretrained(model_spec["name"], revision=expected_revision,
                                     local_files_only=True).cuda().half().eval()
    model.requires_grad_(False)

    def score_forward(inputs, embedding=None):
        kwargs = dict(inputs)
        if embedding is not None:
            kwargs["inputs_embeds"] = embedding
        hidden = model(**kwargs).last_hidden_state.float()
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        return torch.nn.functional.normalize(pooled, dim=1) @ qv

    prepared = []
    for schedule in ("C0", "C1"):
        text, spans = serialize_with_spans(common, product, schedule)
        attrs = fixture["source_attributes"] if schedule == "C0" else fixture["source_attributes"][::-1]
        expected_text = fixture["non_attribute_prefix"] + fixture["attribute_separator"].join(attrs)
        if text != expected_text:
            raise ValueError("Frozen input mismatch: " + schedule)
        (out / (schedule + "_input.txt")).write_text(text, encoding="utf-8", newline='\n')
        encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True,
                            return_special_tokens_mask=True, truncation=False)
        offsets = encoded.pop("offset_mapping")[0].tolist()
        special = encoded.pop("special_tokens_mask")[0].bool().cuda()
        inputs = {k: v.cuda() for k, v in encoded.items()}
        length = inputs["input_ids"].shape[1]
        if length != native[schedule]["token_length"] or length > 256:
            raise ValueError("Unexpected tokenization/truncation")
        inputs["position_ids"] = torch.arange(length, device="cuda")[None, :]
        dump(out / (schedule + '_tokenized_input.json'), dict(
            inputs={k: v.cpu().tolist() for k, v in inputs.items()},
            offsets=offsets, special_tokens_mask=special.cpu().tolist(), spans=spans,
            text_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest()))
        with torch.no_grad():
            fresh_native = float(score_forward(inputs).item())
        prepared.append((schedule, spans, offsets, special, inputs, fresh_native))
    # Match the historical diagnostic: retain native FP16-rounded parameters,
    # represented in FP32 for gradients. Query vector is unchanged.
    model.float()
    tokens, entries, checks = [], [], []
    for schedule, spans, offsets, special, inputs, fresh_native in prepared:
        emb = model.get_input_embeddings()(inputs["input_ids"]).detach()
        keep = {k: v for k, v in inputs.items() if k != "input_ids"}

        def forward_embeddings(value):
            expanded = {k: v.expand(value.shape[0], -1) for k, v in keep.items()}
            return score_forward(expanded, value)

        with torch.no_grad():
            fid = float(score_forward(inputs).item())
            actual = float(forward_embeddings(emb).item())
        if abs(fid - actual) > 1e-6:
            raise ValueError("input_ids/inputs_embeds forward disagreement")
        old = native[schedule]
        error = abs(actual - old["score"])
        included = (actual > old["competitor_threshold_score"] or
                    (actual == old["competitor_threshold_score"] and
                     product_index < old["competitor_threshold_index"]))
        fresh_included = (fresh_native > old["competitor_threshold_score"] or
                          (fresh_native == old["competitor_threshold_score"] and
                           product_index < old["competitor_threshold_index"]))
        guard = (included == old["inclusion"] and fresh_included == old["inclusion"]
                 and abs(old["signed_margin"]) > 5 * error
                 and abs(old['signed_margin']) > 5 * abs(fresh_native - old['score']))
        ids = inputs["input_ids"][0].tolist()
        labels = []
        for (a, b), is_special in zip(offsets, special.tolist()):
            found = [(name, value) for start, end, name, value in spans if a >= start and b <= end and b > a]
            labels.append(("special", "special token") if is_special else
                          found[0] if len(found) == 1 else ("boundary", "token spans content boundary"))
        ig = IntegratedGradients(forward_embeddings, multiply_by_inputs=True)
        for baseline_name in ("zero", "pad"):
            baseline = emb.clone()
            change = (~special) & inputs["attention_mask"][0].bool()
            baseline[:, change, :] = (0 if baseline_name == "zero" else
                                     model.get_input_embeddings().weight[tokenizer.pad_token_id].detach())
            with torch.no_grad():
                baseline_score = float(forward_embeddings(baseline).item())
            attempts = []
            try:
                for n in (64, 128, 256):
                    attribution, delta = ig.attribute(emb, baselines=baseline, n_steps=n,
                        method="gausslegendre", internal_batch_size=8, return_convergence_delta=True)
                    token_values = attribution[0].sum(-1).detach().cpu().double().tolist()
                    residual = math.fsum(token_values) - (actual - baseline_score)
                    tolerance = max(1e-4, 0.01 * abs(actual - baseline_score))
                    passed = math.isfinite(residual) and abs(residual) <= tolerance
                    attempts.append(dict(nodes=n, residual=residual, tolerance=tolerance,
                                         captum_delta=float(delta.item()), passed=bool(passed)))
                    if passed:
                        break
            except Exception as exc:
                failure = dict(schedule=schedule, baseline=baseline_name, status='failed',
                               completeness_passed=False, boundary_guard=bool(guard),
                               error=repr(exc), traceback=traceback.format_exc(), attempts=attempts)
                checks.append(failure)
                dump(out / (schedule + '_' + baseline_name + '_qa.json'), failure)
                torch.cuda.empty_cache()
                continue
            np.savez_compressed(out / (schedule + '_' + baseline_name + '_features.npz'),
                                attribution=attribution.detach().cpu().numpy(),
                                input_embeddings=emb.cpu().numpy(), baseline_embeddings=baseline.cpu().numpy())
            groups = {}
            for i, ((name, value), number) in enumerate(zip(labels, token_values)):
                groups.setdefault((name, value), []).append(number)
                tokens.append(dict(schedule=schedule, baseline=baseline_name, token_index=i,
                                   token_id=ids[i], token=tokenizer.convert_ids_to_tokens(ids[i]),
                                   start=offsets[i][0], end=offsets[i][1], entry_id=name,
                                   entry_text=value, attribution=number))
            for (name, value), numbers in groups.items():
                entries.append(dict(schedule=schedule, baseline=baseline_name, entry_id=name,
                                    entry_text=value, attribution=math.fsum(numbers)))
            check = dict(schedule=schedule, baseline=baseline_name, nodes=n,
                         status='passed' if passed and guard else 'diagnostic_checks_failed',
                         token_count=len(ids), input_ids_vs_embeds_error=abs(fid-actual),
                         unchanged_special_and_padding=bool(torch.equal(baseline[:, ~change], emb[:, ~change])),
                         native_score=old["score"], native_rank=old["saved_rank"],
                         native_margin=old["signed_margin"], fresh_native_score=fresh_native,
                         fresh_native_included=bool(fresh_included),
                         fresh_native_error=abs(fresh_native - old["score"]), diagnostic_score=actual,
                         diagnostic_native_error=error, diagnostic_included=bool(included),
                         boundary_guard=bool(guard), baseline_score=baseline_score,
                         attribution_sum=math.fsum(token_values), completeness_residual=residual,
                         completeness_tolerance=tolerance, completeness_passed=bool(passed),
                         captum_delta=float(delta.item()), attempts=attempts)
            checks.append(check)
            dump(out / (schedule + "_" + baseline_name + "_qa.json"), check)
            csv_write(out / 'token_attributions.csv', tokens)
            csv_write(out / 'entry_attributions.csv', entries)
            print(schedule, baseline_name, "nodes", n, "completeness", passed, "boundary", guard, flush=True)
    if any(v.get('status') == 'failed' for v in checks):
        dump(out / 'run_report.json', dict(status='inference_failed', plot_ready=False,
              contrast='C1 minus C0', checks=checks, environment=environment))
        return 2
    aligned = []
    for baseline_name in ("zero", "pad"):
        by_schedule = {s: {r["entry_id"]: r for r in entries if r["schedule"] == s and
                          r["baseline"] == baseline_name} for s in ("C0", "C1")}
        for entry_id in sorted(set(by_schedule["C0"]) | set(by_schedule["C1"])):
            a = by_schedule["C0"].get(entry_id)
            b = by_schedule["C1"].get(entry_id)
            if entry_id.startswith("attr_") and (a is None or b is None):
                raise ValueError("Attribute occurrence missing; inspect token boundary mapping")
            if a and b and a["entry_text"] != b["entry_text"]:
                raise ValueError("Entry identity mismatch")
            v0 = a["attribution"] if a else 0.0
            v1 = b["attribution"] if b else 0.0
            aligned.append(dict(baseline=baseline_name, entry_id=entry_id,
                                entry_text=(a or b)["entry_text"], C0=v0, C1=v1, C1_minus_C0=v1-v0))
    csv_write(out / "aligned_entry_changes.csv", aligned)
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    ready = len(checks) == 4 and all(v["completeness_passed"] and v["boundary_guard"] for v in checks)
    for p in HERE.glob('*.py'):
        paths['runner_' + p.name] = p
    for name in ['case_inputs.json', 'PACKAGE_SOURCE.json', 'captum_install_report.json']:
        paths[name] = HERE / name
    for name in ['phase2/config/phase2.json', 'phase2/src/representations.py',
                 'revision_final_strengthening_20260922/PROTOCOL.json',
                 'revision_evidence_20260917/data/encoder_profiles_and_rank_sources.csv']:
        paths[name] = repo / name
    paths['captum_implementation'] = Path(sys.modules[IntegratedGradients.__module__].__file__)
    report = dict(status="passed" if ready else "diagnostic_checks_failed", plot_ready=ready,
                  contrast="C1 minus C0", historical_case=True, native=fixture["native_records"],
                  checks=checks, repo_head=head, torch=torch.__version__, transformers=transformers.__version__,
                  captum=captum.__version__, gpu=torch.cuda.get_device_name(0), model_spec=model_spec,
                  attention_implementation=str(model.config._attn_implementation),
                  environment=environment, query_row_index=query_index, product_catalog_index=product_index,
                  method=dict(implementation='captum.attr.IntegratedGradients', multiply_by_inputs=True,
                              integration='gausslegendre', nodes=[64, 128, 256], internal_batch_size=8),
                  tf32=False, pooling='attention-mask mean in FP32 then L2 normalization',
                  sources={k: dict(path=str(p), sha256=sha(p)) for k, p in paths.items()},
                  precision="FP16-rounded frozen model parameters in FP32 for IG; cached normalized query unchanged")
    dump(out / "run_report.json", report)
    if not ready:
        print("Saved all results and failed checks. Figure generation is withheld; retain the failure.")
        return 2
    if args.plot:
        subprocess.run([sys.executable, str(HERE / "plot_c0_c1.py"), str(out)], check=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, help="New directory; existing directories are refused")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    out = (args.output or args.repo / "revision_french_c0_c1_captum_20261001").resolve()
    if out.exists():
        ap.error("Output directory already exists. Choose a fresh --output directory.")
    out.mkdir(parents=True)
    try:
        return run(args, out)
    except Exception as exc:
        dump(out / "run_failure.json", dict(status="failed", error=repr(exc), traceback=traceback.format_exc()))
        print("Failed; see", out / "run_failure.json", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
