"""Phase VI: freeze -> development-only screening/seed gate -> locked test evaluation."""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import os
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "phase5_mitigation"))
import screen as prior
from models import StructuralAggregator, field_key

STEMS = prior.STEMS
BASELINES = ["Original", "Canonical", "Set-Mean", "Set-Attention_s42"]
METHODS = ["S1", "S2", "S3", "S4"]
CONFIG = ROOT / "PHASE6_CONFIG.json"
OUT = HERE / "results"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")
    tmp.replace(path)


def now():
    return pd.Timestamp.now(tz="UTC").isoformat()


def cost(run, stage, seconds, **extra):
    with (HERE / "cost.jsonl").open("a", encoding="utf8") as stream:
        stream.write(json.dumps(dict(run=run, stage=stage, seconds=seconds, utc=now(), **extra)) + "\n")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Data:
    def __init__(self, config=None):
        self.products, self.queries, self.judgments, self.pi, self.qi = prior.data()
        self.non, self.attributes, self.indices, self.unique = prior.field_data()
        self.lookup = {a: i for i, a in enumerate(self.unique)}
        self.queries_v, _ = prior.original_cache("wands", "bge_base")
        self.config = config or json.loads((ROOT / "phase5_mitigation/config.json").read_text())
        self.triples = json.loads((ROOT / "phase5_mitigation/triples.json").read_text())
        train_products = {t[k] for t in self.triples for k in ("positive", "negative")}
        keys = sorted({field_key(a) for pid in train_products
                       for a in self.products[self.pi[pid]]["attributes"] if field_key(a) is not None})
        self.vocab = {key: i + 1 for i, key in enumerate(keys)}
        self.atom_fields = np.asarray([self.vocab.get(field_key(a), 0) for a in self.unique], dtype="i8")
        self.lengths = np.asarray(list(map(len, self.indices)))
        self.infer_order = np.argsort(self.lengths, kind="stable")
        self.sample = sorted(range(len(self.products)), key=lambda i:
                             hashlib.sha256(f'phase6-invariance42:{self.products[i]["product_id"]}'.encode()).hexdigest())[:128]
        self.exact = self.judgments[self.judgments.label.eq("Exact")][["query_id", "product_id"]].copy()
        self.exact["product_id"] = self.exact.product_id.astype(str)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.at = torch.tensor(self.attributes, device=self.device)
        self.nt = torch.tensor(self.non, device=self.device)
        self.qt = torch.tensor(self.queries_v, device=self.device)
        self.ft = torch.tensor(self.atom_fields, device=self.device)

    def atoms(self, index, schedule):
        p = self.products[index]
        if schedule == "C0":
            return list(p["attributes"])
        if schedule == "C1":
            return list(reversed(p["attributes"]))
        seed = prior.CFG["attribute_permutation_seeds"][int(schedule[-1]) - 1]
        return prior._random_order(p["attributes"], str(p["product_id"]), seed)

    def batch(self, ids, schedule="C0"):
        lists = [self.indices[i] if schedule == "C0" else [self.lookup[a] for a in self.atoms(i, schedule)] for i in ids]
        width = max(1, max(map(len, lists)))
        ix = np.zeros((len(ids), width), dtype="i8")
        mask = np.zeros_like(ix, dtype=bool)
        for row, values in enumerate(lists):
            ix[row, :len(values)] = values
            mask[row, :len(values)] = True
        index = torch.tensor(ix, device=self.device)
        return self.at[index], torch.tensor(mask, device=self.device), self.nt[list(ids)], self.ft[index]

    def frame(self, product_vectors, split):
        f = self.exact[self.exact.query_id.isin(self.config[split])].reset_index(drop=True)
        qids = sorted(f.query_id.unique())
        scores = self.queries_v[[self.qi[int(q)] for q in qids]] @ product_vectors.T
        rr = prior.ranks(scores)[1]
        local = {qid: i for i, qid in enumerate(qids)}
        ar = rr[[local[q] for q in f.query_id], [self.pi[p] for p in f.product_id]]
        for schedule in STEMS:
            f[schedule] = ar
        return f

    def invariance(self, model, name):
        model.eval()
        vectors, traces = [], []
        with torch.inference_mode():
            for schedule, condition in zip(STEMS, prior.CFG["primary_variant_family"]):
                atoms = [self.atoms(i, schedule) for i in self.sample]
                for i, ordered in zip(self.sample, atoms):
                    assert Counter(ordered) == Counter(self.products[i]["attributes"])
                    assert prior._plain(self.products[i], ordered) == prior.build(
                        self.products[i], condition, prior.CFG["attribute_permutation_seeds"])
                traces.append([hashlib.sha256(json.dumps(a, ensure_ascii=False).encode()).hexdigest() for a in atoms])
                vectors.append(model(*self.batch(self.sample, schedule)).cpu().numpy())
        v = np.stack(vectors)
        l2 = np.linalg.norm(v - v[0:1], axis=2).astype(float)
        vd = v.astype("f8")
        cos = np.abs(1 - (vd * vd[0:1]).sum(2) / (np.linalg.norm(vd, axis=2) * np.linalg.norm(vd[0:1], axis=2)))
        record = dict(name=name, products=len(self.sample), schedules=STEMS,
                      independent_forward_passes=7, comparisons_including_C0=int(l2.size),
                      l2_max=float(l2.max()), l2_mean=float(l2.mean()),
                      cosine_difference_max=float(cos.max()), cosine_difference_mean=float(cos.mean()),
                      product_ids=[str(self.products[i]["product_id"]) for i in self.sample],
                      schedule_atom_hashes={s: t for s, t in zip(STEMS, traces)},
                      products_with_changed_order=sum(len({t[i] for t in traces}) > 1 for i in range(len(self.sample))),
                      finite=bool(np.isfinite(v).all()), passed=bool(np.isfinite(v).all() and l2.max() <= 1e-5))
        dump(HERE / "invariance" / (name + ".json"), record)
        assert record["products_with_changed_order"] >= 100, "Schedules are not actually permuting inputs"
        assert record["passed"], "STOP: permutation invariance failed; retrieval has not run"
        return record

    def generate(self, model, schedule="C0"):
        model.eval()
        result = np.empty_like(self.non)
        with torch.inference_mode():
            for offset in range(0, len(self.products), 128):
                ids = self.infer_order[offset:offset + 128].tolist()
                result[ids] = model(*self.batch(ids, schedule)).cpu().numpy()
        assert np.isfinite(result).all()
        assert np.max(np.abs(np.linalg.norm(result, axis=1) - 1)) < 1e-5
        return result


def prepare():
    if CONFIG.exists():
        return validate_config()
    started = time.monotonic()
    d = Data()
    old = d.config
    assert len(d.products) == 42994 and len(d.triples) == 546
    train, dev, test = (set(old[k]) for k in ("train", "validation", "test"))
    assert not (train & dev or train & test or dev & test)
    assert {x["query_id"] for x in d.triples} <= train
    labels = d.judgments.assign(product_id=d.judgments.product_id.astype(str)).set_index(["query_id", "product_id"]).label
    for t in d.triples:
        assert labels.loc[t["query_id"], t["positive"]] == "Exact"
        assert labels.loc[t["query_id"], t["negative"]] == "Irrelevant"
    cache_dir = ROOT / "phase3/results/phase3_invariant_method/embeddings"
    nontext = ["\n".join(t for t in [p["title"], p.get("class", ""), p.get("category", ""), p.get("description", "")] if t)
               for p in d.products]
    cache_records = []
    source_paths = [ROOT / "phase5_mitigation/config.json", ROOT / "phase5_mitigation/triples.json",
                    ROOT / "phase4/config/splits.json", ROOT / "phase4/results/wands_product_features.csv",
                    ROOT / "phase5_mitigation/SET_ATTENTION_REPORT.md", ROOT / "phase5_pift/PI_FT_REPORT.md"]
    source_paths += [ROOT / f"phase2/data/processed/wands_{f}" for f in ("products.jsonl.gz", "queries.csv", "judgments.csv")]
    for key, texts in [("wands_unique_attributes", d.unique), ("wands_title_description", nontext)]:
        signature = hashlib.sha256("\0".join(texts).encode()).hexdigest()
        matches = [f for f in cache_dir.glob(key + "_*.json") if json.loads(f.read_text())["source_sha256"] == signature]
        assert len(matches) == 1
        meta = json.loads(matches[0].read_text())
        assert meta["model"] == prior.SPEC
        a = np.load(matches[0].with_suffix(".npy"), mmap_mode="r")
        assert list(a.shape) == [len(texts), 768] and a.dtype == np.float32
        assert np.max(np.abs(np.linalg.norm(a, axis=1) - 1)) < 1e-5
        cache_records.append(dict(path=str(matches[0].relative_to(ROOT)), metadata=meta))
        source_paths += [matches[0], matches[0].with_suffix(".npy")]
    prov = json.loads((ROOT / "phase3/results/target_only_permutations/provenance.json").read_text())
    entry = next(e for e in prov["sources"] if e["dataset"] == "wands" and e["model"] == "bge_base")
    query_path = ROOT / entry["query_embedding"]
    query_meta = json.loads(query_path.with_suffix(".json").read_text())
    assert query_meta["model"] == prior.SPEC
    query_text = d.queries["query"].fillna("").astype(str).tolist()
    assert query_meta["source_sha256"] == hashlib.sha256("\0".join(query_text).encode()).hexdigest()
    assert d.queries_v.shape == (480, 768) and d.queries_v.dtype == np.float32
    assert np.max(np.abs(np.linalg.norm(d.queries_v, axis=1) - 1)) < 1e-5
    source_paths += [query_path, query_path.with_suffix(".json")]
    source_paths += [ROOT / f"phase5_mitigation/results/{b}/{f}" for b in BASELINES
                     for f in ("catalog_pairs.parquet", "catalog_queries.csv", "target_pairs.parquet")]
    source_paths += [ROOT / "phase5_mitigation/checkpoints/Set-Attention_s42/weights.pt"]
    source_paths += [HERE / "PROTOCOL.md", HERE / "models.py", HERE / "run.py"]
    source_paths += [ROOT / f for f in ("phase5_mitigation/screen.py", "phase4/scripts/common.py",
                                      "phase2/src/representations.py", "phase2/src/encoding.py",
                                      "phase3/scripts/run_reranker.py", "phase3/scripts/analyze_target_only_permutations.py")]
    dev_metrics = {}
    for name in BASELINES:
        df = pd.read_csv(ROOT / f"phase5_mitigation/results/{name}/catalog_queries.csv")
        df = df[df.query_id.isin(dev)]
        assert len(df) == 17
        dev_metrics[name] = float(df["Recall@100"].mean())
    vocab = dict(UNK_ID=0, keys=d.vocab, scope="native keys from products in the 546 training triples only")
    dump(HERE / "field_vocabulary.json", vocab)
    source_paths.append(HERE / "field_vocabulary.json")
    cfg = dict(phase=6, frozen_utc=now(), track="raw_WANDS_BGE", train=old["train"], validation=old["validation"], test=old["test"],
               catalog_products=42994, training_triples=546, eligible_queries={k: int(d.exact[d.exact.query_id.isin(old[k])].query_id.nunique())
                                                                            for k in ("train", "validation", "test")},
               model=prior.SPEC, schedules=STEMS, permutation_seeds=prior.CFG["attribute_permutation_seeds"],
               methods=METHODS, hidden=128, blocks=1, heads=4, learning_rates=[.001, .0003], max_epochs=10,
               patience=3, batch_size=16, optimizer="AdamW", weight_decay=.01, gradient_clip=1., temperature=.05,
               seed42_first=True, extra_seeds=[43, 44], seed_gate="S1/S2/S3 dev Recall@100 > archived Set-Attention and invariance passes",
               lr_search_stop="Stop remaining LR candidates for a core method once its dev R100 exceeds Set-Attention and Canonical; immediately confirm seeds",
               output_dim=768, non_attribute_mix=.5, precision="FP32 aggregator and scoring; original cached BGE FP16 forward/FP32 normalized pooling",
               tie_rule="descending FP32 dot product, then stable source catalog index", dev_baselines=dev_metrics,
               bootstrap_draws=10000, bootstrap_seed=20260963, invariance_products=128, invariance_max_l2=1e-5,
               field_vocab_size=len(d.vocab) + 1, field_vocab_scope=vocab["scope"],
               catalog_unique_nonempty_field_keys=len({field_key(a) for a in d.unique if field_key(a) is not None}),
               unknown_unique_atoms=int((d.atom_fields == 0).sum()), duplicate_atom_products=int(sum(len(v) != len(set(v)) for v in d.indices)),
               query_cache=str(query_path.relative_to(ROOT)), field_caches=cache_records,
               trainable_encoder_parameters=0, test_selection_prohibited=True,
               environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
                                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None),
               sources=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in source_paths])
    dump(CONFIG, cfg)
    cost("prepare", "cache_validation_and_setup", time.monotonic() - started)
    print("FROZEN", cfg["eligible_queries"], "vocabulary", cfg["field_vocab_size"], "dev baselines", dev_metrics, flush=True)
    return cfg


def validate_config():
    cfg = json.loads(CONFIG.read_text(encoding="utf8"))
    for source in cfg["sources"]:
        assert sha(ROOT / source["path"]) == source["sha256"], "Frozen source changed: " + source["path"]
    return cfg


def model_for(d, method, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return StructuralAggregator(method, len(d.vocab) + 1, hidden=d.config["hidden"], blocks=d.config["blocks"]).to(d.device)


def candidate(d, method, seed, lr):
    name = f"{method}_s{seed}_lr{lr:g}"
    folder = HERE / "checkpoints" / name
    metadata = folder / "selected.json"
    if metadata.exists():
        meta = json.loads(metadata.read_text())
        assert meta["config_sha256"] == sha(CONFIG)
        assert meta["weights_sha256"] == sha(folder / "weights.pt")
        return meta
    assert not folder.exists(), "Incomplete candidate retained; inspect before resuming it"
    folder.mkdir(parents=True)
    model = model_for(d, method, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=d.config["weight_decay"])
    rng = random.Random(seed)
    logs = []
    best, best_epoch, stale = -1., None, 0
    wall = time.monotonic()
    try:
        for epoch in range(1, d.config["max_epochs"] + 1):
            model.train()
            examples = list(d.triples)
            rng.shuffle(examples)
            sync()
            start = time.monotonic()
            losses = []
            for offset in range(0, len(examples), d.config["batch_size"]):
                g = examples[offset:offset + d.config["batch_size"]]
                pos = model(*d.batch([d.pi[x["positive"]] for x in g]))
                neg = model(*d.batch([d.pi[x["negative"]] for x in g]))
                query = d.qt[[d.qi[x["query_id"]] for x in g]]
                loss = F.softplus(((query * neg).sum(1) - (query * pos).sum(1)) / .05).mean()
                assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                opt.step()
                losses.append((loss.item(), len(g)))
            sync()
            training_s = time.monotonic() - start
            cost(name, "training", training_s, epoch=epoch)
            start = time.monotonic()
            inv = d.invariance(model, f"{name}_epoch{epoch}")
            cost(name, "invariance", time.monotonic() - start, epoch=epoch)
            start = time.monotonic()
            pv = d.generate(model)
            generation_s = time.monotonic() - start
            cost(name, "dev_catalog_generation", generation_s, epoch=epoch)
            start = time.monotonic()
            frame = d.frame(pv, "validation")
            _, pq = prior.metric_frame(frame)
            score = float(pq["Recall@100"].mean())
            ranking_s = time.monotonic() - start
            cost(name, "dev_ranking", ranking_s, epoch=epoch)
            logs.append(dict(epoch=epoch, loss=sum(v * n for v, n in losses) / sum(n for _, n in losses),
                             dev_recall100=score, training_seconds=training_s, generation_seconds=generation_s,
                             ranking_seconds=ranking_s, invariance_max_l2=inv["l2_max"]))
            pd.DataFrame(logs).to_csv(folder / "epochs.csv", index=False)
            print(name, "epoch", epoch, "dev R100", round(100 * score, 4), "loss", round(logs[-1]["loss"], 5),
                  "seconds", round(training_s + generation_s + ranking_s, 2), flush=True)
            if score > best:
                best, best_epoch, stale = score, epoch, 0
                torch.save(model.state_dict(), folder / "weights.pt")
                frame.to_parquet(folder / "dev_pairs.parquet", index=False)
                pq.to_csv(folder / "dev_queries.csv")
                dump(folder / "invariance.json", inv)
            else:
                stale += 1
            if stale >= d.config["patience"]:
                break
        meta = dict(name=name, method=method, seed=seed, lr=lr, best_epoch=best_epoch, epochs_run=len(logs),
                    dev_recall100=best, parameters=sum(p.numel() for p in model.parameters()),
                    config_sha256=sha(CONFIG), weights_sha256=sha(folder / "weights.pt"), test_used=False,
                    training_seconds=sum(e["training_seconds"] for e in logs), completed_utc=now(),
                    wall_seconds=time.monotonic() - wall)
        dump(metadata, meta)
        return meta
    except BaseException as exc:
        dump(folder / "interrupted.json", dict(error=repr(exc), utc=now(), wall_seconds=time.monotonic() - wall))
        raise


def train_all():
    cfg = validate_config()
    if (HERE / "selection.json").exists():
        print("Selection already locked; training skipped", flush=True)
        return
    d = Data(cfg)
    results, selected, gates = [], {}, {}
    for method in METHODS:
        trials = []
        for lr in cfg["learning_rates"]:
            result = candidate(d, method, 42, lr)
            results.append(result)
            trials.append(result)
            if method != "S4" and result["dev_recall100"] > max(cfg["dev_baselines"]["Set-Attention_s42"], cfg["dev_baselines"]["Canonical"]):
                break
        winner = sorted(trials, key=lambda t: (-t["dev_recall100"], t["best_epoch"], cfg["learning_rates"].index(t["lr"])))[0]
        selected[method + "_s42"] = winner
        inv = json.loads((HERE / "checkpoints" / winner["name"] / "invariance.json").read_text())
        gate = method != "S4" and winner["dev_recall100"] > cfg["dev_baselines"]["Set-Attention_s42"] and inv["passed"]
        gates[method] = dict(passed=gate, dev_recall100=winner["dev_recall100"], threshold=cfg["dev_baselines"]["Set-Attention_s42"],
                             reason="S4 outside core seed gate" if method == "S4" else "development only; no test evaluated", extra_seeds=[])
        dump(HERE / "seed_gate_progress.json", gates)
        if gate:
            for seed in cfg["extra_seeds"]:
                result = candidate(d, method, seed, winner["lr"])
                results.append(result)
                selected[f"{method}_s{seed}"] = result
                gates[method]["extra_seeds"].append(seed)
                dump(HERE / "seed_gate_progress.json", gates)
    # Target-only selection is irrevocably made from development, before opening test results.
    core = [m for m in ("S1", "S2", "S3") if gates[m]["passed"]]
    diagnostic = not core
    candidates = core or ["S1", "S2", "S3"]
    target_method = max(candidates, key=lambda m: selected[m + "_s42"]["dev_recall100"])
    lock = dict(locked_utc=now(), config_sha256=sha(CONFIG), candidates=results, selected=selected, gates=gates,
                target_only_method=target_method, target_only_negative_diagnostic=diagnostic, test_used=False)
    dump(HERE / "selection.json", lock)
    pd.DataFrame(results).to_csv(HERE / "development_candidates.csv", index=False)
    print("SELECTION LOCKED", {k: (v["name"], v["dev_recall100"]) for k, v in selected.items()}, flush=True)


def evaluate_all():
    cfg = validate_config()
    selection = json.loads((HERE / "selection.json").read_text())
    assert selection["config_sha256"] == sha(CONFIG) and not selection["test_used"]
    selection_hash = sha(HERE / "selection.json")
    d = Data(cfg)
    inv_summary = dict(config_sha256=sha(CONFIG), selection_sha256=selection_hash, models={})
    for key, selected in selection["selected"].items():
        folder = OUT / key
        complete = folder / "evaluation.json"
        if complete.exists():
            meta = json.loads(complete.read_text())
            assert meta["selection_sha256"] == selection_hash
            inv_summary["models"][key] = json.loads((folder / "invariance.json").read_text())
            continue
        folder.mkdir(parents=True, exist_ok=True)
        model = model_for(d, selected["method"], selected["seed"])
        ck = HERE / "checkpoints" / selected["name"] / "weights.pt"
        assert sha(ck) == selected["weights_sha256"]
        model.load_state_dict(torch.load(ck, map_location=d.device, weights_only=True))
        inv = d.invariance(model, key + "_selected")
        start = time.monotonic()
        vectors = d.generate(model)
        generation_s = time.monotonic() - start
        np.save(folder / "products.npy", vectors)
        cost(key, "final_catalog_generation", generation_s)
        start = time.monotonic()
        frame = d.frame(vectors, "test")
        frame.to_parquet(folder / "catalog_pairs.parquet", index=False)
        _, pq = prior.metric_frame(frame)
        pq.to_csv(folder / "catalog_queries.csv")
        cost(key, "test_ranking", time.monotonic() - start)
        # A second evaluation retains seven independently recomputed catalog representations.
        regenerated = frame[["query_id", "product_id", "C0"]].copy()
        full_audit = []
        do_target = selected["method"] == selection["target_only_method"]
        target = frame[["query_id", "product_id", "C0"]].copy() if do_target else None
        qids = sorted(frame.query_id.unique())
        qmap = {q: i for i, q in enumerate(qids)}
        qv = d.queries_v[[d.qi[int(q)] for q in qids]]
        scores_c0 = qv @ vectors.T if do_target else None
        for schedule in STEMS[1:]:
            start = time.monotonic()
            other = d.generate(model, schedule)
            difference = np.linalg.norm(vectors - other, axis=1)
            assert difference.max() <= 1e-5, "STOP: full catalog invariance check failed"
            other_frame = d.frame(other, "test")
            regenerated[schedule] = other_frame.C0
            full_audit.append(dict(schedule=schedule, l2_max=float(difference.max()), l2_mean=float(difference.mean()),
                                   changed_exact_ranks=int((other_frame.C0 != frame.C0).sum()),
                                   changed_inclusion20=int(((other_frame.C0 <= 20) != (frame.C0 <= 20)).sum()),
                                   changed_inclusion100=int(((other_frame.C0 <= 100) != (frame.C0 <= 100)).sum())))
            if do_target:
                scores = qv @ other.T
                values = np.empty(len(frame), dtype="i4")
                for qid, group in frame.groupby("query_id", sort=False):
                    ix = np.array([d.pi[p] for p in group.product_id])
                    values[group.index] = prior.rank_target(scores_c0[qmap[qid]], ix, scores[qmap[qid], ix])
                target[schedule] = values
            cost(key, "independent_schedule_audit", time.monotonic() - start, schedule=schedule)
        regenerated.to_parquet(folder / "regenerated_catalog_pairs.parquet", index=False)
        _, regenerated_pq = prior.metric_frame(regenerated)
        regenerated_pq.to_csv(folder / "regenerated_catalog_queries.csv")
        if do_target:
            # Verify the inherited target reinsertion routine agrees exactly for C0.
            for qid, group in frame.groupby("query_id", sort=False):
                ix = np.array([d.pi[p] for p in group.product_id])
                check = prior.rank_target(scores_c0[qmap[qid]], ix, scores_c0[qmap[qid], ix])
                assert np.array_equal(check, frame.loc[group.index, "C0"])
            target.to_parquet(folder / "regenerated_target_pairs.parquet", index=False)
            # Deployment uses the one cached invariant vector, identical across schedules.
            frame.to_parquet(folder / "target_pairs.parquet", index=False)
            pq.to_csv(folder / "target_queries.csv")
        inv["full_catalog_regeneration"] = full_audit
        inv["regenerated_test_metrics"] = {k: float(regenerated_pq[k].mean()) for k in ("Recall@100", "VI@20", "VI@100")}
        dump(folder / "invariance.json", inv)
        inv_summary["models"][key] = inv
        dump(complete, dict(completed_utc=now(), selection_sha256=selection_hash, checkpoint_sha256=sha(ck),
                            products_sha256=sha(folder / "products.npy"), final_generation_seconds=generation_s,
                            target_only=do_target, test_queries=len(pq), exact_pairs=len(frame),
                            primary_metrics="one cached invariant C0 product vector; independently regenerated ranks audited separately"))
        dump(ROOT / "PHASE6_INVARIANCE.json", inv_summary)
        print("EVALUATED", key, "test R100", round(100 * pq["Recall@100"].mean(), 4),
              "regenerated VI100", regenerated_pq["VI@100"].mean(), flush=True)
    dump(ROOT / "PHASE6_INVARIANCE.json", inv_summary)
    dump(HERE / "evaluation_complete.json", dict(completed_utc=now(), selection_sha256=selection_hash,
                                                models=list(selection["selected"])))


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "train", "evaluate"])
    action = parser.parse_args().action
    {"prepare": prepare, "train": train_all, "evaluate": evaluate_all}[action]()


if __name__ == "__main__":
    main()
