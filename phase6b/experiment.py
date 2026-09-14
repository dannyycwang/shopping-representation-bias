"""Bounded, judged-only VI-B training; all mutations stay in phase6b/."""
from pathlib import Path
from collections import Counter
import argparse
import json
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "phase6"))
import run as p6
from residual_models import ResidualAggregator

CONFIG = HERE / "PHASE6B_CONFIG.json"
OUT = HERE / "results"
STEMS = p6.STEMS
prior = p6.prior
dump, sha, now = p6.dump, p6.sha, p6.now
BASE_CK = ROOT / "phase5_mitigation/checkpoints/Set-Attention_s42/weights.pt"
FAMILIES = {"D": "DeepSets", "S": "SetTransformer"}
STATS = ["base_attribute_norm", "delta_norm", "delta_to_base", "scaled_delta_to_base", "final_base_cosine", "alpha"]


def cost(name, stage, start, **extra):
    with (HERE / "cost.jsonl").open("a", encoding="utf8") as stream:
        stream.write(json.dumps(dict(name=name, stage=stage, seconds=time.monotonic() - start, **extra)) + "\n")


def validate():
    cfg = json.loads(CONFIG.read_text(encoding="utf8"))
    for entry in cfg["sources"]:
        assert sha(ROOT / entry["path"]) == entry["sha256"], entry["path"]
    for entry in json.loads((HERE / "historical_sources.json").read_text(encoding="utf8")):
        assert sha(ROOT / entry["path"]) == entry["sha256"], "Historical artifact changed: " + entry["path"]
    return cfg


class Data(p6.Data):
    def invariance(self, model, name):
        model.eval()
        vectors, traces = [], {}
        with torch.inference_mode():
            for schedule, condition in zip(STEMS, prior.CFG["primary_variant_family"]):
                atoms = [self.atoms(i, schedule) for i in self.sample]
                for i, ordered in zip(self.sample, atoms):
                    assert Counter(ordered) == Counter(self.products[i]["attributes"])
                    assert prior._plain(self.products[i], ordered) == prior.build(self.products[i], condition, prior.CFG["attribute_permutation_seeds"])
                traces[schedule] = [p6.hashlib.sha256(json.dumps(a).encode()).hexdigest() for a in atoms]
                vectors.append(model(*self.batch(self.sample, schedule)).cpu().numpy())
        v = np.stack(vectors)
        diff = np.linalg.norm(v - v[:1], axis=2)
        vd = v.astype("f8")
        cos = abs(1 - (vd * vd[:1]).sum(2) / (np.linalg.norm(vd, axis=2) * np.linalg.norm(vd[:1], axis=2)))
        record = dict(name=name, products=128, schedules=STEMS, l2_max=float(diff.max()), l2_mean=float(diff.mean()),
                      cosine_difference_max=float(cos.max()), cosine_difference_mean=float(cos.mean()),
                      product_ids=[str(self.products[i]["product_id"]) for i in self.sample], schedule_atom_hashes=traces,
                      products_with_changed_order=sum(len({traces[s][i] for s in STEMS}) > 1 for i in range(128)),
                      passed=bool(np.isfinite(v).all() and diff.max() <= 1e-5), base_recomputed=True, residual_recomputed=True)
        dump(HERE / "invariance" / f"{name}.json", record)
        assert record["passed"] and record["products_with_changed_order"] >= 100, "STOP: invariance failed before retrieval"
        return record

    def generate_stats(self, model):
        model.eval()
        vectors = np.empty_like(self.non)
        stats = np.empty((len(self.products), len(STATS)), dtype="f4")
        with torch.inference_mode():
            for offset in range(0, len(self.products), 128):
                ids = self.infer_order[offset:offset + 128].tolist()
                final, base, base_attr, delta, alpha = model.components(*self.batch(ids))
                bn, dn = base_attr.norm(dim=1), delta.norm(dim=1)
                values = torch.stack((bn, dn, dn / bn.clamp(min=1e-12), alpha * dn / bn.clamp(min=1e-12),
                                      torch.nn.functional.cosine_similarity(final, base), alpha.expand(len(ids))), dim=1)
                vectors[ids] = final.cpu().numpy()
                stats[ids] = values.cpu().numpy()
        assert np.isfinite(vectors).all() and np.isfinite(stats).all()
        assert abs(np.linalg.norm(vectors, axis=1) - 1).max() < 1e-5
        return vectors, pd.DataFrame(stats, columns=STATS)


def model_for(data, family, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    state = torch.load(BASE_CK, map_location="cpu", weights_only=True)
    return ResidualAggregator(FAMILIES[family], len(data.vocab) + 1, state).to(data.device)


def metrics(frame):
    _, pq = prior.metric_frame(frame)
    return pq, {m: float(pq[m].mean()) for m in ["Recall@20", "Recall@100", "Robust@100", "VI@20", "VI@100", "Never@100"]}


def prepare():
    if CONFIG.exists():
        return validate()
    p6.validate_config()
    old = json.loads((ROOT / "PHASE6_CONFIG.json").read_text(encoding="utf8"))
    audit = json.loads((HERE / "PHASE6B_TRAINING_AUDIT.json").read_text())
    sources = [HERE / n for n in ["PROTOCOL.md", "experiment.py", "residual_models.py", "data_audit.py", "test_residual.py",
                                 "PHASE6B_TRAINING_AUDIT.json", "historical_sources.json", "PHASE6B_USER_INSTRUCTIONS.txt"]]
    sources += list((HERE / "data").glob("*"))
    cfg = dict(phase="VI-B", frozen_utc=now(), track="raw_WANDS_BGE", train=old["train"], validation=old["validation"], test=old["test"],
               model=old["model"], hidden=128, blocks=1, heads=4, frozen_field_vocabulary=old["field_vocab_size"],
               architectures=FAMILIES, regimes=["T0", "T1", "T2"], learning_rates=[1e-4, 3e-4], max_epochs=20, patience=4,
               batch_size=16, temperature=.05, weight_decay=.01, gradient_clip=1., improvement_tolerance=1e-12,
               alpha_initial=.05, alpha="sigmoid(shared scalar beta)", zero_initialized_output=True,
               residual_formula="normalize(.5*non + .5*normalize(normalize(SetAttentionWeightedFieldSum) + sigmoid(beta)*raw_branch_output))",
               frozen_base=str(BASE_CK.relative_to(ROOT)), base_trainable=False, encoder_trainable=False,
               epoch_zero_is_candidate=True, initialization_test_only_identity_guard=True, init_max_l2=1e-5, init_max_recall_drop_pp=.05,
               dev_baselines=old["dev_baselines"], seed=42, confirmation_seeds=[43, 44],
               seed_gate="Each architecture's primary selected winner dev R100 >= Canonical, with invariance; inherited base alone is not learned gain",
               target_gate="Primary winner/confirmation seed: best_epoch > 0 and dev R100 > Set-Attention and invariant",
               scale_fractions=[.25, .5, 1.], scale_objective="T2 for both architectures, fixed full-T2-selected LR across fractions",
               bootstrap_seed=20260963, bootstrap_draws=10000, schedules=STEMS, field_encoding_reused=True,
               training_examples={k: v["examples"] for k, v in audit["datasets"].items()},
               environment=old["environment"], sources=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in sources])
    dump(CONFIG, cfg)
    data = Data(cfg)
    initial = []
    cached = np.load(ROOT / "phase5_mitigation/results/Set-Attention_s42/products.npy")
    for family in FAMILIES:
        model = model_for(data, family, 42)
        data.invariance(model, f"{family}_initial_s42")
        vectors, stats = data.generate_stats(model)
        maxdiff = float(np.linalg.norm(vectors - cached, axis=1).max())
        assert maxdiff <= 1e-5 and stats.delta_norm.max() == 0, "STOP: initialization is not the frozen base"
        for split in ["validation", "test"]:
            frame = data.frame(vectors, split)
            pq, m = metrics(frame)
            archived = pd.read_csv(ROOT / "phase5_mitigation/results/Set-Attention_s42/catalog_queries.csv")
            baseline = float(archived[archived.query_id.isin(cfg[split])]["Recall@100"].mean())
            assert abs(m["Recall@100"] - baseline) <= .0005, "STOP: initial retrieval drop"
            folder = HERE / "initialization" / family
            folder.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(folder / f"{split}_pairs.parquet", index=False)
            pq.to_csv(folder / f"{split}_queries.csv")
            initial.append(dict(family=family, seed=42, epoch=0, split=split, **m, max_L2_to_archived_base=maxdiff,
                                mean_cosine_to_base=float(stats.final_base_cosine.mean()), alpha=.05, mean_delta_norm=0., passed=True))
        print("INITIALIZATION", family, "test R100", round(100 * m["Recall@100"], 5), "max vector difference", maxdiff, flush=True)
    pd.DataFrame(initial).to_csv(HERE / "initialization.csv", index=False)
    dump(HERE / "initialization_passed.json", dict(passed=True, config_sha256=sha(CONFIG), utc=now(), test_use="identity guard only, no parameter/model selection"))


def better(a, b, tol=1e-12):
    return a["Recall@100"] > b["Recall@100"] + tol or (abs(a["Recall@100"] - b["Recall@100"]) <= tol and a["Robust@100"] > b["Robust@100"] + tol)


def train_candidate(data, family, regime, seed, lr):
    name = f"{family}_{regime}_s{seed}_lr{lr:g}"
    folder = HERE / "checkpoints" / name
    if (folder / "selected.json").exists():
        selected = json.loads((folder / "selected.json").read_text())
        assert selected["config_sha256"] == sha(CONFIG) and selected["weights_sha256"] == sha(folder / "weights.pt")
        return selected
    assert not folder.exists(), "Incomplete candidate retained; inspect before resume"
    folder.mkdir(parents=True)
    examples = json.loads((HERE / "data" / f"{regime}.json").read_text())
    assert {e["query_id"] for e in examples} <= set(data.config["train"])
    model = model_for(data, family, seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=.01)
    rng = random.Random(seed)
    step_logs, epochs = [], []
    best_metrics, best_epoch, stale = None, 0, 0
    wall = time.monotonic()
    try:
        for epoch in range(data.config["max_epochs"] + 1):
            loss_values, grad_values = [], []
            training_s = 0.
            if epoch:
                order = list(examples)
                rng.shuffle(order)
                model.train()
                p6.sync()
                start = time.monotonic()
                for offset in range(0, len(order), 16):
                    group = order[offset:offset + 16]
                    ids = sorted({data.pi[p] for e in group for p in [e["positive"]] + e["negatives"]})
                    lookup = {p: i for i, p in enumerate(ids)}
                    vectors = model(*data.batch(ids))
                    width = 1 + max(len(e["negatives"]) for e in group)
                    ix = np.zeros((len(group), width), dtype="i8")
                    mask = np.zeros((len(group), width), dtype=bool)
                    for i, e in enumerate(group):
                        row = [lookup[data.pi[p]] for p in [e["positive"]] + e["negatives"]]
                        ix[i, :len(row)] = row
                        mask[i, :len(row)] = True
                    qv = data.qt[[data.qi[e["query_id"]] for e in group]]
                    logits = (qv[:, None, :] * vectors[torch.tensor(ix, device=data.device)]).sum(-1) / .05
                    logits = logits.masked_fill(~torch.tensor(mask, device=data.device), -torch.inf)
                    loss = (torch.logsumexp(logits, dim=1) - logits[:, 0]).mean()
                    assert torch.isfinite(loss)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
                    opt.step()
                    loss_values.append((float(loss.item()), len(group)))
                    grad_values.append(float(norm))
                    step_logs.append(dict(epoch=epoch, step=offset // 16, loss=float(loss.item()), gradient_norm_before_clip=float(norm),
                                          alpha=float(model.beta.sigmoid()), positives=len(group), negatives=int(mask.sum() - len(group))))
                p6.sync()
                training_s = time.monotonic() - start
                cost(name, "training", start, epoch=epoch)
            start = time.monotonic()
            invariant = data.invariance(model, f"{name}_epoch{epoch}")
            cost(name, "sampled_invariance", start, epoch=epoch)
            start = time.monotonic()
            vectors, stats = data.generate_stats(model)
            cost(name, "dev_catalog_generation", start, epoch=epoch)
            start = time.monotonic()
            frame = data.frame(vectors, "validation")
            pq, current = metrics(frame)
            cost(name, "dev_ranking", start, epoch=epoch)
            if epoch == 0:
                assert stats.delta_norm.max() == 0 and stats.final_base_cosine.mean() > 1 - 1e-6
            record = dict(epoch=epoch, train_loss=sum(x * n for x, n in loss_values) / sum(n for _, n in loss_values) if epoch else None,
                          gradient_mean=float(np.mean(grad_values)) if epoch else 0., gradient_max=max(grad_values, default=0.),
                          training_seconds=training_s, **current, **{k: float(v) for k, v in stats.mean().items()}, invariance_max_L2=invariant["l2_max"])
            epochs.append(record)
            pd.DataFrame(epochs).to_csv(folder / "epochs.csv", index=False)
            pd.DataFrame(step_logs).to_csv(folder / "steps.csv", index=False)
            if best_metrics is None or better(current, best_metrics):
                best_metrics, best_epoch, stale = current, epoch, 0
                torch.save(model.state_dict(), folder / "weights.pt")
                frame.to_parquet(folder / "dev_pairs.parquet", index=False)
                pq.to_csv(folder / "dev_queries.csv")
                dump(folder / "invariance.json", invariant)
                dump(folder / "selected_diagnostics.json", {k: float(v) for k, v in stats.mean().items()})
            elif epoch:
                stale += 1
            print(name, "epoch", epoch, "dev R100", round(100 * current["Recall@100"], 4), "best", best_epoch,
                  "alpha", round(record["alpha"], 5), "cos", round(record["final_base_cosine"], 6), flush=True)
            if stale >= data.config["patience"]:
                break
        result = dict(name=name, family=family, architecture=FAMILIES[family], regime=regime, seed=seed, lr=lr,
                      best_epoch=best_epoch, epochs_run=len(epochs) - 1, examples=len(examples),
                      mean_negatives=float(np.mean([len(e["negatives"]) for e in examples])),
                      dev_recall100=best_metrics["Recall@100"], dev_robust100=best_metrics["Robust@100"],
                      parameters=sum(p.numel() for p in trainable), frozen_base_parameters=sum(p.numel() for p in model.base.parameters()),
                      config_sha256=sha(CONFIG), weights_sha256=sha(folder / "weights.pt"),
                      test_used_for_selection=False, completed_utc=now(), wall_seconds=time.monotonic() - wall)
        dump(folder / "selected.json", result)
        return result
    except BaseException as error:
        dump(folder / "interrupted.json", dict(error=repr(error), utc=now()))
        raise


def choose(candidates):
    return sorted(candidates, key=lambda r: (-round(r["dev_recall100"], 12), -round(r["dev_robust100"], 12),
                                            r["best_epoch"], r["examples"], r["lr"]))[0]


def train_all():
    cfg = validate()
    guard = json.loads((HERE / "initialization_passed.json").read_text())
    assert guard["passed"] and guard["config_sha256"] == sha(CONFIG)
    if (HERE / "selection.json").exists():
        print("Selections already locked", flush=True)
        return
    data = Data(cfg)
    candidates, selected, family_winners = [], {}, {}
    for family in FAMILIES:
        for regime in cfg["regimes"]:
            trials = [train_candidate(data, family, regime, 42, lr) for lr in cfg["learning_rates"]]
            candidates.extend(trials)
            selected[f"{family}_{regime}_s42"] = choose(trials)
        winner = choose([selected[f"{family}_{regime}_s42"] for regime in cfg["regimes"]])
        family_winners[family] = next(k for k, value in selected.items() if value["name"] == winner["name"])
        dump(HERE / "primary_selection_progress.json", dict(selected=selected, family_winners=family_winners, test_used=False))
    # Use the full-T2 LR at every scale; fractions never replace a primary winner.
    for family in FAMILIES:
        lr = selected[f"{family}_T2_s42"]["lr"]
        for fraction in [.25, .5]:
            regime = f"T2_f{fraction:g}"
            result = train_candidate(data, family, regime, 42, lr)
            candidates.append(result)
            selected[f"{family}_{regime}_s42"] = result
    seed_gates = {}
    for family, key in family_winners.items():
        winner = selected[key]
        passed = winner["dev_recall100"] >= cfg["dev_baselines"]["Canonical"]
        seed_gates[family] = dict(primary_key=key, passed=passed, threshold=cfg["dev_baselines"]["Canonical"], additional_seeds=[])
        if passed:
            for seed in cfg["confirmation_seeds"]:
                result = train_candidate(data, family, winner["regime"], seed, winner["lr"])
                candidates.append(result)
                selected[f"{family}_{winner['regime']}_s{seed}"] = result
                seed_gates[family]["additional_seeds"].append(seed)
        dump(HERE / "seed_gate_progress.json", seed_gates)
    primary_and_seeds = list(family_winners.values()) + [k for k, r in selected.items() if r["seed"] != 42]
    target_keys = [k for k in primary_and_seeds if selected[k]["best_epoch"] > 0 and
                   selected[k]["dev_recall100"] > cfg["dev_baselines"]["Set-Attention_s42"] + 1e-12]
    lock = dict(locked_utc=now(), config_sha256=sha(CONFIG), candidates=candidates, selected=selected,
                family_winners=family_winners, seed_gates=seed_gates, target_keys=target_keys,
                initialization_test_identity_guard_only=True, test_used_for_selection=False)
    dump(HERE / "selection.json", lock)
    pd.DataFrame(candidates).to_csv(HERE / "candidates.csv", index=False)
    print("SELECTION LOCKED", family_winners, "targets", target_keys, flush=True)


def evaluate_all():
    cfg = validate()
    lock = json.loads((HERE / "selection.json").read_text())
    assert lock["config_sha256"] == sha(CONFIG) and not lock["test_used_for_selection"]
    data = Data(cfg)
    invariance = dict(selection_sha256=sha(HERE / "selection.json"), models={})
    for key, selected in lock["selected"].items():
        folder = OUT / key
        if (folder / "evaluation.json").exists():
            meta = json.loads((folder / "evaluation.json").read_text())
            assert meta["selection_sha256"] == sha(HERE / "selection.json")
            invariance["models"][key] = json.loads((folder / "invariance.json").read_text())
            continue
        folder.mkdir(parents=True, exist_ok=True)
        model = model_for(data, selected["family"], selected["seed"])
        weights = HERE / "checkpoints" / selected["name"] / "weights.pt"
        assert sha(weights) == selected["weights_sha256"]
        model.load_state_dict(torch.load(weights, map_location=data.device, weights_only=True))
        inv = data.invariance(model, key + "_selected")
        start = time.monotonic()
        vectors, stats = data.generate_stats(model)
        cost(key, "final_catalog_generation", start)
        np.save(folder / "products.npy", vectors)
        stats.insert(0, "product_id", [str(p["product_id"]) for p in data.products])
        stats.to_parquet(folder / "residual_diagnostics.parquet", index=False)
        dump(folder / "residual_summary.json", {col: dict(mean=float(stats[col].mean()), min=float(stats[col].min()),
                    p05=float(stats[col].quantile(.05)), median=float(stats[col].median()), p95=float(stats[col].quantile(.95)),
                    max=float(stats[col].max())) for col in STATS})
        start = time.monotonic()
        frame = data.frame(vectors, "test")
        frame.to_parquet(folder / "catalog_pairs.parquet", index=False)
        pq, m = metrics(frame)
        pq.to_csv(folder / "catalog_queries.csv")
        cost(key, "test_ranking", start)
        regenerated = frame[["query_id", "product_id", "C0"]].copy()
        is_target = key in lock["target_keys"]
        target = regenerated.copy() if is_target else None
        qids = sorted(frame.query_id.unique())
        qmap = {q: i for i, q in enumerate(qids)}
        qv = data.queries_v[[data.qi[q] for q in qids]]
        base_scores = qv @ vectors.T if is_target else None
        audit = []
        for schedule in STEMS[1:]:
            start = time.monotonic()
            other = data.generate(model, schedule)
            diff = np.linalg.norm(vectors - other, axis=1)
            assert diff.max() <= 1e-5, "STOP: full catalog vector invariance failed before schedule ranking"
            g = data.frame(other, "test")
            regenerated[schedule] = g.C0
            audit.append(dict(schedule=schedule, max_L2=float(diff.max()), mean_L2=float(diff.mean()),
                              changed_ranks=int((g.C0 != frame.C0).sum()),
                              changed_inclusion20=int(((g.C0 <= 20) != (frame.C0 <= 20)).sum()),
                              changed_inclusion100=int(((g.C0 <= 100) != (frame.C0 <= 100)).sum())))
            if is_target:
                scores = qv @ other.T
                result = np.empty(len(frame), dtype="i4")
                for qid, group in frame.groupby("query_id", sort=False):
                    ix = np.array([data.pi[p] for p in group.product_id])
                    result[group.index] = prior.rank_target(base_scores[qmap[qid]], ix, scores[qmap[qid], ix])
                target[schedule] = result
            cost(key, "full_schedule_audit", start, schedule=schedule)
        regenerated.to_parquet(folder / "regenerated_catalog_pairs.parquet", index=False)
        rpq, rm = metrics(regenerated)
        rpq.to_csv(folder / "regenerated_catalog_queries.csv")
        inv["full_catalog_audit"] = audit
        inv["regenerated_metrics"] = rm
        if is_target:
            for qid, group in frame.groupby("query_id", sort=False):
                ix = np.array([data.pi[p] for p in group.product_id])
                assert np.array_equal(prior.rank_target(base_scores[qmap[qid]], ix, base_scores[qmap[qid], ix]), group.C0)
            frame.to_parquet(folder / "target_pairs.parquet", index=False)
            target.to_parquet(folder / "regenerated_target_pairs.parquet", index=False)
            inv["regenerated_target_metrics"] = metrics(target)[1]
        dump(folder / "invariance.json", inv)
        invariance["models"][key] = inv
        dump(folder / "evaluation.json", dict(completed_utc=now(), selection_sha256=sha(HERE / "selection.json"),
                    weights_sha256=sha(weights), products_sha256=sha(folder / "products.npy"), target_only=is_target,
                    exact_pairs=len(frame), queries=len(pq), test_metrics=m))
        dump(HERE / "PHASE6B_INVARIANCE.json", invariance)
        print("TEST", key, "R100", round(100 * m["Recall@100"], 4), "regenerated VI100", rm["VI@100"], flush=True)
    dump(HERE / "PHASE6B_INVARIANCE.json", invariance)
    dump(HERE / "evaluation_complete.json", dict(completed_utc=now(), selection_sha256=sha(HERE / "selection.json"), models=list(lock["selected"])))


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "train", "evaluate"])
    {"prepare": prepare, "train": train_all, "evaluate": evaluate_all}[parser.parse_args().action]()
