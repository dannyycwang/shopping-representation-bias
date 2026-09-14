"""Read-only reuse of frozen WANDS/BGE caches, serializers, ranking, and metric definitions."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'phase6'))
import run as pilot
from value_models import field_key, ValueAttention

prior = pilot.prior
STEMS = pilot.STEMS
dump, sha, now = pilot.dump, pilot.sha, pilot.now
CONFIG = HERE / 'PHASE6_CONFIG.json'
METHODS = {'C1': 'Contextual DeepSets', 'C2': 'Set Transformer', 'C3': 'Set Transformer + Field-ID', 'C4': 'Circular Attention'}
GAINS = {'Exact': 3, 'Partial': 1, 'Irrelevant': 0}
RELATIONS = [('Exact', 'Irrelevant', 1.), ('Exact', 'Partial', .5), ('Partial', 'Irrelevant', .5)]


def validate():
    cfg = json.loads(CONFIG.read_text(encoding='utf8'))
    for e in cfg['sources'] + json.loads((HERE / 'historical_sources.json').read_text(encoding='utf8')):
        assert sha(ROOT / e['path']) == e['sha256'], e['path']
    return cfg


def cost(name, stage, start, **extra):
    with (HERE / 'cost.jsonl').open('a', encoding='utf8') as out:
        out.write(json.dumps(dict(name=name, stage=stage, seconds=time.monotonic()-start, **extra))+'\n')


def graded_queries(frame):
    """Condensed graded NDCG on all judged products, independently of Exact support."""
    records = []
    for qid, g in frame.groupby('query_id', sort=True):
        gains = g.label.map(GAINS).to_numpy(dtype=float)
        n = min(10, len(g))
        discount = 1 / np.log2(np.arange(2, n+2))
        ideal = np.sort(gains)[::-1][:n] @ discount
        row = dict(query_id=int(qid), judged_pairs=len(g), positive_idcg=bool(ideal > 0))
        for s in STEMS:
            order = np.argsort(g[s].to_numpy(), kind='stable')
            row[s+'_cNDCG@10'] = float(gains[order][:n] @ discount / ideal) if ideal else np.nan
        row['cNDCG@10'] = float(np.mean([row[s+'_cNDCG@10'] for s in STEMS]))
        records.append(row)
    return pd.DataFrame(records).set_index('query_id')


def exact_queries(frame):
    return prior.metric_frame(frame[frame.label.eq('Exact')] if 'label' in frame else frame)[1]


def measures(frame):
    exact, graded = exact_queries(frame), graded_queries(frame)
    m = {k: float(exact[k].mean()) for k in ['Recall@20','Recall@100','VI@20','VI@100','Robust@100','Never@100','worst_schedule@100']}
    m['cNDCG@10'] = float(graded['cNDCG@10'].mean())
    return m, exact, graded


class Data(pilot.Data):
    def __init__(self, config=None):
        super().__init__(config)
        train = self.judgments[self.judgments.query_id.isin(self.config['train'])]
        products = set(train.product_id.astype(str))
        keys = sorted({field_key(a) for p in products for a in self.products[self.pi[p]]['attributes'] if field_key(a) is not None})
        self.vocab = {k: i+1 for i, k in enumerate(keys)}
        self.atom_fields = np.array([self.vocab.get(field_key(a), 0) for a in self.unique], dtype='i8')
        self.ft = torch.tensor(self.atom_fields, device=self.device)
        self.judgments.product_id = self.judgments.product_id.astype(str)
        self.groups = {int(q): {label: sorted(g.loc[g.label.eq(label),'product_id'].tolist()) for label in GAINS}
                       for q, g in train.assign(product_id=train.product_id.astype(str)).groupby('query_id')}
        self.eligible = sorted(q for q, g in self.groups.items() if any(g[a] and g[b] for a,b,_ in RELATIONS))
        self.features = pd.read_csv(ROOT/'phase4/results/wands_product_features.csv', dtype={'product_id': str}).set_index('product_id')
        self.fitids = set(self.features.index[self.features.fully_fits])
        report_keys = sorted({field_key(a) or '<UNK-native>' for a in self.unique})
        self.report_keys = report_keys
        self.report_key_index = {k:i for i,k in enumerate(report_keys)}
        self.report_atom_fields = np.array([self.report_key_index[field_key(a) or '<UNK-native>'] for a in self.unique])

    def judged_frame(self, vectors, split):
        f = self.judgments[self.judgments.query_id.isin(self.config[split])][['query_id','product_id','label']].copy().reset_index(drop=True)
        qids = sorted(f.query_id.unique())
        scores = self.queries_v[[self.qi[q] for q in qids]] @ vectors.T
        rr = prior.ranks(scores)[1]
        qi = {q:i for i,q in enumerate(qids)}
        values = rr[[qi[q] for q in f.query_id], [self.pi[p] for p in f.product_id]]
        for s in STEMS:
            f[s] = values
        return f

    def invariance(self, model, name):
        model.eval()
        values, traces = [], {}
        with torch.inference_mode():
            for s, condition in zip(STEMS, prior.CFG['primary_variant_family']):
                ordered = [self.atoms(i,s) for i in self.sample]
                for i, atoms in zip(self.sample,ordered):
                    assert Counter(atoms) == Counter(self.products[i]['attributes'])
                    assert prior._plain(self.products[i],atoms) == prior.build(self.products[i],condition,prior.CFG['attribute_permutation_seeds'])
                traces[s] = [hashlib.sha256(json.dumps(a,ensure_ascii=False).encode()).hexdigest() for a in ordered]
                values.append(model(*self.batch(self.sample,s)).cpu().numpy())
        v = np.stack(values)
        d = np.linalg.norm(v-v[:1],axis=2)
        vd = v.astype('f8')
        c = np.abs(1 - (vd*vd[:1]).sum(2)/(np.linalg.norm(vd,axis=2)*np.linalg.norm(vd[:1],axis=2)))
        r = dict(name=name, products=len(self.sample), schedules=STEMS, max_L2=float(d.max()), mean_L2=float(d.mean()),
                 max_cosine_difference=float(c.max()), mean_cosine_difference=float(c.mean()),
                 changed_input_products=sum(len({traces[s][i] for s in STEMS})>1 for i in range(len(self.sample))),
                 schedule_atom_hashes=traces, actual_forward_passes=True, passed=bool(np.isfinite(v).all() and d.max()<=1e-5))
        dump(HERE/'invariance'/f'{name}.json',r)
        assert r['passed'] and r['changed_input_products']>=100
        return r

    def generate(self, model, schedule='C0', diagnostics=False):
        model.eval()
        result = np.empty_like(self.non)
        rows, field_sum, field_count = [], np.zeros(len(self.report_keys)), np.zeros(len(self.report_keys),dtype='i8')
        with torch.inference_mode():
            for offset in range(0,len(self.products),128):
                ids = self.infer_order[offset:offset+128].tolist()
                batch = self.batch(ids,schedule)
                final,w,attr = model.components(*batch)
                result[ids] = final.cpu().numpy()
                if diagnostics:
                    wa = w.cpu().numpy().astype('f8')
                    for i,index in enumerate(ids):
                        n = self.lengths[index]
                        weights = wa[i,:n]
                        ent = float(-(weights*np.log(weights.clip(1e-30))).sum())
                        rows.append(dict(product_id=str(self.products[index]['product_id']), fields=int(n), entropy=ent,
                                         normalized_entropy=ent/np.log(n) if n>1 else 1., max_weight=float(weights.max()) if n else 0.,
                                         effective_fields=float(np.exp(ent)) if n else 0., weight_sum=float(weights.sum())))
                        a = self.indices[index] if schedule=='C0' else [self.lookup[z] for z in self.atoms(index,schedule)]
                        native = self.report_atom_fields[a]
                        field_sum += np.bincount(native,weights=weights,minlength=len(field_sum))
                        field_count += np.bincount(native,minlength=len(field_count))
        assert np.isfinite(result).all() and abs(np.linalg.norm(result,axis=1)-1).max()<1e-5
        if diagnostics:
            fields = pd.DataFrame({'field_key':self.report_keys,'occurrences':field_count,'weight_sum':field_sum,
                                   'mean_weight':field_sum/np.maximum(field_count,1)})
            return result,pd.DataFrame(rows),fields
        return result


def model_for(data, method, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return ValueAttention(method,len(data.vocab)+1).to(data.device)
