"""Audit the user-specified existing case; tokenize seven inputs without inference.

Reconstruct this one query's target-only ranks from saved embeddings/scores by
explicitly excluding the old target. No retrieval experiment or new ordering.
"""
import gzip
import os
import sys
from collections import Counter, defaultdict, deque
import numpy as np
import pandas as pd
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from threadpoolctl import threadpool_limits
from io_common import ROOT, BASE, DATA, STEMS, dump, load, source, sha

sys.path.insert(0, str(ROOT / 'phase2'))
from src.representations import build, _plain, _random_order
threadpool_limits(4)

def records(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield __import__('json').loads(line)

def one_pair(path):
    f = pd.read_parquet(path)
    g = f[(f.query_id.astype(int) == 409) & (f.product_id.astype(str) == '24318')]
    assert len(g) == 1
    return g.iloc[0]

def main():
    cfg_path = ROOT / 'phase2/config/phase2.json'
    split_path = ROOT / 'phase4/config/splits.json'
    cfg, splits = load(cfg_path), load(split_path)
    spec = next(x for x in cfg['models'] if x['key'] == 'gte_modernbert')
    assert spec['max_tokens'] == 8192 and cfg['encoding']['query_prefix'] == ''
    assert 409 in splits['wands_heldout'] and 409 not in splits['wands_dev']
    qp = ROOT / 'phase2/data/processed/wands_queries.csv'
    queries = pd.read_csv(qp)
    query_row = queries[queries.query_id.astype(int).eq(409)]
    assert len(query_row) == 1 and query_row.iloc[0]['query'] == 'teal chair'
    rp = ROOT / 'phase3/results/target_only_permutations/wands_gte_modernbert_pairs.parquet'
    ranks = one_pair(rp)
    assert ranks.label == 'Exact' and int(ranks.C0) == 11 and int(ranks.C2s5) == 939
    cp = ROOT / 'phase3/results/target_only_permutations/candidates.csv'
    candidates = pd.read_csv(cp)
    c = candidates[(candidates.dataset == 'wands') & (candidates.retriever == 'gte_modernbert') &
                   (candidates.query_id == 409) & (candidates.product_id.astype(str) == '24318')]
    assert len(c) == 1 and c.iloc[0]['query'] == 'teal chair' and c.iloc[0]['label'] == 'Exact'
    assert all(int(c.iloc[0][s]) == int(ranks[s]) for s in STEMS)
    pp = ROOT / 'phase2/data/processed/wands_products.jsonl.gz'
    product = next(p for p in records(pp) if str(p['product_id']) == '24318')
    attrs = product['attributes']
    tok = AutoTokenizer.from_pretrained(spec['name'], revision=spec['revision'], local_files_only=True)
    # This tokenizer uses Hugging Face's unset-length sentinel, not a finite cap.
    # Native encoding passes max_length explicitly; verify the pinned model cap.
    model_config_path = hf_hub_download(spec['name'], 'config.json', revision=spec['revision'], local_files_only=True)
    model_config = load(model_config_path)
    assert model_config['max_position_embeddings'] == spec['max_tokens'] == 8192
    encoder_source = (ROOT / 'phase2/src/encoding.py').read_text(encoding='utf-8')
    assert 'max_length=self.spec["max_tokens"]' in encoder_source
    dump(DATA / 'gte_pinned_model_config.json', model_config)
    prior_lengths = pd.read_csv(BASE / 'data/all_seven_token_lengths.csv', dtype={'product_id': str})
    prior = prior_lengths[(prior_lengths.dataset == 'wands') & (prior_lengths.model == 'gte_modernbert') & (prior_lengths.product_id.astype(str) == '24318')]
    assert len(prior) == 1
    source_paths = [cfg_path, split_path, qp, rp, cp, pp, BASE / 'data/all_seven_token_lengths.csv', ROOT / 'phase2/src/representations.py', ROOT / 'phase2/src/encoding.py']
    # The encoder's saved inputs have no product prefix; _plain builds fixed blocks.
    inputs, token_rows = {}, []
    catalog = None
    for s, condition in zip(STEMS, cfg['primary_variant_family']):
        path = ROOT / f'phase2/data/representations/wands/{s}.jsonl.gz'
        source_paths.append(path)
        if s == 'C0':
            recs = list(records(path))
            catalog = [str(r['product_id']) for r in recs]
            text = next(r['text'] for r in recs if str(r['product_id']) == '24318')
        else:
            text = next(r['text'] for r in records(path) if str(r['product_id']) == '24318')
        ordered = list(attrs) if s == 'C0' else list(reversed(attrs)) if s == 'C1' else _random_order(attrs, '24318', cfg['attribute_permutation_seeds'][int(s[-1]) - 1])
        assert Counter(ordered) == Counter(attrs)
        assert text == build(product, condition, cfg['attribute_permutation_seeds']) == _plain(product, ordered)
        # Track repeated entry occurrences individually, preserving multiplicity.
        available = defaultdict(deque)
        for i, entry in enumerate(attrs, 1):
            available[entry].append(i)
        entry_ids = [available[entry].popleft() for entry in ordered]
        n = len(tok(text, add_special_tokens=True, truncation=False)['input_ids'])
        without = len(tok(text, add_special_tokens=False, truncation=False)['input_ids'])
        assert n == int(prior.iloc[0][f'tokens_{s}']) == 1059
        assert n <= spec['max_tokens'] and n - without == tok.num_special_tokens_to_add(pair=False) == 2
        inputs[s] = dict(condition=condition, text=text, attribute_entries=ordered,
                         original_entry_ids=entry_ids, tokens_including_special=n,
                         target_only_rank=int(ranks[s]), catalog_wide_rank_not_used=int(ranks[f'whole_{s}']))
        token_rows.append(dict(schedule=s, tokens_without_special=without, special_tokens=n - without,
                               tokens_including_special=n, encoder_limit=8192, fully_retained=True))
    # Independent old-entry exclusion audit, one query only, using saved vectors.
    provp = ROOT / 'phase3/results/target_only_permutations/provenance.json'
    prov = next(x for x in load(provp)['sources'] if x['dataset'] == 'wands' and x['model'] == 'gte_modernbert')
    qfile, pfile = (ROOT / prov[key] for key in ['query_embedding', 'product_embedding'])
    source_paths.extend([provp, qfile, pfile, ROOT / 'phase3/scripts/analyze_target_only_permutations.py'])
    qvectors, pvectors = np.load(qfile, mmap_mode='r'), np.load(pfile, mmap_mode='r')
    qi = queries.query_id.astype(int).tolist().index(409)
    pi = catalog.index('24318')
    base = np.asarray(qvectors[qi:qi+1], dtype='f4') @ np.asarray(pvectors, dtype='f4').T
    base = base[0]
    indices = np.arange(len(catalog))
    competitors = indices != pi
    assert len(catalog) == 42994 and competitors.sum() == 42993
    rank_checks = []
    for s in STEMS:
        path = ROOT / f'phase2/results/phase2_pair_ranks/wands_gte_modernbert_native_{s}.parquet'
        source_paths.append(path)
        raw = one_pair(path)
        assert raw.label == 'Exact'
        score = base[pi] if s == 'C0' else np.float32(raw.score)
        rank = 1 + np.count_nonzero(competitors & ((base > score) | ((base == score) & (indices < pi))))
        assert rank == int(ranks[s]), (s, int(rank), int(ranks[s]))
        assert int(raw['rank']) == int(ranks[f'whole_{s}'])
        rank_checks.append(dict(schedule=s, reconstructed_target_only_rank=int(rank), saved_target_only_rank=int(ranks[s]),
                                catalog_wide_rank_not_used=int(raw['rank']), alternative_score=float(score)))
    pd.DataFrame(token_rows).to_csv(DATA / 'figure1_case_token_counts.csv', index=False)
    pd.DataFrame(rank_checks).to_csv(DATA / 'figure1_case_rank_checks.csv', index=False)
    excerpt_ids = [1, 2, 3, 4]  # First four source-order entries, not outcome-selected fields.
    excerpt = {s: [dict(original_entry_id=i, position=pos, text=attrs[i-1])
                   for pos, i in enumerate(inputs[s]['original_entry_ids'], 1) if i in excerpt_ids] for s in ['C0', 'C2s5']}
    dump(DATA / 'figure1_case_inputs.json', dict(dataset='wands', query_id=409, query='teal chair', product_id='24318',
          label='Exact', model=spec, product=product, schedules=inputs, display_excerpt=excerpt,
          excerpt_selection='First four C0 attribute entries; displayed in each schedule\'s relative order.'))
    dump(DATA / 'figure1_case_audit.json', dict(status='PASS', case_preselected_by_user=True, post_hoc_illustrative=True,
          dataset='wands', query_id=409, query='teal chair', product_id='24318', model=spec,
          heldout=True, highest_relevance_label='Exact', relabelled=False,
          attribute_entry_count=len(attrs), unique_attribute_strings=len(Counter(attrs)),
          repeated_strings={a: n for a, n in Counter(attrs).items() if n > 1},
          all_seven_multisets_and_multiplicities_preserved=True, all_fixed_blocks_and_separators_preserved=True,
          tokenizer_model_max_length=int(tok.model_max_length),
          tokenizer_length_note='The tokenizer value is the unset-length sentinel, not the operational cap. DenseEncoder._forward_native explicitly passes max_length=8192; pinned model max_position_embeddings is 8192.',
          pinned_model_max_position_embeddings=int(model_config['max_position_embeddings']),
          pinned_model_config_sha256=sha(model_config_path), effective_native_token_limit=8192,
          pinned_tokenizer_files=[{'name': p.name, 'sha256': sha(p)} for p in __import__('pathlib').Path(model_config_path).parent.glob('*') if p.name in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']],
          empty_product_prefix=True,
          tokens=token_rows, ranks=rank_checks, old_target_excluded=True, competitors_fixed_raw_C0=42993,
          catalog_size_after_replacement=42994, tie_rule='Descending fp32 score, then original catalog index.',
          audit_method='One query scored against saved C0 embeddings, old target explicitly excluded. Alternate target scores read from existing rank files. No encoding, training, or new schedules.',
          sources=[source(p) for p in source_paths if p.exists()]))
    print('PASS: seven complete inputs, 1,059 tokens including 2 special tokens; ranks 11 and 939; raw C0 competitors fixed.')
    print('Attribute entries:', len(attrs), 'Excerpt:', excerpt)

if __name__ == '__main__':
    main()
