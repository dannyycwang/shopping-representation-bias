from common import *
from datetime import datetime, timezone
import subprocess

path = HERE/'PROSPECTIVE_PROTOCOL.json'
assert not path.exists(), 'Protocol already frozen; never overwrite it.'
payload = dict(
    frozen_utc=datetime.now(timezone.utc).isoformat(),
    framing_pdf=r'C:/Users/ycw/Downloads/Same_Product__Different_Visibility__Representation_Robustness_in_Neural_E_Commerce_Retrieval (4).pdf',
    task_source=r'C:/Users/ycw/.codex/attachments/1f086614-4aea-4a1f-85e2-124f38e62c2f/pasted-text.txt',
    datasets=['wands','esci'], models=CFG['models'], encoder_profile='native',
    canonical_rules={
      'lexical_ascending': 'Ascending (complete entry.casefold(), complete original entry).',
      'lexical_descending': 'Descending (complete entry.casefold(), complete original entry).',
      'field_priority_type': 'Historical rule_type, fixed without evaluating held-out outcomes: field is text before first colon, else first 40 characters, lowercased. Entries with producttype/type/style as a substring of field come first; intrinsic (entry.casefold(), entry) resolves every non-identical tie.'},
    duplicates='Keep every occurrence. Equal complete strings are indistinguishable; their incoming order cannot affect serialized text.',
    preservation='Only complete attribute entries move; fixed text, entry labels, separators and section placement are unchanged.',
    reuse='Reuse inherited ranks only for exactly matching rule, model, query embeddings, profile and catalog. Reuse canonical_raw for ascending and rule_type for field priority when exact. M2 remains separate.',
    new_inference='Frozen pretrained weights only; no training. fp16 model, fp32 native pooling and normalization; same historical query embeddings. Stable length-bucket batching from current frozen phase2 encoding.py; preserve its implementation metadata. GTE uses SDPA. Exact fp32 dot-product ranks, descending score then ascending original catalog index.',
    ranking_support='Full original catalogs and queries; summarize Hq = Exact (WANDS) or E (ESCI). WANDS primary uses existing 384 held-out query IDs; ESCI uses existing 500-query evaluation sample, no new held-out split is invented.',
    comparisons='All six non-C0 raw schedules vs C0 at K=20,100; all 21 pairs supplementary. Three canonical rules vs C0 and mean-of-seven raw schedule Recall. Never choose a test-set winner.',
    bootstrap=dict(draws=DRAWS,seed=SEED,unit='query cluster',paired='Same draws retain all Hq products and all schedule contrasts; same sorted eligible query IDs across encoders and K.',interval='95% percentile; descriptive, uncorrected; no equivalence margin.'),
    fitting='Count every original serialized input separately with encoder-specific pinned tokenizer, truncation=False, add_special_tokens=True, actual empty product prefix. Filter evaluated targets only; never filter competitors.',
    family_expansion='Optional 16/32 expansion omitted to prioritize requested membership, fitting and canonical controls. No new schedule selected.',
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'phase2/config/phase2.json',ROOT/'phase4/config/splits.json',ROOT/'phase2/src/encoding.py',ROOT/'phase2/src/representations.py',Path(__file__),HERE/'scripts/common.py']},
    git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
for key in ['framing_pdf','task_source']: payload[key+'_sha256']=sha(payload[key])
dump(path,payload)
(HERE/'PROSPECTIVE_PROTOCOL.sha256').write_text(sha(path)+'\n')
(HERE/'initial_git_status.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),encoding='utf8')
print('FROZEN',path,sha(path),flush=True)
