"""Freeze new-model follow-up before held-out evaluation; never overwrite protocol."""
from common import *
import importlib.metadata

def main():
    draft='--draft' in sys.argv
    if (HERE/'PROTOCOL.json').exists():
        protocol();print('Existing protocol verified; unchanged.');return
    pre=read(HERE/'audit/preflight.json');models={}
    for key,name,length,dim,pool,pad,prompts in [
        ('qwen3_4b','Qwen/Qwen3-Embedding-4B',32768,2560,'last nonpadding token','left',{'query':'query','document':'document'}),
        ('embeddinggemma2','google/embeddinggemma-2',8192,768,'attention-mask mean including prompt after native projection','right',{'query':'SearchQuery','document':'Document'})]:
        folder=HERE/'audit/models'/name.split('/')[-1];api=read(folder/'api.json');cfg=read(folder/'config.json');st=read(folder/'config_sentence_transformers.json')
        models[key]=dict(key=key,name=name,revision=api['sha'],tokenizer_revision=api['sha'],native_supported_length=length,
            config_max_position_embeddings=cfg.get('max_position_embeddings',cfg.get('text_config',{}).get('max_position_embeddings')),
            max_length=length,dimension=dim,pooling=pool,projection='512->768 native per-token' if key=='embeddinggemma2' else 'none',
            dtype='bfloat16',pooling_dtype='bfloat16 official operation order',normalization='official native L2 followed by FP32 unit normalization',
            scoring='numpy FP32 matmul; 4 BLAS threads; full query matrix per schedule',attention_backend='sdpa',padding_side=pad,
            device='cuda',prompt_names=prompts,prefixes={r:st['prompts'][n] for r,n in prompts.items()},
            document_wrapper='Document default wrapper; full inherited text retained as content; no separate title' if key=='embeddinggemma2' else 'none',
            execution_seed=2026101601,batching={'base':1 if key=='qwen3_4b' else 16,'order':'stable ascending effective token length then input content hash',
            'limits':[[512,16],[1024,8],[2048,4],[4096,2],[length,1]] if key=='embeddinggemma2' else [[length,1]],
            'oom_policy':'Do not reuse partial batch. Log event, stop profile; batch=1 retry uses a separately fingerprinted deviation, never alter length silently.'},
            use_cache=False,text_only=True,quantization=False,truncate_dim=None)
    files=pre['source_files']+[source(p) for p in (HERE/'audit/models').glob('*/*') if p.is_file()]
    files += [source(p) for p in (HERE/'scripts').glob('*.py')]
    files += [source(p) for p in (HERE/'tests').glob('*.py')]
    files += [source(HERE/'audit/attachments.json'),source(HERE/'audit/implementation_sources.json')]
    versions={n:importlib.metadata.version(n) for n in ['torch','transformers','sentence-transformers','huggingface-hub','numpy','pandas','tokenizers','pyarrow']}
    x=dict(protocol_kind='new-model follow-up after historical outcomes; NOT whole-study preregistration',frozen_utc=now(),git_commit=pre['git_commit'],
        models=models,optional_harrier_enabled=False,source_files=files,populations=pre['populations'],software=versions,hardware=pre['hardware'],
        schedules=STEMS,permutation_seeds=CFG['attribute_permutation_seeds'],K=[20,100],canonical_rules=RULES,
        gains=GAINS,highest_labels={'wands':'Exact','esci':'E'},split='existing phase4 splits; no resampling',
        ranking='Descending exact FP32 score, ascending original catalog index. Target-only removes old C0 entry before insertion.',
        bootstrap={'seed':SEED,'draws':DRAWS,'unit':'query cluster','ci':'unadjusted 95% percentile','pairing':'identical draws for sorted query IDs on identical support'},
        analysis={'primary':'full highest-label support, query-macro PI/VI/PO and catalog-wide metrics',
            'secondary':'all-seven fitting target pairs; all competitors retained; no causal truncation claim',
            'primary_contrast':'C1 minus C0','supplementary_contrasts':[s+' minus C0' for s in STEMS[2:]],
            'membership':'integer highest-label gained/lost counts; full Top-K membership stored separately',
            'graded':'direct gains; unjudged zero only for standard DCG; IDCG from all judged labels; condensed removes unjudged before discount',
            'canonical':'only after both primary models RQ1/RQ2 complete; paired against within-query seven-raw mean; all six transitions',
            'cross_model':'descriptive full support; model-specific fitting not same population; optional common-pair intersection'},
        technical_tests={'selection':'WANDS development targets by SHA256(recent-model-tech|product_id); shortest/longest within first 32; query first 3 by ID hash',
            'repeat_count':3,'reference':'official SentenceTransformer.encode; shared frozen backbone, independently implemented pooling',
            'atol':1e-3,'rtol':1e-3,'min_cosine':0.9999,'norm_atol':2e-6,
            'tolerance_basis':'BF16 native reference component rounding; strict ID equality and output comparison; NOT ranking epsilon',
            'ranking_epsilon':0,'holdout_used_for_selection':False},
        boundary_audit={'selection':'all crossing witnesses with abs(margin) <= measured technical repeat score range, all exact ties, plus first 8 crossing pairs by salted ID hash per dataset/intervention/K',
            'repeat_count':3,'profile':'rerun original batches containing selected queries, targets, and Kth competitors with all original batch companions; score the full catalog using the identical FP32 matrix shape and frozen profile; other vectors retain their compatible original profile',
            'preserve_original':True,'label':'numerical_ambiguity if any cutoff decision differs; no epsilon-based rank correction'},
        allowed='No fine-tuning, reranking, expansion, prompt search, images, paid compute, model substitution, dimension slicing or quantization')
    if draft:
        dump(HERE/'audit/protocol_candidate.json',x);print('Technical candidate written; held-out encoding forbidden.');return
    dump(HERE/'PROTOCOL.json',x);(HERE/'PROTOCOL.sha256').write_text(sha(HERE/'PROTOCOL.json')+'\n')
    status('freeze',protocol_sha256=sha(HERE/'PROTOCOL.json'))
    print('FROZEN',sha(HERE/'PROTOCOL.json'),flush=True)

if __name__=='__main__':main()
