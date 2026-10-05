"""Frozen Figure 1 teal-chair case. Official Captum only; immutable attempts."""
from pathlib import Path
import argparse, ast, csv, datetime, gzip, hashlib, importlib.metadata, importlib.util
import inspect, json, math, os, platform, shutil, subprocess, sys, time, traceback
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
threadpool_limits(4)
HERE = Path(__file__).resolve().parent
STEMS = ['C0','C1','C2s1','C2s2','C2s3','C2s4','C2s5']
ENDS = ['C0','C2s5']

def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def dump(p, v):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def arrhash(a):
    a=np.asarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def writecsv(p, rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(p,index=False,float_format='%.17g',encoding='utf8',lineterminator='\n')
def loadmod(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def pick(df): return df[(df.query_id.astype(int)==409)&(df.product_id.astype(str)=='24318')].iloc[0]
def hash_sources(out,paths):
    result={str(p):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(set(map(Path,paths))) if p.is_file()}
    dump(out/'environment/source_hashes.json',result)
    return result

def serialize(rep,p,s):
    ids=list(range(len(p['attributes'])))
    if s=='C1': ids.reverse()
    elif s.startswith('C2s'): ids=rep._random_order(ids,str(p['product_id']),20260910+int(s[-1]))
    text='';spans=[];positions={}
    def add(value,entry):
        nonlocal text
        spans.append({'start':len(text),'end':len(text)+len(value),'entry_id':entry,'text':value});text+=value
    for field,value in rep._blocks(p,[p['attributes'][i] for i in ids]):
        if text: add('\n','section_separators')
        if field=='attributes':
            for j,i in enumerate(ids):
                if j:add(p['attribute_separator'],'attribute_separators')
                add(p['attributes'][i],f'attr_{i:03d}');positions[f'attr_{i:03d}']=j+1
        else:add(value,'fixed_'+field)
    cond={'C0':'C0_original','C1':'C1_reverse_attribute_order'}.get(s,'C2_random_attribute_order_s'+s[-1])
    assert text==rep.build(p,cond,list(range(20260911,20260916))),s
    assert all(spans[i]['end']==spans[i+1]['start'] for i in range(len(spans)-1))
    return text,spans,positions,ids

def prepare(repo,out,protocol):
    import torch,transformers,captum
    from transformers import AutoTokenizer
    start=out/'environment/start.json'
    if not start.exists():
        dump(start,{'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repo':str(repo),
                    'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                    'dirty_status':subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=repo,text=True),
                    'agents_md_found':[]})
    env={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
         'versions':{x:importlib.metadata.version(x) for x in ['torch','transformers','captum','tokenizers','numpy','pandas','scipy','matplotlib','pyarrow']},
         'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'cuda':torch.version.cuda,
         'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
         'existing_environment_reused':True,'packages_installed_or_upgraded':False,'threads':4,
         'native_dtype':'FP16 model, FP32 CLS and L2 normalization','gradient_dtype':'FP16-rounded parameters represented in FP32','tf32':False}
    dump(out/'environment/runtime.json',env)
    (out/'environment/pip_freeze.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))
    (out/'environment/nvidia_smi.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
    cfg=read(repo/'phase2/config/phase2.json');spec=next(x for x in cfg['models'] if x['key']=='gte_modernbert')
    assert spec['revision']==protocol['model']['revision'] and spec['max_tokens']==8192 and spec['native_pooling']=='CLS'
    rep=loadmod(repo/'phase2/src/representations.py','frozen_teal_rep')
    common=loadmod(repo/'revision_final_strengthening_20260922/scripts/common.py','frozen_teal_common');common.verify()
    products,queries=common.load('wands');indices=[i for i,p in enumerate(products) if str(p['product_id'])=='24318']
    assert len(indices)==1 and len(products)==42994
    pi=indices[0];p=products[pi];qi=list(map(int,queries.query_id)).index(409)
    assert queries.iloc[qi]['query']=='teal chair'
    dump(out/'inputs/product_record.json',p)
    sources=[repo/x for x in ['phase2/config/phase2.json','phase2/src/encoding.py','phase2/src/representations.py',
       'phase2/data/processed/wands_products.jsonl.gz','phase2/data/processed/wands_queries.csv',
       'revision_final_strengthening_20260922/scripts/common.py','revision_final_strengthening_20260922/BOUNDARY_AUDIT.md',
       'revision_final_strengthening_20260922/data/historical_case_boundary.csv',
       'revision_evidence_20260917/data/encoder_profiles_and_rank_sources.csv',
       'phase3/scripts/analyze_target_only_permutations.py',
       'phase3/results/target_only_permutations/wands_gte_modernbert_pairs.parquet']]
    snap=Path.home()/'.cache/huggingface/hub/models--Alibaba-NLP--gte-modernbert-base/snapshots'/spec['revision']
    assert (snap/'model.safetensors').is_file(),'Pinned checkpoint unavailable'
    sources+=list(snap.glob('*.*'))
    tokenizer=AutoTokenizer.from_pretrained(spec['name'],revision=spec['revision'],local_files_only=True)
    sequences=[];tokenrows=[];input_checks=[];occurrences=[]
    for s in STEMS:
        text,spans,pos,ids=serialize(rep,p,s)
        assert text==common.serialize(p,s)
        (out/'inputs'/f'{s}.txt').write_bytes(text.encode('utf8'))
        savedpath=repo/f'phase2/data/representations/wands/{s}.jsonl.gz'
        if savedpath.exists():
            with gzip.open(savedpath,'rt',encoding='utf8') as f:
                saved=next(json.loads(line) for line in f if str(json.loads(line)['product_id'])=='24318')
            assert saved['text']==text,(s,'saved serialization differs')
            sources.append(savedpath)
        else: raise FileNotFoundError(savedpath)
        tok=tokenizer(text,truncation=False,return_offsets_mapping=True,return_special_tokens_mask=True)
        assert len(tok['input_ids'])==1059<=8192
        labels=[]
        for j,((a,b),special,mask) in enumerate(zip(tok['offset_mapping'],tok['special_tokens_mask'],tok['attention_mask'])):
            hit=[x for x in spans if b>a and a>=x['start'] and b<=x['end']]
            entry='padding' if not mask else 'special' if special else hit[0]['entry_id'] if len(hit)==1 else 'boundary'
            labels.append(entry)
            tokenrows.append({'schedule':s,'token_index':j,'token_id':tok['input_ids'][j],
                'token':tokenizer.convert_ids_to_tokens(tok['input_ids'][j]),'start':a,'end':b,
                'special':bool(special),'attention_mask':mask,'entry_id':entry,
                'crossed_entries':json.dumps([x['entry_id'] for x in spans if max(a,x['start'])<min(b,x['end'])])})
        dump(out/'inputs'/f'{s}_tokenization.json',{**dict(tok),'entry_ids':labels,'character_spans':spans,
               'attribute_positions':pos,'special_positions':[j for j,v in enumerate(tok['special_tokens_mask']) if v]})
        for j,i in enumerate(ids):
            key=f'attr_{i:03d}';sp=next(x for x in spans if x['entry_id']==key)
            ti=[k for k,(a,b) in enumerate(tok['offset_mapping']) if max(a,sp['start'])<min(b,sp['end'])]
            occurrences.append({'schedule':s,'entry_id':key,'attribute_position':j+1,'text':p['attributes'][i],
                 'char_start':sp['start'],'char_end':sp['end'],'overlapping_token_indices':json.dumps(ti),
                 'token_start_inclusive':min(ti) if ti else None,'token_end_exclusive':max(ti)+1 if ti else None,
                 'assigned_token_indices':json.dumps([k for k,v in enumerate(labels) if v==key])})
            sequences.append({'schedule':s,'position':j+1,'entry_id':key,'text':p['attributes'][i]})
        input_checks.append({'schedule':s,'sha256':hashlib.sha256(text.encode()).hexdigest(),'tokens':len(labels),
             'frozen_serializer_equal':True,'saved_serialization_equal':True,'original_characters_preserved':sorted(text)==sorted(serialize(rep,p,'C0')[0])})
    writecsv(out/'inputs/attribute_sequences.csv',sequences);writecsv(out/'inputs/occurrence_spans.csv',occurrences)
    writecsv(out/'inputs/token_spans.csv',tokenrows);dump(out/'inputs/input_checks.json',input_checks)
    prov=common.provenance();prov=prov[(prov.dataset=='wands')&(prov.model=='gte_modernbert')].set_index('schedule')
    hist=pd.read_csv(repo/'revision_final_strengthening_20260922/data/historical_case_boundary.csv',float_precision='round_trip')
    hist=hist[(hist.model=='gte_modernbert')&(hist.query_id==409)&(hist.product_id==24318)].set_index('schedule')
    target=pick(common.pairread(repo/'phase3/results/target_only_permutations/wands_gte_modernbert_pairs.parquet'))
    qpath=repo/prov.loc['C0','query_embedding'];ppath=repo/prov.loc['C0','product_embedding']
    qcache=np.load(qpath,mmap_mode='r');pcache=np.load(ppath,mmap_mode='r');q=np.array(qcache[qi],copy=True)
    assert qcache.shape==(480,768) and pcache.shape==(42994,768) and q.dtype==np.float32
    assert abs(float(np.linalg.norm(q))-1)<1e-5
    norms=np.linalg.norm(pcache,axis=1);assert np.max(abs(norms-1))<1e-5
    # Preserve original full-query FP32 GEMM shape and exact comparisons, NOT GEMV.
    scores=(qcache@pcache.T)[qi].copy()
    comp=np.array([i for i in range(len(products)) if i!=pi]);order=comp[np.lexsort((comp,-scores[comp]))]
    threshold_index=int(order[19]);threshold=float(scores[threshold_index])
    np.save(out/'fixed_query_embedding.npy',q)
    writecsv(out/'fixed_competitors.csv',[{'catalog_index':int(i),'product_id':str(products[i]['product_id']),'C0_score':float(scores[i])} for i in comp])
    records=[];checks=[]
    for s in STEMS:
        rp=repo/prov.loc[s,'rank_source'];ep=repo/prov.loc[s,'product_embedding'];sources += [rp,ep,ep.with_suffix('.json')]
        row=pick(common.pairread(rp));sv=float(scores[pi]) if s=='C0' else float(row.score)
        dot=float(q@np.load(ep,mmap_mode='r')[pi]);rank=int(common.old_rank_function()(scores,np.array([pi]),np.array([sv],dtype='f4'))[0])
        brute=1+int(np.sum((scores[comp]>sv)|((scores[comp]==sv)&(comp<pi))))
        assert rank==brute
        native_rank=int(target[s]);h=hist.loc[s];included=bool(native_rank<=20)
        match=(sv==float(h.score)==float(row.score) and rank==native_rank==int(h.saved_rank)
           and threshold==float(h.competitor_threshold_score) and threshold_index==int(h.competitor_threshold_index)
           and str(products[threshold_index]['product_id'])==str(int(h.competitor_threshold_id)) and abs(dot-sv)<=2e-6)
        rec={'schedule':s,'score':sv,'saved_rank':native_rank,'reconstructed_rank':rank,'cache_dot_score':dot,
             'cache_dot_minus_saved':dot-sv,'inclusion':included,'reconstructed_inclusion':rank<=20,
             'threshold':threshold,'threshold_competitor_id':str(products[threshold_index]['product_id']),
             'threshold_competitor_index':threshold_index,'signed_margin':sv-threshold,'catalog_index':pi,
             'catalog_size':len(products),'competitors':len(comp),'tokens':1059,'token_cap':8192,'passed':bool(match)}
        records.append(rec);checks.append({'schedule':s,'passed':bool(match),'rank_mismatch':rank!=native_rank})
    writecsv(out/'native_records.csv',records)
    dump(out/'cache_checks.json',{'passed':all(x['passed'] for x in checks),'checks':checks,'query_index':qi,'catalog_index':pi,
         'query_norm':float(np.linalg.norm(q)),'product_norm_min':float(norms.min()),'product_norm_max':float(norms.max()),
         'query_vector_hash':arrhash(q),'score_computation':'full 480 x 768 @ 768 x 42994 FP32 GEMM; target excluded; stable original catalog-index ties; no epsilon'})
    sources+=[qpath,qpath.with_suffix('.json'),HERE/'PROTOCOL.json',Path(__file__),
       Path(inspect.getfile(captum.attr.IntegratedGradients)),Path(inspect.getfile(transformers.models.modernbert.modeling_modernbert))]
    hash_sources(out,sources)
    assert all(x['passed'] for x in checks),'Native/cache provenance mismatch; see retained checks'
    print('PREPARED: 104 occurrences; seven 1059-token inputs; ranks',[x['saved_rank'] for x in records],flush=True)
    return spec,records,p,env

def infer(repo,out,protocol,spec,records,product,report):
    import torch
    from transformers import AutoModel,AutoTokenizer
    from captum.attr import IntegratedGradients
    if not torch.cuda.is_available(): raise RuntimeError('Required CUDA GPU unavailable')
    torch.set_num_threads(4);torch.manual_seed(20260904);torch.cuda.manual_seed_all(20260904)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AutoModel.from_pretrained(spec['name'],revision=spec['revision'],local_files_only=True,attn_implementation='sdpa').cuda().eval().half()
    model.requires_grad_(False)
    tokenizer=AutoTokenizer.from_pretrained(spec['name'],revision=spec['revision'],local_files_only=True)
    q=torch.from_numpy(np.load(out/'fixed_query_embedding.npy')).cuda()
    tok={s:read(out/'inputs'/f'{s}_tokenization.json') for s in STEMS}
    supported=inspect.signature(model.forward).parameters
    inputs={s:{k:torch.tensor([tok[s][k]],device='cuda') for k in ['input_ids','attention_mask'] if k in supported} for s in STEMS}
    assert 'inputs_embeds' in supported
    profile={'class':type(model).__name__,'forward_signature':str(inspect.signature(model.forward)),
       'supports_inputs_embeds':True,'injection':'native inputs_embeds feeds ModernBertEmbeddings norm and dropout before original encoder',
       'attention_implementation':model.config._attn_implementation,'reference_compile_initial':model.config.reference_compile,
       'tf32':False,'native_batch_size':1,'historical_batching':'length-bucket batch size 2 above 1024 tokens',
       'historical_compile_state':'not recorded in original cache; cannot claim bitwise execution equivalence',
       'profile_changes':[],'model_config':model.config.to_dict()}
    dump(out/'environment/model_profile.json',profile)
    (out/'environment/modernbert_forward_source.txt').write_text(inspect.getsource(type(model).forward)+'\n'+inspect.getsource(type(model.embeddings)),encoding='utf8')
    def score_ids(s):
        hidden=model(**inputs[s]).last_hidden_state
        return torch.nn.functional.normalize(hidden[:,0].float(),dim=1)@q
    fresh={}
    try:
        with torch.inference_mode():
            for s in STEMS: fresh[s]=float(score_ids(s).item())
    except Exception as e:
        dump(out/'environment/native_initial_failure.json',{'error':repr(e),'traceback':traceback.format_exc()})
        if not model.config.reference_compile: raise
        model.config.reference_compile=False
        profile['profile_changes'].append('reference_compile disabled after recorded native failure; SDPA unchanged')
        with torch.inference_mode():
            for s in STEMS:fresh[s]=float(score_ids(s).item())
    profile['reference_compile_native']=model.config.reference_compile
    # Hash native parameter values before converting THAT model, not reloading a checkpoint.
    hh=hashlib.sha256()
    for name,par in model.named_parameters():hh.update(name.encode());hh.update(par.detach().cpu().numpy().tobytes())
    profile['fp16_parameter_hash']=hh.hexdigest()
    model.float()
    hh=hashlib.sha256()
    for name,par in model.named_parameters():
        assert torch.equal(par,par.half().float())
        hh.update(name.encode());hh.update(par.detach().cpu().numpy().tobytes())
    profile['fp32_rounded_parameter_hash']=hh.hexdigest()
    profile['same_parameters_roundtrip_exact']=True
    # Disable reference compilation only when required by gradient execution, and log it.
    prepared={};forward_checks=[]
    comp=pd.read_csv(out/'fixed_competitors.csv',float_precision='round_trip')
    native={r['schedule']:r for r in records}
    for s in STEMS:
        r=native[s];fs=fresh[s]
        rank=1+int(((comp.C0_score>fs)|((comp.C0_score==fs)&(comp.catalog_index<r['catalog_index']))).sum())
        forward_checks.append({'schedule':s,'saved_native':r['score'],'saved_rank':r['saved_rank'],
              'fresh_native':fs,'fresh_minus_saved':fs-r['score'],'fresh_rank':rank,'fresh_inclusion':rank<=20})
    def make_forward(s):
        mask=inputs[s]['attention_mask']
        def forward_embeddings(value):
            hidden=model(inputs_embeds=value,attention_mask=mask.expand(value.shape[0],-1)).last_hidden_state
            return torch.nn.functional.normalize(hidden[:,0].float(),dim=1)@q
        return forward_embeddings
    for s in ENDS:
        emb=model.get_input_embeddings()(inputs[s]['input_ids']).detach()
        fn=make_forward(s)
        with torch.no_grad(): fid=float(score_ids(s).item());actual=float(fn(emb).item())
        probe=emb.clone().requires_grad_(True)
        grad=torch.autograd.grad(fn(probe).sum(),probe)[0]
        finite=bool(torch.isfinite(grad).all());gradnorm=float(grad.norm().item())
        assert finite and gradnorm>0 and abs(fid-actual)<=1e-6,'Forward/gradient equivalence failed'
        r=native[s];margin=r['signed_margin'];th=r['threshold'];ti=r['threshold_competitor_index'];pi=r['catalog_index']
        include=lambda v:v>th or (v==th and pi<ti)
        guard=include(actual)==r['inclusion'] and include(fresh[s])==r['inclusion'] and abs(margin)>5*max(abs(actual-r['score']),abs(fresh[s]-r['score']))
        fc=next(x for x in forward_checks if x['schedule']==s)
        fc.update(diagnostic=actual,diagnostic_minus_saved=actual-r['score'],diagnostic_minus_fresh=actual-fresh[s],
                  diagnostic_inclusion=include(actual),input_ids_vs_embeds_error=abs(fid-actual),gradient_finite=finite,
                  gradient_norm=gradnorm,boundary_guard=guard,native_margin=margin)
        prepared[s]={'emb':emb,'fn':fn,'actual':actual,'guard':guard}
        del grad,probe
    profile['reference_compile_diagnostic']=model.config.reference_compile
    profile['cuda_peak_memory_allocated']=torch.cuda.max_memory_allocated()
    dump(out/'environment/model_profile.json',profile);writecsv(out/'forward_checks.csv',forward_checks)
    dc=prepared['C2s5']['actual']-prepared['C0']['actual'];nc=native['C2s5']['score']-native['C0']['score']
    cc={'native_contrast':nc,'diagnostic_contrast':dc,'difference':dc-nc,'tolerance':max(1e-4,.01*abs(nc)),
        'passed':abs(dc-nc)<=max(1e-4,.01*abs(nc))}
    dump(out/'native_contrast_check.json',cc)
    report['forward_ready']=all(prepared[s]['guard'] for s in ENDS) and cc['passed']
    assert report['forward_ready'],'Native agreement / boundary checks failed'
    entries=[(f'attr_{i:03d}',v) for i,v in enumerate(product['attributes'])]
    entries += [('fixed_'+f,product[f]) for f in product['section_order'] if f!='attributes' and product.get(f)]
    entries += [(x,x.replace('_',' ')) for x in ['attribute_separators','section_separators','boundary','special','padding']]
    entrykeys=[k for k,v in entries]
    attempts=[];convergence=[];baseline_checks=[];all_tokens=[];all_entries=[];all_changes=[]
    for bname in ['PAD','Zero']:
        report['baselines'][bname]={'status':'running','passed':False}
        base={};baseline_scores={}
        for s in ENDS:
            v=prepared[s];base[s]=v['emb'].clone()
            change=torch.tensor([bool(m) and not bool(sp) for m,sp in zip(tok[s]['attention_mask'],tok[s]['special_tokens_mask'])],device='cuda')
            base[s][:,change,:]=model.get_input_embeddings().weight[tokenizer.pad_token_id].detach() if bname=='PAD' else 0
            assert torch.equal(base[s][:,~change],v['emb'][:,~change])
            with torch.no_grad():baseline_scores[s]=float(v['fn'](base[s]).item())
            np.savez_compressed(out/f'inputs/{s}_{bname}_embeddings.npz',input_embeddings=v['emb'].cpu().numpy(),
                         baseline_embeddings=base[s].cpu().numpy(),attention_mask=inputs[s]['attention_mask'].cpu().numpy())
        same_tensor=bool(torch.equal(base['C0'],base['C2s5']));same_masks=tok['C0']['attention_mask']==tok['C2s5']['attention_mask']
        bc={'baseline':bname,'same_baseline_tensor':same_tensor,'same_masks':same_masks,
            'same_special_positions':tok['C0']['special_positions']==tok['C2s5']['special_positions'],
            'C0_baseline_hash':arrhash(base['C0'].cpu().numpy()),'C2s5_baseline_hash':arrhash(base['C2s5'].cpu().numpy()),
            'F_C0':prepared['C0']['actual'],'F_C2s5':prepared['C2s5']['actual'],
            'F_b_C0':baseline_scores['C0'],'F_b_C2s5':baseline_scores['C2s5'],
            'baseline_score_difference':baseline_scores['C2s5']-baseline_scores['C0']}
        baseline_checks.append(bc);dump(out/'baseline_score_checks.json',baseline_checks)
        assert same_tensor and same_masks and bc['same_special_positions'] and bc['baseline_score_difference']==0,'Unequal paired baseline requires diagnosis'
        prev=None;chosen=None
        for n in protocol['method']['nodes']:
            pair={};pair_pass=True
            for s in ENDS:
                print(f'IG {bname} {s}: {n} Gauss-Legendre nodes, internal_batch_size=1',flush=True)
                attempt_path=out/'attempts'/bname/f'n{n:03d}'/s;attempt_path.mkdir(parents=True)
                started=time.perf_counter();v=prepared[s]
                dump(attempt_path/'started.json',{'baseline':bname,'schedule':s,'nodes':n,'protocol_hash':sha(out/'PROTOCOL.json')})
                try:
                    ig=IntegratedGradients(v['fn'],multiply_by_inputs=True)
                    a,delta=ig.attribute(v['emb'],baselines=base[s],n_steps=n,method='gausslegendre',internal_batch_size=1,return_convergence_delta=True)
                    aa=a.detach().cpu().numpy();assert np.isfinite(aa).all()
                    # FP64 accounting of the saved FP32 attribution tensor prevents avoidable aggregation rounding.
                    tv=aa.astype(np.float64).sum(axis=-1)[0]
                    ev=np.array([math.fsum(float(tv[j]) for j,k in enumerate(tok[s]['entry_ids']) if k==key) for key in entrykeys])
                    ts=float(aa.sum(dtype=np.float64));ss=math.fsum(tv);es=math.fsum(ev)
                    residual=es-(v['actual']-baseline_scores[s]);tol=max(1e-4,.001*abs(v['actual']-baseline_scores[s]))
                    aggregation=max(abs(ts-ss),abs(ss-es))<=1e-9
                    passed=abs(residual)<=tol and aggregation
                    qa={'baseline':bname,'schedule':s,'nodes':n,'status':'passed' if passed else 'failed',
                        'F_x':v['actual'],'F_b':baseline_scores[s],'tensor_sum':ts,'token_sum':ss,'entry_sum':es,
                        'residual':residual,'tolerance':tol,'completeness_passed':abs(residual)<=tol,
                        'aggregation_passed':aggregation,'captum_delta':float(delta.item()),'seconds':time.perf_counter()-started,
                        'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
                    np.savez_compressed(attempt_path/'feature_attributions.npz',attribution=aa)
                    tr=[{'baseline':bname,'schedule':s,'nodes':n,'token_index':j,'entry_id':tok[s]['entry_ids'][j],
                       'token_id':tok[s]['input_ids'][j],'token':tokenizer.convert_ids_to_tokens(tok[s]['input_ids'][j]),
                       'start':tok[s]['offset_mapping'][j][0],'end':tok[s]['offset_mapping'][j][1],'attribution':float(num)} for j,num in enumerate(tv)]
                    er=[{'baseline':bname,'schedule':s,'nodes':n,'entry_id':key,'entry_text':label,'attribution':float(num),
                          'position':tok[s]['attribute_positions'].get(key)} for (key,label),num in zip(entries,ev)]
                    writecsv(attempt_path/'token_attributions.csv',tr);writecsv(attempt_path/'entry_attributions.csv',er)
                    pair[s]={'values':ev,'tokens':tr,'entries':er,'qa':qa};pair_pass &= passed
                    del a,aa,delta
                except Exception as exc:
                    qa={'baseline':bname,'schedule':s,'nodes':n,'status':'error','error':repr(exc),'traceback':traceback.format_exc(),'seconds':time.perf_counter()-started}
                    pair_pass=False
                    torch.cuda.empty_cache()
                dump(attempt_path/'qa.json',qa);attempts.append(qa);writecsv(out/'completeness_attempts.csv',attempts)
            if len(pair)!=2:
                report['baselines'][bname]={'status':'failed','passed':False,'reason':'Endpoint execution error; retained attempts','nodes':n}
                break
            d=pair['C2s5']['values']-pair['C0']['values']
            expected=dc-bc['baseline_score_difference'];res=math.fsum(d)-expected;btol=max(1e-4,.01*abs(dc))
            l1=None if prev is None else float(np.abs(d-prev).sum());linf=None if prev is None else float(np.abs(d-prev).max())
            ctol=max(1e-4,.02*float(np.abs(d).sum()));cpass=prev is not None and l1<=ctol
            row={'baseline':bname,'nodes':n,'previous_nodes':None if prev is None else n//2,'A_passed':bool(pair_pass),
                 'contrast_sum':math.fsum(d),'expected_contrast':expected,'contrast_residual':res,'B_tolerance':btol,'B_passed':abs(res)<=btol,
                 'L1_difference':l1,'Linf_difference':linf,'D_L1':float(np.abs(d).sum()),'C_tolerance':ctol,'C_passed':bool(cpass)}
            convergence.append(row);writecsv(out/'step_convergence.csv',convergence)
            changes=[{'baseline':bname,'nodes':n,'entry_id':key,'entry_text':label,
                'C0_position':tok['C0']['attribute_positions'].get(key),'C2s5_position':tok['C2s5']['attribute_positions'].get(key),
                'IG_C0':float(pair['C0']['values'][i]),'IG_C2s5':float(pair['C2s5']['values'][i]),'C2s5_minus_C0':float(d[i]),
                'previous_difference':None if prev is None else float(prev[i]),'step_change':None if prev is None else float(d[i]-prev[i])}
                for i,(key,label) in enumerate(entries)]
            writecsv(out/'attempts'/bname/f'n{n:03d}'/'aligned_entry_changes.csv',changes)
            chosen=(pair,changes,n)
            passed=pair_pass and abs(res)<=btol and cpass
            print('PAIRED',bname,n,row,flush=True)
            report['baselines'][bname]={'status':'passed' if passed else 'inconclusive','passed':bool(passed),'nodes':n,
                                      'A_passed':bool(pair_pass),'B_passed':abs(res)<=btol,'C_passed':bool(cpass),
                                      'reason':None if passed else 'A/B/C not simultaneously satisfied'}
            dump(out/'run_report.json',report)
            if passed: break
            prev=d.copy()
        if chosen:
            pair,changes,n=chosen;all_changes+=changes
            for s in ENDS:all_tokens+=pair[s]['tokens'];all_entries+=pair[s]['entries']
        report['PAD_plot_ready']=bool(report['forward_ready'] and report['baselines'].get('PAD',{}).get('passed',False))
        dump(out/'run_report.json',report)
    writecsv(out/'token_attributions.csv',all_tokens);writecsv(out/'entry_attributions.csv',all_entries)
    writecsv(out/'aligned_entry_changes.csv',all_changes)
    for b in ['PAD','Zero']:writecsv(out/f'aligned_entry_changes_{b}.csv',[r for r in all_changes if r['baseline']==b])
    maps={b:{r['entry_id']:r for r in all_changes if r['baseline']==b} for b in ['PAD','Zero']}
    sens=[]
    if maps['PAD'] and maps['Zero']:
        for key,label in entries:
            a=maps['PAD'][key]['C2s5_minus_C0'];b=maps['Zero'][key]['C2s5_minus_C0'];eps=protocol['sensitivity_near_zero']
            category='both_near_zero' if max(abs(a),abs(b))<=eps else 'one_near_zero' if min(abs(a),abs(b))<=eps else 'same_direction' if a*b>0 else 'opposite_direction'
            sens.append({'entry_id':key,'entry_text':label,'PAD_order_difference':a,'Zero_order_difference':b,
                         'Zero_minus_PAD_order_difference':b-a,'category':category,'near_zero_threshold':eps,
                         'PAD_validated':report['baselines']['PAD']['passed'],'Zero_validated':report['baselines']['Zero']['passed']})
        writecsv(out/'baseline_sensitivity.csv',sens)
    report['four_IG_statuses']={f'{s}/{b}':{'endpoint_completeness':next((a.get('completeness_passed',False) for a in reversed(attempts) if a['baseline']==b and a['schedule']==s),False),
          'paired_validation_status':report['baselines'][b]['status'],'nodes':report['baselines'][b].get('nodes')} for b in ['PAD','Zero'] for s in ENDS}
    report['overall_status']='passed' if all(x['passed'] for x in report['baselines'].values()) else 'PAD_passed_Zero_failed' if report['PAD_plot_ready'] else 'inconclusive'
    report['baseline_robust_claim_allowed']=False
    report['execution_profile_identical_claim_allowed']=False
    dump(out/'run_report.json',report)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--repo',type=Path,default=HERE.parent)
    ap.add_argument('--output',type=Path,default=HERE);ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    repo=args.repo.resolve();out=args.output.resolve()
    if (out/'run_report.json').exists():
        raise SystemExit('Existing result preserved; choose a new --output run_02, run_03, etc.')
    out.mkdir(parents=True,exist_ok=True)
    if out!=HERE:
        for name in ['PROTOCOL.json','run_teal_captum.py','plot_teal.py','validate_teal.py']:
            if (HERE/name).exists():shutil.copy2(HERE/name,out/name)
    protocol=read(out/'PROTOCOL.json');phash=sha(out/'PROTOCOL.json')
    (out/'PROTOCOL.sha256').write_text(phash+'\n',encoding='ascii')
    report={'overall_status':'running','native_ready':False,'PAD_plot_ready':False,'forward_ready':False,
      'case':protocol['case'],'contrast':protocol['difference'],'protocol_sha256':phash,
      'baselines':{b:{'status':'not_run','passed':False} for b in ['PAD','Zero']},'failure_reasons':[]}
    dump(out/'run_report.json',report)
    try:
        spec,records,p,env=prepare(repo,out,protocol);report['native_ready']=True;report['environment']=env
        dump(out/'run_report.json',report)
        if not args.prepare_only:infer(repo,out,protocol,spec,records,p,report)
        else:report['overall_status']='prepared_only'
    except Exception as exc:
        report['overall_status']='blocked';report['failure_reasons'].append(repr(exc))
        dump(out/'run_failure.json',{'error':repr(exc),'traceback':traceback.format_exc()})
        print(traceback.format_exc(),flush=True)
    assert sha(out/'PROTOCOL.json')==phash,'Protocol changed during execution'
    dump(out/'run_report.json',report)
    print('FINAL',report['overall_status'],'native_ready',report['native_ready'],'PAD_plot_ready',report['PAD_plot_ready'],flush=True)
    return 0 if report['overall_status']=='passed' else 2
if __name__=='__main__':raise SystemExit(main())
