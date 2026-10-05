"""Independent accounting, artifact and preservation audit; no model inference."""
from pathlib import Path
import argparse, collections, hashlib, html, json, math, re, xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from pypdf import PdfReader
from run_teal_captum import read,dump,sha,arrhash,STEMS,ENDS

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);args=ap.parse_args();out=args.output.resolve()
    r=read(out/'run_report.json');protocol=read(out/'PROTOCOL.json');checks=[]
    def ck(name,value,detail=None):checks.append({'name':name,'passed':bool(value),'detail':detail})
    ck('protocol_frozen_before_IG',sha(out/'PROTOCOL.json')==r['protocol_sha256']==(out/'PROTOCOL.sha256').read_text().strip())
    ck('four_combinations_only',set(r.get('four_IG_statuses',{}))=={f'{s}/{b}' for s in ENDS for b in ['PAD','Zero']})
    native=pd.read_csv(out/'native_records.csv',float_precision='round_trip');comp=pd.read_csv(out/'fixed_competitors.csv',float_precision='round_trip')
    ck('catalog_target_resolved_from_ID',len(comp)==42993 and not (comp.product_id==24318).any())
    ix=int(native.catalog_index.iloc[0]);ordered=comp.sort_values(['C0_score','catalog_index'],ascending=[False,True],kind='stable')
    threshold=ordered.iloc[19]
    for row in native.to_dict('records'):
        rank=1+int(((comp.C0_score>row['score'])|((comp.C0_score==row['score'])&(comp.catalog_index<ix))).sum())
        ck('native_rank_'+row['schedule'],rank==row['saved_rank']==row['reconstructed_rank'])
        ck('fixed_threshold_'+row['schedule'],row['threshold']==float(threshold.C0_score) and row['threshold_competitor_index']==int(threshold.catalog_index))
    tok={s:read(out/'inputs'/f'{s}_tokenization.json') for s in STEMS}
    texts=[(out/'inputs'/f'{s}.txt').read_bytes().decode('utf8') for s in STEMS]
    p=read(out/'inputs/product_record.json');nattrs=len(p['attributes']);groups=[f'attr_{i:03d}' for i in range(nattrs)]
    groups+=['fixed_'+f for f in p['section_order'] if f!='attributes' and p.get(f)]
    groups+=['attribute_separators','section_separators','boundary','special','padding']
    for s,text in zip(STEMS,texts):
        t=tok[s];ck(s+'_length_untruncated',len(t['input_ids'])==1059<=8192)
        ck(s+'_characters_preserved',collections.Counter(text)==collections.Counter(texts[0]))
        ck(s+'_occurrence_positions',sorted(t['attribute_positions'].values())==list(range(1,nattrs+1)))
        expected=[]
        for j,((a,b),special,mask) in enumerate(zip(t['offset_mapping'],t['special_tokens_mask'],t['attention_mask'])):
            contained=[sp for sp in t['character_spans'] if b>a and a>=sp['start'] and b<=sp['end']]
            expected.append('padding' if not mask else 'special' if special else contained[0]['entry_id'] if len(contained)==1 else 'boundary')
        ck(s+'_every_token_assigned_once',expected==t['entry_ids'] and len(expected)==1059)
        for sp in t['character_spans']:
            ck(s+'_span_'+str(sp['start']),text[sp['start']:sp['end']]==sp['text'])
    from safetensors import safe_open
    checkpoint=Path.home()/'.cache/huggingface/hub/models--Alibaba-NLP--gte-modernbert-base/snapshots'/protocol['model']['revision']/'model.safetensors'
    with safe_open(checkpoint,framework='pt',device='cpu') as f:
        weightkey=next(k for k in f.keys() if k.endswith('embeddings.tok_embeddings.weight'))
        padvec=f.get_slice(weightkey)[50283:50284].half().float().numpy()[0]
    baselinechecks=read(out/'baseline_score_checks.json')
    for b in ['PAD','Zero']:
        arrays={}
        for s in ENDS:
            arrays[s]=np.load(out/f'inputs/{s}_{b}_embeddings.npz')
            a=arrays[s];t=tok[s];change=(np.array(t['attention_mask'],dtype=bool)&~np.array(t['special_tokens_mask'],dtype=bool))
            ck(f'{s}_{b}_special_padding_unchanged',np.array_equal(a['input_embeddings'][:,~change],a['baseline_embeddings'][:,~change]))
            expected=np.broadcast_to(padvec if b=='PAD' else np.zeros(768,dtype='f4'),a['baseline_embeddings'][:,change].shape)
            ck(f'{s}_{b}_word_vectors_match_definition',np.array_equal(a['baseline_embeddings'][:,change],expected))
            ck(f'{s}_{b}_masks_retained',np.array_equal(a['attention_mask'][0],t['attention_mask']))
        bc=next(x for x in baselinechecks if x['baseline']==b)
        ck(b+'_baseline_equal_and_hashes',np.array_equal(arrays['C0']['baseline_embeddings'],arrays['C2s5']['baseline_embeddings']) and
            bc['C0_baseline_hash']==arrhash(arrays['C0']['baseline_embeddings'])==bc['C2s5_baseline_hash'])
        ck(b+'_baseline_score_identity',bc['baseline_score_difference']==bc['F_b_C2s5']-bc['F_b_C0']==0)
    all_attempts=[]
    for qp in sorted((out/'attempts').glob('*/*/*/qa.json')):
        q=read(qp);s=q['schedule'];b=q['baseline'];n=q['nodes'];name=f'{b}_{s}_{n}'
        if q['status']=='error':ck(name+'_error_retained',bool(q.get('error')));continue
        a=np.load(qp.parent/'feature_attributions.npz')['attribution'];tr=pd.read_csv(qp.parent/'token_attributions.csv',float_precision='round_trip')
        er=pd.read_csv(qp.parent/'entry_attributions.csv',float_precision='round_trip');tv=a.astype('f8').sum(axis=-1)[0]
        ck(name+'_tensor_token_exact',np.array_equal(tv,tr.attribution.to_numpy()))
        ev=np.array([math.fsum(tv[j] for j,k in enumerate(tok[s]['entry_ids']) if k==key) for key in groups])
        ck(name+'_entry_order_all_groups',list(er.entry_id)==groups)
        ck(name+'_token_entry_exact',np.array_equal(ev,er.attribution.to_numpy()))
        residual=math.fsum(ev)-(q['F_x']-q['F_b']);tol=max(1e-4,.001*abs(q['F_x']-q['F_b']))
        ck(name+'_A_recomputed',abs(residual-q['residual'])<1e-12 and tol==q['tolerance'] and (abs(residual)<=tol)==q['completeness_passed'])
        ck(name+'_captum_delta_agreement',abs(q['captum_delta']-residual)<1e-6)
        ck(name+'_three_level_accounting',max(abs(q['tensor_sum']-q['token_sum']),abs(q['token_sum']-q['entry_sum']))<=1e-9)
        ck(name+'_zero_special_padding',all(v==0 for k,v in zip(groups,ev) if k in ['special','padding']))
        all_attempts.append(q)
    cv=pd.read_csv(out/'step_convergence.csv',float_precision='round_trip')
    for b,g in cv.groupby('baseline',sort=False):
        prev=None
        for row in g.to_dict('records'):
            n=int(row['nodes']);df=pd.read_csv(out/'attempts'/b/f'n{n:03d}'/'aligned_entry_changes.csv',float_precision='round_trip')
            d=df.C2s5_minus_C0.to_numpy();expected=df.IG_C2s5.to_numpy()-df.IG_C0.to_numpy()
            ck(f'{b}_{n}_difference_exact',np.array_equal(d,expected))
            ck(f'{b}_{n}_B_recomputed',abs((math.fsum(d)-row['expected_contrast'])-row['contrast_residual'])<1e-12 and
                (abs(row['contrast_residual'])<=row['B_tolerance'])==row['B_passed'])
            if prev is not None:
                l1=float(np.abs(d-prev).sum());linf=float(np.abs(d-prev).max());tol=max(1e-4,.02*float(np.abs(d).sum()))
                ck(f'{b}_{n}_C_recomputed',abs(l1-row['L1_difference'])<1e-12 and abs(linf-row['Linf_difference'])<1e-12 and tol==row['C_tolerance'] and (l1<=tol)==row['C_passed'])
            else:ck(f'{b}_{n}_first_step_not_stability_pass',not row['C_passed'])
            prev=d
        last=g.iloc[-1];passed=bool(last.A_passed and last.B_passed and last.C_passed)
        ck(b+'_reported_status_honest',passed==r['baselines'][b]['passed'])
        ck(b+'_bounded_staircase',list(g.nodes) in [[64,128],[64,128,256],[64,128,256,512]])
    ck('PAD_plot_gate',r['PAD_plot_ready']==bool(r['baselines']['PAD']['passed'] and r['forward_ready'] and r['native_ready']))
    for sp in (out/'figure_data').glob('*_scene.json'):
        stem=sp.stem.replace('_scene','');pages=read(sp);xml=ET.parse(out/'figures'/f'{stem}.drawio')
        ck(stem+'_native_editable_no_images',not any('image=' in c.get('style','') for c in xml.findall('.//mxCell')))
        ck(stem+'_page_counts',len(pages)==len(xml.findall('diagram'))==len(PdfReader(out/'figures'/f'{stem}.pdf').pages))
        for i,(page,di) in enumerate(zip(pages,xml.findall('diagram')),1):
            cells=di.findall('.//mxCell')[2:];ck(stem+f'_p{i}_object_count',len(cells)==len(page['items']))
            for item,cell in zip(page['items'],cells):
                if item['kind']=='text':
                    value=html.unescape(re.sub('<[^>]*>','',cell.get('value','').replace('<br>','\n')))
                    ck(stem+f'_p{i}_text_'+cell.get('id'),value==item['text'])
    if r['PAD_plot_ready']:
        data=pd.read_csv(out/'aligned_entry_changes_PAD.csv',float_precision='round_trip').set_index('entry_id')
        plotted=pd.read_csv(out/'figure_data/PAD_full_plotted_values.csv',float_precision='round_trip').set_index('entry_id')
        ck('full_plot_all_entries',list(data.index)==list(plotted.index))
        for col in ['IG_C0','IG_C2s5','C2s5_minus_C0']:ck('full_plot_'+col,np.array_equal(data[col].to_numpy(),plotted[col].to_numpy()))
        compact=pd.read_csv(out/'figure_data/PAD_compact_plotted_values.csv',float_precision='round_trip').set_index('entry_id')
        ck('compact_prespecified_first_ten',list(compact.index[:10])==[f'attr_{i:03d}' for i in range(10)])
        for col in ['IG_C0','IG_C2s5','C2s5_minus_C0']:
            ck('other_exact_'+col,compact.loc['other_attributes',col]==math.fsum(data.loc[[f'attr_{i:03d}' for i in range(10,104)],col]))
    else:ck('no_unvalidated_attribution_figure',not list((out/'figures').glob('PAD_attribution*.pdf')))
    protected=out/'environment/protected_files_before.json'
    if protected.exists():
        repo=Path(read(out/'environment/start.json')['repo']);before=read(protected)
        mismatch=[name for name,h in before.items() if not (repo/name).exists() or sha(repo/name)!=h]
        ck('French_Figure1_manuscript_existing_changes_preserved',not mismatch,mismatch)
    source=read(out/'environment/source_hashes.json');changed=[name for name,meta in source.items() if sha(name)!=meta['sha256']]
    ck('source_files_unchanged',not changed,changed)
    result={'status':'passed' if all(c['passed'] for c in checks) else 'failed','check_count':len(checks),
      'failed_checks':[c for c in checks if not c['passed']],'interpretation':'Audit correctness; does not turn failed IG convergence into a pass.','checks':checks}
    dump(out/'qa/artifact_and_numeric_audit.json',result)
    print(result['status'],len(checks),'checks; failures:',result['failed_checks'])
    return 0 if result['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
