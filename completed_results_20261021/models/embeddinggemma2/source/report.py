"""Report only verified completed cells; explicitly retain blocked and unrun stages."""
from common import *

def run():
    design=protocol();statuses=[];rows=[];summary=HERE/'tables/main_candidates.csv'
    table=pd.read_csv(summary) if summary.exists() else pd.DataFrame()
    for model in design['models']:
        for ds in ['wands','esci']:
            for stage in ['smoke','encode','evaluate','boundary','canonical']:
                path=HERE/f'status/{model}_{"all" if stage=="smoke" else ds}_{stage}.json'
                value=read(path) if path.exists() else dict(status='not_run',reason='Prerequisite incomplete or stage not executed')
                statuses.append(dict(model=model,dataset=ds,stage=stage,**{k:v for k,v in value.items() if k not in ['model','dataset','stage']}))
    initial=read(HERE/'audit/preexisting_changes.json');preserved={p:sha(ROOT/p)==s for p,s in initial.items()};assert all(preserved.values())
    manifest=dict(updated_utc=now(),git_commit=design['git_commit'],protocol_sha256=sha(HERE/'PROTOCOL.json'),models=design['models'],hardware=design['hardware'],software=design['software'],stages=statuses,
        source_files=design['source_files'],preexisting_modified_files_preserved=preserved,active_models=design.get('active_models',list(design['models'])),deferred_models=design.get('deferred_models',{}),scope_amendments=design.get('scope_amendments',[]),
        files=[source(p) for p in HERE.rglob('*') if p.is_file() and not any(x in p.parts for x in ['.venv','cache','models','__pycache__']) and p.name not in ['manifest.json'] and not p.name.endswith('.tmp')])
    dump(HERE/'manifest.json',manifest)
    pd.DataFrame([{k:r.get(k,'') for k in ['model','dataset','stage','status','reason']} for r in statuses]).to_csv(HERE/'tables/stage_status.csv',index=False)
    lines=['# 新模型補充實驗執行報告','',f'更新時間（UTC）：{now()}。本次是看到既有研究結果後制定的新模型 follow-up，不是整篇研究的 preregistration。','',
        '目前執行範圍：'+', '.join(design.get('active_models',list(design['models'])))+'。暫緩模型：'+(', '.join(design.get('deferred_models',{})) or '無')+'；暫緩是使用者指示，不是模型失敗，恢復前須有使用者明確指示。','',
        '## 已完成的資料與實作核對','',
        '- WANDS：完整 42,994 products；96 development／384 held-out；308 eligible queries，21,299 Exact pairs。',
        '- ESCI：既有 500-query sample 的完整 10,076-product pool；499 eligible queries，4,434 E pairs。',
        '- 七種 raw schedules 的全部商品文字與歷史保存字串逐筆相等；attribute multiplicity、固定區塊與三個 canonical rules 的輸入不變性均已核對。',
        '- 新增專用 adapter；舊 MiniLM/BGE/GTE 程式及既有未提交修改的 hashes 保持不變。',
        '- 修正後 direct gains、query-macro、exact rank insertion、tie、自我移除、membership cancellation 與 checkpoint integrity 已有驗收測試。','',
        '## 模型與階段狀態','','| Model | Dataset | Stage | Status |','|---|---|---|---|']
    lines += [f"| {r['model']} | {r['dataset']} | {r['stage']} | {r['status']} |" for r in statuses]
    lines += ['','## 正式結果','']
    if len(table):
        for r in table.to_dict('records'):
            lines.append(f"- {r['model']} / {r['dataset']}: C0 Recall@20={r['C0_Recall']:.6f}; raw mean={r['raw_mean_Recall']:.6f}; target-only VI@20={r['target_VI']:.6f} [{r['target_VI_ci_low']:.6f}, {r['target_VI_ci_high']:.6f}]; catalog-wide VI@20={r['catalog_VI']:.6f} [{r['catalog_VI_ci_low']:.6f}, {r['catalog_VI_ci_high']:.6f}].")
    else:lines.append('尚無通過完整 catalog encoding 與 evaluation 完成旗標的正式結果；不以技術樣本估計 VI 或填入零值。')
    lines += ['','## 技術驗證、阻礙與限制','']
    for model in design['models']:
        path=HERE/f'smoke/{model}/validation.json'
        if path.exists():
            v=read(path)
            if v['status']=='completed':lines.append(f"- {model}: 官方 reference 比對通過；三次固定 profile repeated-input score range={v['max_repeat_score_range']}; rank changes={v['repeat_rank_changes']}。Batch-size 差異另存，不能混合執行 profiles。")
            elif v['status']=='blocked':lines.append(f"- {model}: blocked — {v.get('exception')}: {v.get('reason')}")
            else:lines.append(f"- {model}: {v['status']}；技術驗證尚未完成，尚不能產生正式結果。")
    for r in statuses:
        if r['status']=='blocked' and r['stage']!='smoke':lines.append(f"- {r['model']} / {r['dataset']} / {r['stage']}: {r.get('reason','')}。")
    lines += ['','未完成模型不替換、不縮小維度、不量化。尚未執行的 canonical 階段不宣稱 VI=0 的實測結果。所有結果僅適用於既定模型、catalog、query support、七種 schedules 與 K；不同模型 fitting supports 的差異不是同一母體比較，full/fitting 差異不是 truncation 因果效果。CI 含零不表示等效。','',
        '重現命令見 RUN_COMMANDS.md；query-level evidence、pair ranks、full scores 與 token audit 只在對應階段完成後產生。']
    (HERE/'REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    tex=[r'% Supplement only; original manuscript is not overwritten.',r'% This is a new-model follow-up after inspection of historical outcomes.',r'% Amend Section 4.2: the original no-prefix / mean-or-CLS descriptions apply only to the original three encoders.']
    if len(table):
        tex += [r'\paragraph{Recent-model follow-up.} We retained the original catalogs, query supports, judgments, and seven attribute schedules. Dedicated model-native preprocessing, pooling, and projection were used with full-dimensional unit-normalized embeddings and exact FP32 ranking. This follow-up was specified after inspecting the historical study results.']
        if design.get('deferred_models'):tex += [r"Qwen3-Embedding-4B was deferred at the user's request after the EmbeddingGemma 2/ESCI result became available. It has no completed retrieval result in this follow-up. The remaining active-model experiments and canonical comparisons use EmbeddingGemma 2; this scope amendment is documented separately."]
        for r in table.to_dict('records'):
            tex.append(f"For {r['model'].replace('_',r' ')} on {r['dataset'].upper()}, source-order Recall@20 was {r['C0_Recall']:.4f}; target-only VI@20 was {100*r['target_VI']:.2f}\\% (95\\% query-bootstrap CI [{100*r['target_VI_ci_low']:.2f}, {100*r['target_VI_ci_high']:.2f}]).")
    else:tex += [r'% No full-catalog new-model retrieval result has completed. No empirical result paragraph is supplied.']
    (HERE/'MANUSCRIPT_ADDITIONS.tex').write_text('\n'.join(tex)+'\n',encoding='utf8')
    status('report',state='completed',formal_result_cells=len(table))
    from finalize_manifest import main as finalize
    finalize()

