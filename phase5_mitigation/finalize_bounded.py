"""Finalize the budget-limited screen; never select an incomplete lambda grid."""
from screen import *
from progress_report import main as progress
def main():
    progress();c=setup();df=pd.read_csv(OUT/'completed_screening_metrics.csv');ci=pd.read_csv(OUT/'completed_paired_bootstrap.csv')
    assert not (HERE/'selection.json').exists(),'Use the complete report when selection exists'
    planned=['Original','Canonical','Set-Mean','Perm-FT_s42','Perm-FT + Consistency (selected)','Set-Attention_s42']
    v1=[];v2=[]
    for name in planned+['Perm-FT-cons0.05_s42 (unselected trial)']:
        key=name.replace(' (unselected trial)','');a=df[(df.method==key)&(df['mode']=='catalog')];b=df[(df.method==key)&(df['mode']=='target')]
        r={'Method':name,'Status':'complete' if len(a) else 'not available'}
        for metric in ['Recall@20','Recall@100','VI@20','VI@100','Robust@100','Never@100']:
            r[metric]=float(a.iloc[0][metric]) if len(a) else np.nan
        r['Delta Recall@100']=r['Recall@100']-float(df[(df.method=='Original')&(df['mode']=='catalog')].iloc[0]['Recall@100']);v1.append(r)
        if name not in ['Canonical','Set-Mean']:
            t={'Method':name,'Status':r['Status']}
            for metric in ['Recall@100','VI@20','VI@100','Robust@100','Never@100']:t[metric]=float(b.iloc[0][metric]) if len(b) else np.nan
            t['Mean Inclusion@100']=t.pop('Recall@100');v2.append(t)
    a=pd.DataFrame(v1);b=pd.DataFrame(v2);a.to_csv(OUT/'table_v1.csv',index=False);b.to_csv(OUT/'table_v2.csv',index=False)
    def table(frame):
        frame=frame.copy()
        for k in frame.columns:
            if k not in ['Method','Status']:frame[k]=frame[k].map(lambda x:'—' if pd.isna(x) else f'{100*x:.3f}')
        return frame.to_markdown(index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5));labels={'Original':(8,8),'Canonical':(8,8),'Set-Mean':(8,8),'Perm-FT_s42':(8,-18),'Perm-FT-cons0.05_s42':(8,10)}
    for _,r in df[df['mode']=='catalog'].iterrows():
        label=r.method.replace('_s42','').replace('Perm-FT-cons0.05','Consistency 0.05\n(unselected)')
        ax.scatter(100*r['VI@20'],100*r['Recall@100']);ax.annotate(label,(100*r['VI@20'],100*r['Recall@100']),xytext=labels.get(r.method,(8,8)),textcoords='offset points',fontsize=9)
    ax.set(xlabel='VI@20 (%) - lower is better',ylabel='Seven-order mean Recall@100 (%)',title='Budget-limited WANDS + BGE screening (seed 42)');ax.set_xlim(-.7,18);ax.set_ylim(61.6,63.8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'screening.png',dpi=180);fig.savefig(OUT/'screening.pdf');plt.close(fig)
    original=df[(df.method=='Original')&(df['mode']=='catalog')].iloc[0]
    statements=[]
    for name in ['Perm-FT_s42','Perm-FT-cons0.05_s42']:
        r=df[(df.method==name)&(df['mode']=='catalog')].iloc[0]
        statements.append(f'- {name}: Recall@100 {100*r["Recall@100"]:.3f}%，VI@20 {100*r["VI@20"]:.3f}%，相對 VI 降幅 {100*(1-r["VI@20"]/original["VI@20"]):.2f}%；Robust@100 {100*r["Robust@100"]:.3f}%，Never@100 {100*r["Never@100"]:.3f}%。')
    costs=[json.loads(l) for l in (HERE/'cost.jsonl').read_text().splitlines()]
    source=json.loads((HERE/'source_audit.json').read_text());checks=[]
    for item in source:
        path=ROOT/item['path'];checks.append({'path':item['path'],'unchanged':sha(path)==item['sha256']})
    assert all(x['unchanged'] for x in checks);dump(HERE/'source_integrity_check.json',checks)
    text=['# PHASE5_REPORT — 預算限定的 Phase V 篩選','',
    '**執行狀態：已達凍結的 4 小時上限，完整實驗矩陣未完成。這是已完成部分的正式交付，不是整個 Phase V 已完成的宣告。**','',
    '## Executive summary','',
    '**A. KEEP CURRENT PAPER STRUCTURE。** 目前已完成的兩個 learned trials 沒有支持明顯更好的 robustness–effectiveness trade-off；不足以升級為 diagnosis + solution。這是依現有證據的保守建議，不是否定尚未完成的 M3 或尚未訓練的 M4。',
    '',*statements,'',
    '兩個 learned trials 均只有 seed 42。Consistency 的 .05 是固定執行順序中的第一個 trial，**不是 best lambda**。λ=.1 僅完成訓練與 C0 embedding；λ=.5 未訓練；M4 未訓練，因此不能提供其 ≥100 商品的 trained-model invariance test。已有合成輸入架構單元測試不替代這項實驗。',
    '', '## Exact setup / Method definitions / Training / Hyperparameters','',
    (HERE/'README.md').read_text(),
    '', '### 實際資料與監督範圍','',pd.read_csv(HERE/'data_coverage.csv').to_markdown(index=False),
    '72 個訓練查詢中 54 個有 Exact，產生 546 組 supervised triples；24 validation 查詢中 17 個有 Exact；384 歷史 held-out 查詢中 308 個有 Exact。產品目錄跨 split 共用，表內保留訓練產品與 held-out targets 的 overlap，不把 product-disjoint generalization 當成本輪結果。',
    '', '## Table V1 — catalog-wide quality and robustness','',
    '數字為百分比；Delta 為百分點。破折號表示缺結果，不表示零。選定 consistency 的必要 row 保留為未完成，另列已完成但未選定的 .05 trial。',
    '',table(a),'', '## Table V2 — target-only inclusion','',table(b),
    '每個 target 是獨立反事實，competitors 固定為同一模型的 C0。Mean Inclusion 不是單一 index 的 Recall。',
    '', '## Statistical uncertainty','',ci[(ci['mode']=='catalog')&(ci.metric.isin(['Recall@100','VI@20','VI@100','Robust@100','Never@100']))].to_markdown(index=False,floatfmt='.3f'),
    '上述為百分點的 paired query bootstrap、10,000 次、seed 20260963；區間未作多重比較校正，作描述性 screening uncertainty，不能逐格挑顯著性。CI 包含零不證明等效。未執行 seeds 43/44，沒有 training-seed mean±std，不能捏造為零變異。',
    '', '## Fully-fitting subset','',pd.read_csv(OUT/'completed_fully_fitting.csv').query("mode == 'catalog'")[['method','queries','pairs','Recall@100','VI@20','Robust@100']].to_markdown(index=False,floatfmt='.5f'),
    '此表為比例值。只限制 targets 為原七順序都 ≤512 tokens 的共同商品，完整目錄不縮減。M2/M3 tokenizer 與文本完全一致。競爭商品仍可能截斷，所以不是全目錄無截斷實驗。M4 field-truncation 與實測 invariance 結果未取得。',
    '', '## Failure cases / Interpretation','',
    '目前不能把 training loss 下降解讀為有效解法。M2 的 VI 下降有限，Recall 點估計較低，Never 增加；stable exclusion 是重要反例。對照 Canonical 與 Set-Mean，應同時看品質與穩定性，不能用單一 VI 宣告成功。',
    '單 epoch、546 對 supervision、無 unaugmented FT control、17 個 eligible validation queries、歷史 test exposure、單 seed 都限制推論。未完成矩陣是執行限制，不能寫成「consistency 或 learned aggregation 全面失敗」。本輪沒有支持 C，也不把簡單已知 training 方法包裝為新穎架構。',
    '', '## Compute / Stop / Reproducibility','',
    f'計入成本合計 {sum(x["seconds"] for x in costs)/3600:.5f} 小時；上限判斷在 batch 邊界觸發。最後中止 stage 以檔案時間估計，可能包含數秒 transition overhead。此為 task wall time，含 CPU／等待與未知中斷的保守計帳，不是純 GPU kernel 小時。',
    '',pd.DataFrame(costs)[['name','seconds']].to_markdown(index=False,floatfmt='.3f'),
    '詳見 INTERRUPTIONS.md、run_initial_interrupted.log、run.log、resume_errors.log。M2 大部分花費在 inference；最初執行時間估計不足，4 小時上限未覆蓋全矩陣。沒有私自延長預算或挑選較好試驗。',
    '新增檔案清單及 SHA256 見 phase5_mitigation/delivery_manifest.json；checkpoint metadata 含 seed 與權重 hash。原稿與既有結果來源全部通過 source_integrity_check.json 的 unchanged 檢查。',
    '重現已完成報告：`phase2/.venv/Scripts/python.exe phase5_mitigation/finalize_bounded.py`。完整實驗命令仍在 README，但目前 cap 已用盡，直接 run 不會重啟計算。任何後續預算必須另記 amendment，不可覆蓋本輪 freeze。報告依賴新增 tabulate==0.9.0。',
    '', '## Implication for WWW Paper','',
    '1. 現有結果增加了一項有用的負面控制：訓練 loss 改善不保證 relevance–visibility trade-off 改善；但不實質支持方法論文定位。',
    '2. 目前不建議新增主文 solution claim。可將 M2 與未選定 .05 trial 作附錄 screening evidence，明示矩陣不完整。',
    '3. 若未來完整確認有必要進主文，可縮短 appearance-first transfer 與重複逐模型數字；保留 target-only、fully-fitting 和候選 loss 的核心證據。',
    '4. 選擇 A：維持 diagnosis paper。沒有足夠證據選 B 或 C；M4 的潛力尚未被本輪測定。',
    '5. 最強可支持句子：**In this budget-limited WANDS screening, permutation fine-tuning and one unselected consistency trial did not establish a better relevance–robustness trade-off; reduced instability must be interpreted alongside stable exclusion.**',
    '', '原稿未修改。後續 Phase VI 跨資料／模型確認目前不建議啟動；應先完成尚缺的 WANDS screening，且須有獨立的計算預算記錄。','']
    (ROOT/'PHASE5_REPORT.md').write_text('\n'.join(text),encoding='utf8')
    dump(HERE/'bounded_completion.json',{'status':'budget_limited_incomplete','recommendation':'A','selected_lambda':None,'completed_trials':['Perm-FT_s42','Perm-FT-cons0.05_s42'],'unavailable':['M3 selected','M4','additional seeds'],'manuscript_changed':False})
    files=[f for f in HERE.rglob('*') if f.is_file() and '__pycache__' not in str(f) and f.name!='delivery_manifest.json']+[ROOT/'PHASE5_REPORT.md']
    dump(HERE/'delivery_manifest.json',[{'path':str(f.relative_to(ROOT)),'sha256':sha(f)} for f in files])
    print('\n'.join(statements),flush=True)
if __name__=='__main__':main()

