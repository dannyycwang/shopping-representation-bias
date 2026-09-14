from screen import *

def main():
    c=setup();selection=json.loads((HERE/'selection.json').read_text());test=c['test']
    methods=['Original','Canonical','Set-Mean','Perm-FT_s42',selection['selected'],'Set-Attention_s42']
    base=pd.read_csv(OUT/'Original/catalog_queries.csv').set_index('query_id');base=base.loc[base.index.isin(test)]
    rows=[];cis=[];target=[];fits=[];seedrows=[];states=[]
    features=pd.read_csv(ROOT/'phase4/results/wands_product_features.csv',dtype={'product_id':str})
    fitids=set(features.loc[features.fully_fits,'product_id'])
    for name in methods:
        for mode in ['catalog','target']:
            pq=pd.read_csv(OUT/name/f'{mode}_queries.csv').set_index('query_id');pq=pq.loc[pq.index.isin(test)]
            assert set(pq.index)==set(base.index)
            b=pd.read_csv(OUT/'Original'/f'{mode}_queries.csv').set_index('query_id').loc[pq.index]
            row={'Method':name,'eligible_queries':len(pq),**pq.mean().to_dict()}
            for metric in ['Recall@20','Recall@100','VI@20','VI@100','Robust@100','Never@100']:
                mu,lo,hi=bootstrap(pq[metric]-b[metric]);cis.append({'method':name,'mode':mode,'metric':metric,'delta':mu,'ci_low':lo,'ci_high':hi})
                if metric=='Recall@100':row.update({'Delta Recall@100':mu,'delta_ci_low':lo,'delta_ci_high':hi})
            if mode=='catalog':rows.append(row)
            else:target.append(row)
            f=pd.read_parquet(OUT/name/f'{mode}_pairs.parquet');f=f[f.query_id.isin(test)]
            row['highest_pairs']=len(f);row['unique_products']=f.product_id.nunique()
            ff=f[f.product_id.astype(str).isin(fitids)];_,fq=metric_frame(ff)
            fits.append({'method':name,'mode':mode,'pairs':len(ff),'queries':len(fq),**fq.mean().to_dict()})
            # State transitions use identical pairs, never cherry-picked targets.
            bf=pd.read_parquet(OUT/'Original'/f'{mode}_pairs.parquet').set_index(['query_id','product_id']).loc[pd.MultiIndex.from_frame(f[['query_id','product_id']])]
            actual=f[STEMS].to_numpy()<=100;original=bf[STEMS].to_numpy()<=100
            dr=pd.DataFrame({'query_id':f.query_id.to_numpy(),'rescued':(actual&~original).mean(1),'newly_missed':(~actual&original).mean(1)})
            dr['net']=dr.rescued-dr.newly_missed
            states.append({'method':name,'mode':mode,**dr.groupby('query_id').mean().mean().to_dict()})
    v1=pd.DataFrame(rows);v2=pd.DataFrame(target);ci=pd.DataFrame(cis)
    v1.to_csv(OUT/'table_v1.csv',index=False);v2.to_csv(OUT/'table_v2.csv',index=False);ci.to_csv(OUT/'paired_bootstrap.csv',index=False)
    pd.DataFrame(fits).to_csv(OUT/'fully_fitting.csv',index=False);pd.DataFrame(states).to_csv(OUT/'net_inclusion_changes.csv',index=False)
    for path in sorted(OUT.glob('*_s*/catalog_queries.csv')):
        pq=pd.read_csv(path);pq=pq[pq.query_id.isin(test)];seedrows.append({'method':path.parent.name,**pq.mean(numeric_only=True).to_dict()})
    seeddf=pd.DataFrame(seedrows);seeddf.to_csv(OUT/'all_seed_results.csv',index=False)
    seeddf['family']=seeddf.method.str.replace(r'_s\d+$','',regex=True)
    seeddf.groupby('family')[['Recall@100','VI@20','Robust@100']].agg(['mean','std','count']).to_csv(OUT/'seed_summary.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4.5))
    for i,r in v1.iterrows():
        ax.scatter(100*r['VI@20'],100*r['Recall@100']);ax.annotate(r.Method.replace('_s42',''),(100*r['VI@20'],100*r['Recall@100']),xytext=(5,5+(i%2)*8),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Query-macro VI@20 (%) — lower is better',ylabel='Seven-order mean Recall@100 (%)',title='WANDS + BGE: historical held-out screening');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'screening.png',dpi=180);fig.savefig(OUT/'screening.pdf');plt.close(fig)
    old=ROOT/'phase3/results/phase3_invariant_method/embeddings';fields=json.loads(next(old.glob('wands_unique_attributes_*.json')).read_text())
    costs=[json.loads(x) for x in (HERE/'cost.jsonl').read_text().splitlines()]
    inv=json.loads((OUT/'Set-Attention_s42/invariance.json').read_text())
    def table(df,cols):
        df=df[cols].copy()
        for col in cols:
            if col!='Method':df[col]=df[col].map(lambda v:f'{100*v:.3f}')
        return df.to_markdown(index=False)
    learned=v1[v1.Method.str.contains('_s42')];orig=v1.iloc[0]
    passing=learned[(learned['Recall@100']>=orig['Recall@100'])&(learned['VI@20']<=.5*orig['VI@20'])&(learned['Robust@100']>orig['Robust@100'])]
    promising=learned[(learned['Recall@100']>=orig['Recall@100']-.01)&(learned['VI@20']<=.75*orig['VI@20'])]
    # No novelty or independent confirmation follows from this screen alone.
    recommendation='B. ADD MITIGATION BASELINE ONLY' if len(passing) or len(promising) else 'A. KEEP CURRENT PAPER STRUCTURE'
    lines=['# PHASE V — 小型學習式 mitigation 篩選報告','',f'產生時間：{pd.Timestamp.now(tz="UTC").isoformat()}','',
    '## Executive summary','',f'**建議：{recommendation}。**',
    '本輪允許失敗；成功與否同時看相關商品召回、排列失穩與穩定漏失。以下皆為實際產物重新聚合的數據，不改寫論文、不重跑原有 baseline。',
    '這是既有 WANDS 歷史 held-out 上的小型 screening，不是全新確認集，也不能由單一資料集推廣到其他模型／schema。',
    '', '## Exact setup, training, and hyperparameters','',
    f'固定 catalog 42,994 商品；原 96 開發查詢分成 72 train／24 validation；原 384 held-out 查詢不動，其中 {len(base)} 個有 Exact 標籤，納入 query-macro 評估。共 {c["training_triples"]} 組 Exact/明確 Irrelevant 訓練對。IDs 見 `phase5_mitigation/config.json` 與 `triples.json`。',
    '未找到原有 WANDS BGE 微調訓練流程，因此新增並明確記錄 pairwise softmax ranking loss；不宣稱沿用不存在的原訓練 loss。單 epoch、每 query 最多 16 positives；完整 shared BGE 微調，microbatch 2 / accumulation 8、lr 1e-5、AdamW weight decay .01、溫度 .05、clip norm 1、FP16 autocast、activation checkpointing。',
    f'BGE：`{SPEC["name"]}@{SPEC["revision"]}`；CLS、512 tokens、unit normalization、FP32 scores、stable catalog-index ties、空 query prefix。',
    'M2 每產品使用完整 atoms 的固定 seeded 隨機排列。M3 以同監督 loss 加 lambda × (1−cosine)，第二個正例表示不進負例池。lambda 僅 .05/.1/.5，依 24 validation queries 的 frozen rule 選擇；訓練 dropout 仍啟用，consistency 同時面對 dropout 與排列變化，不能把訓練作用全歸因於 order。',
    f'選定 M3：`{selection["selected"]}`。全部 validation trials 保存在 `selection.json`，不依 test 選參。',
    'M4 為 768→128→1 attention DeepSets，98,561 trainable parameters、凍結 BGE、無位置編碼。保留原 Set-Mean 非屬性文字處理，先 normalize weighted attribute mean，再 .5/.5 interpolation 和 normalize。單 epoch、batch16、lr1e-3。完整 atom embedding cache 重用；沒有生成文字／LLM factuality 問題。',
    'Canonical 是純 raw-atom sorting；不混同舊版增加 field framing 的 M2 template。',
    '', '## Table V1 — catalog-wide screening','',
    '所有數字為百分比；Delta 為百分點。Recall 為七個完整 catalog schedules 的平均，同時保留 C0／每順序明細於 CSV。VI／Robust／Never 在 pair 內跨七個 ranks 計算後做 query macro。Robust = Always，VI = Sometimes；不能用零 VI 代表沒有漏失。',
    '',table(v1,['Method','Recall@20','Recall@100','Delta Recall@100','VI@20','VI@100','Robust@100','Never@100']),
    '', '## Table V2 — independent target-only interventions','',
    '每個 model 固定所有 competitors 為它自己的 C0，只替換一個 target，移除舊 target 再插入；各 target 為獨立 counterfactual。此處的 Recall 欄位在輸出統一命名，但意義是 mean target-intervention inclusion，**不是單一 index 的 Recall**。',
    '',table(v2[v2.Method.isin(['Original','Perm-FT_s42',selection['selected'],'Set-Attention_s42'])],['Method','Recall@100','VI@20','VI@100','Robust@100','Never@100']),
    '', '## Uncertainty and seeds','',
    '10,000 paired query bootstrap draws；保留 seed 20260963。CI 包含零不證明等效／沒有損失。不同訓練 seed 的 std 不是 query sampling uncertainty；只有一個 seed 時 std 未定義。',
    '',ci[(ci['mode']=='catalog')&(ci.metric=='Recall@100')].to_markdown(index=False),
    '', '全部 seed 與 std：`results/all_seed_results.csv`、`results/seed_summary.csv`。只有 validation 達預定門檻且總預算足夠才增加 seeds；未執行的 seed 不當成成功或失敗。沒有 unaugmented fine-tuning control，因此不能唯一歸因於 permutation augmentation。',
    '', '## Fully-fitting subset and field truncation','',
    f'沿用已稽核的 `phase4/results/wands_product_features.csv`；{len(fitids)} 商品在所有七個 flattened inputs 均 ≤512 tokens。M2/M3 的 tokenizer 與輸入不變，所以 fitting support 完全相同；只限制評估 targets，完整 competitors 不移除。',
    '',pd.DataFrame(fits).query("mode == 'catalog'")[['method','pairs','queries','Recall@100','VI@20','Robust@100']].to_markdown(index=False),
    f'原 M4 原子 cache 記錄 individual-attribute truncation fraction = {fields["fraction_truncated"]:.8f}，不宣稱所有超長 fields 完整進入 encoder。非屬性文字另依既有 512-token 處理。',
    '', '## M4 invariance verification','',f'100 商品 × 7 shuffled orders；max L2={inv["l2_max"]:.9g}，mean L2={inv["l2_mean"]:.9g}；max absolute cosine difference={inv["cosine_difference_max"]:.9g}，mean={inv["cosine_difference_mean"]:.9g}。',
    '集合架構在數學上 invariant；浮點聚合有微小誤差。部署／評估使用同一 cache vector，故零 measured VI 是結構性結果，並非七次獨立浮點重新聚合下所有 near-ties 都不變的證明。',
    '', '## Failure cases and interpretation','',
    '詳見同 population 的 `net_inclusion_changes.csv`、各 method rank-level parquet、rank_range、worst_schedule、schedule_spread。有限七順序 worst-observed 不等於所有排列的最坏情況；逐產品 Robust Coverage 與最差完整 catalog schedule 也不是相同量。',
    '任何明顯 Recall 損失或 Never 增加都保留。此單 epoch、546 對小樣本的負結果只否定目前設定，不否定學習式 mitigation 的全部可能。無 unaugmented FT control、少量 validation eligible queries、shared catalog/product identities 與歷史 test exposure 均限制推論。',
    '', '## Compute and reproducibility','',f'GPU-task wall time 合計 {sum(x["seconds"] for x in costs)/3600:.3f} 小時（含 embedding/evaluation 的 CPU 等待，非 profiler kernel 時間）。固定上限 4 小時，逐 batch 檢查。baseline caches 的原始成本未重算。',
    '',pd.DataFrame(costs).to_markdown(index=False),
    'Checkpoint：`phase5_mitigation/checkpoints/*/weights.pt`；metadata 附 SHA256 與 seed。程式／設定／結果新增於 `phase5_mitigation/`，頂層僅新增本報告；舊 phase5、LaTeX、PDF 不改。',
    '重現：`phase2/.venv/Scripts/python.exe phase5_mitigation/test_screen.py`；`phase2/.venv/Scripts/python.exe phase5_mitigation/screen.py freeze`；`... screen.py run`；`... screen.py report`。凍結檔存在時不重新抽樣。',
    '', '## Implication for WWW Paper','',
    f'1. 本輪提供對解法可行性的實際 screening；是否支持擴充，以以上召回／失穩聯合結果為準。推薦 **{recommendation}**，不以單一資料集與簡單 attention pooling 宣稱新穎方法貢獻。',
    '2. 若學習 baseline 有改善，可作為 mitigation 對照；否則保留主文 diagnosis，將失敗篩選放補充材料。',
    '3. 若要騰出空間，可縮短 appearance-first 策略轉移與逐模型重複數字；target-only、fully-fitting、候選不可恢復證據應保留。',
    '4. 在跨資料／跨模型確認與更強對照前，保持 representation-robustness diagnosis 的主要定位，不在本輪改寫稿件。',
    '5. 最保守且可支持的一句話：**Representation-invariant encoding removes measured order sensitivity, but its usefulness must be judged jointly with relevant-product inclusion; small-budget learned mitigation does not by itself establish a general solution.**',
    '']
    (ROOT/'PHASE5_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    manifest=[]
    for path in HERE.rglob('*'):
        if path.is_file() and '__pycache__' not in str(path) and path.name not in ['manifest.json','run.log'] and path.suffix not in ['.npy','.pt']:
            manifest.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
    dump(HERE/'manifest.json',manifest)
    print(recommendation,flush=True);print(v1[['Method','Recall@100','VI@20','Robust@100']].to_string(index=False),flush=True)

if __name__=='__main__':main()
