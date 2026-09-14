"""Report completed artifacts without selecting from an incomplete lambda grid."""
from screen import *

def main():
    c=setup();test=c['test'];rows=[];ci=[];fitting=[]
    features=pd.read_csv(ROOT/'phase4/results/wands_product_features.csv',dtype={'product_id':str})
    fitids=set(features.loc[features.fully_fits,'product_id'])
    methods=['Original','Canonical','Set-Mean']+[p.name for p in sorted(OUT.iterdir()) if p.is_dir() and (p/'catalog_queries.csv').exists() and p.name not in ['Original','Canonical','Set-Mean']]
    for mode in ['catalog','target']:
        base=pd.read_csv(OUT/'Original'/f'{mode}_queries.csv').set_index('query_id');base=base.loc[base.index.isin(test)]
        for name in methods:
            pq=pd.read_csv(OUT/name/f'{mode}_queries.csv').set_index('query_id');pq=pq.loc[base.index]
            rows.append({'method':name,'mode':mode,'queries':len(pq),**pq.mean().to_dict()})
            for metric in ['Recall@20','Recall@100','VI@20','VI@100','Robust@100','Never@100']:
                mu,lo,hi=bootstrap(pq[metric]-base[metric]);ci.append({'method':name,'mode':mode,'metric':metric,'delta_pp':100*mu,'ci_low_pp':100*lo,'ci_high_pp':100*hi})
            f=pd.read_parquet(OUT/name/f'{mode}_pairs.parquet');f=f[f.query_id.isin(test)&f.product_id.astype(str).isin(fitids)]
            _,fq=metric_frame(f);fitting.append({'method':name,'mode':mode,'queries':len(fq),'pairs':len(f),**fq.mean().to_dict()})
    df=pd.DataFrame(rows);cs=pd.DataFrame(ci);ff=pd.DataFrame(fitting)
    df.to_csv(OUT/'completed_screening_metrics.csv',index=False);cs.to_csv(OUT/'completed_paired_bootstrap.csv',index=False);ff.to_csv(OUT/'completed_fully_fitting.csv',index=False)
    cols=['method','Recall@100','VI@20','VI@100','Robust@100','Never@100']
    def table(mode):
        x=df[df['mode'].eq(mode)][cols].copy()
        for col in cols[1:]:x[col]=x[col]*100
        return x.to_markdown(index=False,floatfmt='.3f')
    planned=['Perm-FT_s42']+[f'Perm-FT-cons{x}_s42' for x in c['lambdas']]+['Set-Attention_s42']
    states=[]
    for name in planned:
        state='完整七順序結果已保存' if name in methods else ('訓練 checkpoint 已保存；完整評估未完成' if (HERE/'checkpoints'/name/'weights.pt').exists() else '尚未完成訓練')
        states.append(f'- {name}: {state}')
    text=['# Phase V 進度報告（非最終結論）','',f'更新：{pd.Timestamp.now(tz="UTC").isoformat()}','',
          '僅彙整已完成的完整七順序產物。M3 lambda grid 尚未完整時，不選 best lambda；表中單一 trial 不是選定方法。所有 test 數據來自既有歷史 held-out，不稱 pristine confirmation。',
          '',*states,'','## Catalog-wide（百分比）','',table('catalog'),'',
          'Recall 是七順序平均 query-macro；Robust 為七順序都入選，Never 為都不入選。Canonical／Set-Mean 的零 VI 為結構性結果，不代表沒有漏失。',
          '','## Target-only（百分比）','',table('target'),'',
          '此表 Recall@100 欄實際意義為 mean target-intervention inclusion@100，不是單一 index recall。每個模型以自己的 C0 competitors 固定，獨立替換每個 target。',
          '','## 相對 Original 的 paired 95% CI（百分點）','',
          cs[(cs['mode']=='catalog')&(cs.metric.isin(['Recall@100','VI@20','Robust@100']))].to_markdown(index=False,floatfmt='.3f'),'',
          '308 個具有 Exact 標籤的歷史 held-out 查詢；query-cluster paired bootstrap 10,000 次。只有 seed 42，未估計訓練 seed 變異；CI 包含零不證明等效。',
          '','## Fully-fitting targets（比例值）','',ff[ff['mode'].eq('catalog')][['method','queries','pairs','Recall@100','VI@20','Robust@100']].to_markdown(index=False,floatfmt='.5f'),'',
          '沿用既有 product_features 的七順序 ≤512-token support；完整 competitors 不移除。僅篩選 target，不能聲稱整個候選目錄都未截斷。',
          '','完整 setup 見 README.md/config.json，執行中斷見 INTERRUPTIONS.md。4 小時計算上限未變，未完成項目不當成方法失敗。原稿未修改。','']
    (HERE/'PROGRESS_REPORT.md').write_text('\n'.join(text),encoding='utf8')
    print(cs[(cs.method=='Perm-FT_s42')&(cs['mode']=='catalog')].to_string(index=False),flush=True)
if __name__=='__main__':main()
