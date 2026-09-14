"""Read completed rank artifacts only; no training or checkpoint selection."""
import screen as s
from screen import pd, json, ROOT, HERE, OUT

def main():
    cfg=s.setup(); methods=['Original','Canonical','Set-Mean','Set-Attention_s42']
    feat=pd.read_csv(ROOT/'phase4/results/wands_product_features.csv',dtype={'product_id':str})
    ids=set(feat.loc[feat.fully_fits,'product_id']); rows=[]; cis=[]
    for mode in ['catalog','target']:
        for fitting in [False,True]:
            frames={}; tag=mode+('_fully_fitting' if fitting else '')
            for name in methods:
                f=pd.read_parquet(OUT/name/f'{mode}_pairs.parquet')
                f=f[f.query_id.isin(cfg['test'])]
                if fitting:f=f[f.product_id.astype(str).isin(ids)]
                _,q=s.metric_frame(f);frames[name]=q
                rows.append(dict(method=name,mode=tag,pairs=len(f),queries=len(q),**q.mean().to_dict()))
            for base in methods[:-1]:
                a=frames[methods[-1]];b=frames[base]
                assert set(a.index)==set(b.index)
                for key in ['Recall@100','Recall@20','VI@20','VI@100','Robust@100','Never@100']:
                    mu,lo,hi=s.bootstrap(a[key]-b.loc[a.index,key])
                    cis.append(dict(comparison='Set-Attention minus '+base,mode=tag,metric=key,delta_pp=100*mu,ci_low_pp=100*lo,ci_high_pp=100*hi))
    df=pd.DataFrame(rows);ci=pd.DataFrame(cis)
    df.to_csv(OUT/'set_attention_summary.csv',index=False);ci.to_csv(OUT/'set_attention_paired_ci.csv',index=False)
    dev={x:s.mean_metrics(x) for x in methods}
    gate=dev[methods[-1]]['Recall@100']>dev['Set-Mean']['Recall@100']
    assert not gate,'Unexpected development gate: investigate, do not auto-select on test.'
    s.dump(OUT/'set_attention_seed_gate.json',{'validation':dev,'improves_over_set_mean':gate,'additional_seeds':[], 'rule':'Frozen validation gate; no test-based expansion.'})
    inv=json.loads((OUT/methods[-1]/'invariance.json').read_text())
    def table(tag):
        x=df[df['mode']==tag][['method','pairs','queries','Recall@100','VI@20','Robust@100','Never@100','worst_schedule@100']].copy()
        for col in x.columns[3:]:x[col]*=100
        if tag.startswith('target'):x=x.rename(columns={'Recall@100':'Inclusion@100'})
        return x.to_markdown(index=False,floatfmt='.3f')
    text=['# Set-attention 最終篩選報告','',
      '**結論：排列不變性成立，但目前沒有足夠證據支持新增 learned solution 主張。**',
      '這是原 raw serializer 的獨立 track；不能與 labeled PI-FT track 視為完全 matched 比較。保留全 42,994 商品、歷史 held-out queries，單 seed42。',
      '## 全體 catalog（百分比）','',table('catalog'),
      '## Target-only（百分比）','',table('target'),
      'Target competitors 為各方法自己的 C0。Invariant 方法的七版本相同，其 target 表重用固定表示索引排名；這不是只更換 target、所有方法都使用 Original competitors 的跨方法介入。Inclusion 不能稱為一個混合索引的 Recall。',
      '## Paired query bootstrap（百分點）','',ci.to_markdown(index=False,floatfmt='.3f'),
      '10,000 draws，seed20260963，描述性未校正 95% CI，不含 training-seed uncertainty。CI 包含零不證明等效。',
      '## 共同 fully-fitting raw targets','',table('catalog_fully_fitting'),'',table('target_fully_fitting'),
      '使用 phase4/results/wands_product_features.csv 的七原始順序 fully_fits 定義；所有方法限制相同 targets，competitors 仍完整。這個條件子集不保證各 set field 完全不截斷，也不提供截斷的因果分解。',
      '## 訓練、不變性與 seed gate','',
      'Frozen BGE；768→128→1 tanh scorer，98,561 trainable parameters；無 positional input。獨立 attribute embeddings 加權平均後先 normalize，再與非屬性向量各半混合並 normalize。546 triples，一 epoch，batch16，AdamW lr0.001；沒有 architecture search。',
      f'實際 {inv["products"]} 商品 × {inv["shuffles"]} 次重排：max L2={inv["l2_max"]:.9g}，max cosine difference={inv["cosine_difference_max"]:.9g}。七 schedules 共用同一 cached vector，因此 VI=0 是架構/cache 的結果；浮點微差仍可能影響極近 ties。',
      'Validation 的 Original/Set-Mean/Set-attention Recall@100 分別為 '+ '/'.join(f'{dev[x]["Recall@100"]*100:.3f}%' for x in ['Original','Set-Mean','Set-Attention_s42'])+'。Set-attention 未超過 Set-Mean，不符合原凍結的額外 seed gate，故不執行 seeds43/44。這個 gate 已先在 development 檢查，未依 test 決定。',
      '## 成本與限制','',
      '本次 Set-attention 訓練及評估計帳 16.391 秒，重用已快取的 frozen field embeddings；不含先前產生 embeddings 的成本或 torch import。不能據此宣稱完整系統只需 16 秒。既有 unique attribute cache 的 field 截斷率約 0.0085%（129,151 unique fields）。',
      'Set-attention 全體 Recall 點估計只略高於 Set-Mean，低於 Original 與 canonical sorting。穩定漏失仍存在；零 VI 本身不是品質改善。Validation 未通過追加 seed gate；保留單 seed、historical test 與 query-bootstrap 的限制。',
      '重現分析：`phase2/.venv/Scripts/python.exe phase5_mitigation/report_set_attention.py`。排名/checkpoint/invariance.json 保存在 results/Set-Attention_s42 與 checkpoints/Set-Attention_s42。未修改論文。','']
    (HERE/'SET_ATTENTION_REPORT.md').write_text('\n\n'.join(text),encoding='utf8')
    print(df[df['mode']=='catalog'][['method','Recall@100','VI@20']].to_string(index=False))
    print(ci[(ci['mode']=='catalog') & (ci.metric=='Recall@100')].to_string(index=False))

if __name__=='__main__':main()
