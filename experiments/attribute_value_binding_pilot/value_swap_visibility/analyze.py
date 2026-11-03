"""Descriptive full-catalog target-only visibility summaries and auditable cases."""
from common import *

def summarize(g,scope):
    n=len(g);inside=int(g.in_top20_A.sum());outside=n-inside
    entry=int(g.top20_entry.sum());leave=int(g.top20_exit.sum())
    row=dict(scope=scope,model=SPEC['key'],queries=g.query_id.nunique(),pairs=n,products=g.product_id.nunique(),
        **quantile(g.score_delta,'score_delta'),**quantile(g.rank_delta,'rank_delta'),
        entry_n=entry,entry_den=outside,entry_rate=ratio(entry,outside),
        exit_n=leave,exit_den=inside,exit_rate=ratio(leave,inside),
        crossing_n=entry+leave,crossing_den=n,crossing_rate=ratio(entry+leave,n))
    for name in ['improved','declined','unchanged']:
        num=int(g.direction.eq(name).sum());row.update({name+'_n':num,name+'_den':n,name+'_rate':ratio(num,n)})
    return row

def fmt(n,d):return f'{n}/{d} ({100*n/d:.2f}%)' if d else f'{n}/{d} (NA)'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
        ['| '+' | '.join(str(x).replace('|',' / ').replace('\n',' ') for x in row)+' |' for row in rows])

def main():
    assert read(HERE/'qa/pilot/complete.json')['passed'];freeze_check()
    f=frame(HERE/'results/pilot/pairs.csv',dtype={'product_id':str})
    f['rank_bin_A']=pd.cut(f.rank_A,[0,20,100,1000,np.inf],labels=['1-20','21-100','101-1000','1001+']).astype(str)
    csv('results/pilot/pairs_with_rank_bins.csv',f)
    summary=[summarize(f,'all_sampled_pairs')]
    for b in ['1-20','21-100','101-1000','1001+']:
        g=f[f.rank_bin_A.eq(b)]
        if len(g):summary.append(summarize(g,'descriptive_rank_A_'+b))
    csv('results/pilot/summary.csv',summary)
    distribution=[dict(rank_bin=b,numerator=int(f.rank_bin_A.eq(b).sum()),denominator=len(f),
        proportion=float(f.rank_bin_A.eq(b).mean())) for b in ['1-20','21-100','101-1000','1001+']]
    csv('results/pilot/baseline_rank_distribution.csv',distribution)
    cases=[]
    for label,subset in [('top20_entry',f[f.top20_entry]),('top20_exit',f[f.top20_exit])]:
        for r in subset.to_dict('records'):cases.append(dict(case_kind=label,case_selection='all observed events',**r))
    for label,mask in [('extreme_rank_improvement',f.rank_delta.eq(f.rank_delta.min())&f.rank_delta.lt(0)),
                        ('extreme_rank_decline',f.rank_delta.eq(f.rank_delta.max())&f.rank_delta.gt(0))]:
        g=f[mask].copy()
        if len(g):
            g['case_hash']=[pick('extreme_tie',r.query_id,r.product_id) for r in g.itertuples()]
            cases.append(dict(case_kind=label,case_selection='extreme rank_delta; fixed hash tie-break',**g.sort_values('case_hash').iloc[0].to_dict()))
    normal=f.copy();normal['case_hash']=[pick('ordinary_case',r.query_id,r.product_id) for r in normal.itertuples()]
    for r in normal.sort_values('case_hash').head(5).to_dict('records'):
        cases.append(dict(case_kind='ordinary_hash_sample',case_selection='5 smallest fixed hashes of all sampled pairs; no outcome filter',**r))
    csv('results/pilot/cases.csv',cases)
    # Complete original and corrupted text for every selected example.
    inputs=pd.read_parquet(HERE/'data/full_inputs.parquet');caseids={r['product_id'] for r in cases}
    csv('results/pilot/case_full_inputs.csv',inputs[inputs.product_id.isin(caseids)])
    counts=read(HERE/'data/sample_counts.json');cfg=read(HERE/'config.json');s=summary[0]
    nq=f.groupby('query_id').size().value_counts().sort_index()
    csv('results/pilot/targets_per_query_distribution.csv',[
        dict(targets_per_query=int(k),queries=int(v),pairs=int(k*v)) for k,v in nq.items()])
    exclusions=frame(HERE/'data/excluded_products.csv',dtype={'product_id':str})
    ex=exclusions.groupby(['stage','reason']).size().reset_index(name='products')
    csv('results/pilot/exclusion_summary.csv',ex)
    _,_,j=load();poolq=frame(HERE/'data/query_selection.csv').query_id
    excluded_exact=j[j.label.eq('Exact')&j.query_id.isin(poolq)].merge(exclusions,on='product_id',how='inner')
    csv('results/pilot/excluded_exact_pairs.csv',excluded_exact)
    assert len(excluded_exact)==counts['exact_pairs']-counts['eligible_exact_pairs']
    replay=frame(HERE/'qa/pilot/fresh_replay.csv')
    hist=frame(HERE/'qa/pilot/historical_rank_comparison.csv')
    rankhist=int(hist.rank_difference.ne(0).sum())
    rebuilt=(HERE/'baseline_manifest.json').exists()
    if rebuilt:
        bm=read(HERE/'baseline_manifest.json');oldgate=read(HERE/'qa/historical_cache_replay/replay_gate.json')
        cache_note=(f"歷史 cache 的來源 checksum、順序、模型及 profile 符合，但最初 smoke 的 5 個 A 與 5 個 query 向量均未 bitwise 重現，最大分量差 {oldgate['maximum_vector_error']:.7g}。sidecar 缺完整套件版本，不能把差異確定歸因於某一項 runtime 設定。重複執行目前環境的 10 個 batches 全部 bitwise 相同後，僅以既有 DenseEncoder 重建 source-order 全庫與原始 queries，耗時 {bm['seconds']/60:.2f} 分鐘。舊 cache 保留，但未用於主要評分；新 manifest 綁定原凍結抽樣與精確 profile。")
        history_note='下列與歷史結果的差異同時包含 cache 重建的數值差異及 matmul shape 差異，屬相容性紀錄，不是錯填效應。'
    else:
        cache_note='既有全庫及 query cache 的來源 checksum、商品／query 順序及 sidecar 均核對，無須重建。'
        history_note='下列歷史結果差異反映 FP32 matmul shape 差異。'
    examples=[]
    for kind in ['top20_entry','top20_exit','extreme_rank_improvement','extreme_rank_decline']:
        matches=[r for r in cases if r['case_kind']==kind]
        if matches:examples.append(min(matches,key=lambda r:pick('report_case',kind,r['query_id'],r['product_id'])))
    examples += [r for r in cases if r['case_kind']=='ordinary_hash_sample']
    conclusion=f"觀察到排名改善 {fmt(s['improved_n'],s['improved_den'])}、下降 {fmt(s['declined_n'],s['declined_den'])}、不變 {fmt(s['unchanged_n'],s['unchanged_den'])}。這是本次抽樣的描述性比例；有些改善不能推論錯填通常有利。"
    crossing_text=f"觀察到 Top-20 entry {fmt(s['entry_n'],s['entry_den'])}、exit {fmt(s['exit_n'],s['exit_den'])}；crossing 為 {fmt(s['crossing_n'],s['crossing_den'])}。"
    lines=['# 商品欄位值錯填對能見度的影響','',
        '研究對象是同一件實體商品的記錄遭到刻意錯填；不假定文字語意不變。唯一處理是交換兩個完整原始 values（C），對照原記錄（A）。未做最小描述二選一、反事實 query、換序條件、reranking、訓練或論文修改。先前 round2_controlled 已依使用者要求取消，留存檔案未用於此實驗。','',
        '## 1. 是否觀察到排名改善或下降？','',conclusion,'',
        table(['指標','Q1','中位數','Q3'],[
            ['score_C − score_A',*[f"{s['score_delta_'+q]:.7f}" for q in ['q25','median','q75']]],
            ['rank_C − rank_A',*[f"{s['rank_delta_'+q]:g}" for q in ['q25','median','q75']]]]),'',
        'rank_delta < 0 代表排名改善，> 0 代表下降；正 score_delta 代表分數提高。不同 query 的分數差僅在各自 A/C 內比較後作描述性彙整。','',
        '## 2–3. Top-20 進出及本次抽樣比例','',crossing_text,'',
        'Entry 分母只計原本 Top-20 外的 pairs；exit 分母只計原本 Top-20 內的 pairs；crossing 分母為全部 pairs。零分母為 NA。','',
        table(['事件','numerator/denominator'],[[x,fmt(s[n],s[d])] for x,n,d in [
            ('排名改善','improved_n','improved_den'),('排名下降','declined_n','declined_den'),('排名不變','unchanged_n','unchanged_den'),
            ('Top-20 entry','entry_n','entry_den'),('Top-20 exit','exit_n','exit_den'),('Top-20 crossing','crossing_n','crossing_den')]]),'',
        '## 4. 抽樣、原始排名及限制','',
        f"實際核對 WANDS catalog 為 {counts['catalog_products']:,} 件商品、{counts['source_queries']} 個來源 queries；既有 held-out split 與有 Exact qrels 的交集為 {counts['eligible_evaluation_queries']} 個 queries、{counts['exact_pairs']:,} 個 Exact pairs。清理方式沿用既有程式：重複同標籤合併、{counts['contradictory_qrel_pairs_removed']} 個衝突標籤 pairs 排除。",'',
        f"先對每個商品以 seed={SEED} 及獨立 purpose 字串選一組欄位，再檢查全文 A/C 長度，最後 hash 抽取最多 100 queries、每 query 最多 5 個 Exact targets。實際抽中 {counts['sampled_queries']} queries、{counts['sampled_pairs']} pairs、{counts['sampled_distinct_products']} 個不同商品；不足 500 是部分已抽中 query 的 eligible Exact targets 少於 5，未以其他 query 補滿。相同商品的 C 向量按商品 ID/輸入與 profile 重用；不以新分數、原始排名或是否接近 Top-20 選樣。",'',
        f"{counts['parse_eligible_products']:,} 件商品通過通用解析與值條件；排除 {counts['excluded_parse_products']} 件（{counts['excluded_parse_products']}/{counts['catalog_products']}）。可用 Exact pairs 為 {counts['eligible_exact_pairs']:,}。欄位沒有顏色／尺寸或同類型白名單；query 不必提及交換欄位。缺值占位符清單在 config.json；yes/no/0 均保留。選定 key 以 casefold/空白正規化判定身份唯一，其他未選 key 的重複不排除整件商品。索引與原始字串均保留。",'',
        f"長度排除 {counts['excluded_length_products']}/{counts['parse_eligible_products']}；GTE cap={SPEC['max_tokens']}，實際全文含 special tokens，prefix 為空。全庫原始輸入最長 {int(frame(HERE/'data/catalog_token_counts.csv').tokens_A.max()):,} tokens，baseline native truncation 實測 {counts['baseline_native_truncated_products']}/{counts['catalog_products']}。本次抽中 A/C 為 {int(f.tokens_A.min())}–{int(f.tokens_A.max())} tokens，各自實測後長度不同的 pairs 為 {int(f.tokens_A.ne(f.tokens_C).sum())}/{len(f)}，未事先假定 token 數相同。沒有縮短 title/description 或轉成最小描述。",'',
        table(['原始 rank_A','pairs / 全部'],[[r['rank_bin'],fmt(r['numerator'],r['denominator'])] for r in distribution]),'',
        f"原本有 {int(f.in_top20_A.sum())}/{len(f)} pairs 在 Top-20，另有 {int(f.rank_A.between(21,100).sum())}/{len(f)} 在 21–100，並非幾乎全部遠離 Top-20。跨越門檻事件仍僅 {s['crossing_n']}/{len(f)}；曝光進出與一般排名變動應分開看。極端名次變動也可能發生在很後面的順位，不能等同於取得 Top-20 曝光。",'',
        '主要結果包含全部抽樣 pairs。summary.csv 的 rank 區間列僅為原始排名分層描述，未取代整體結果。每 query 可有多個相關 targets，且資料中的商品／類別可能相關；本輪不做把 pairs 視為獨立觀察的顯著性檢定。只用一個 encoder、一次固定欄位交換，不能推論其他模型或多次交換。', '',
        '原始 Exact qrels 的用途是追蹤原本相關的同一商品，因錯填記錄而得到或失去曝光；它不認證錯填文字仍符合 query 語意。排名下降不一律等於模型失敗，排名改善也不等於商品更相關。', '',
        '## 5. 是否值得擴大？','',
        '本次能提供真實完整 catalog 上的可量測變化，適合下一步以預先固定的新 seed／交換組重複，並在資源允許時增加另一個相容 encoder。應分開估計商品、query 與交換組的變異，保留全部樣本及 Top-20 正反向事件。擴大前不應把本次比例解讀為錯填的一般方向或商業效果。','',
        '## 可核對案例','',
        'cases.csv 收錄全部實際 Top-20 entry/exit、最大排名改善與最大下降（明確標為極端案例），以及全部 pairs 中 hash 最小的 5 個普通案例。極值同分以固定 hash 決定；普通案例完全不按結果篩選。下表在 entry/exit 各呈現固定 hash 的一例，其餘事件均可於 CSV 查閱。', '',
        table(['案例類型','query_id / product_id','完整 query','交換欄位與原值（C 對調）','score A→C','rank A→C'],[
            [r['case_kind'],f"{r['query_id']} / {r['product_id']}",r['query'],
             f"{r['key1']}:{r['value1_A']} ↔ {r['key2']}:{r['value2_A']}",
             f"{r['score_A']:.7f} → {r['score_C']:.7f}",f"{r['rank_A']} → {r['rank_C']}"] for r in examples]),'',
        '未出現的案例類型：'+(', '.join(k for k in ['top20_entry','top20_exit','extreme_rank_improvement','extreme_rank_decline'] if not any(r['case_kind']==k for r in cases)) or '無；上述類型均有實例。'),'',
        '## 模型、cache 與檢查','',
        f"模型：{SPEC['name']}，revision `{SPEC['revision']}`。沿用 phase2 DenseEncoder：空 query/document prefix、CLS、FP16 model、FP32 pooling/L2 normalization、SDPA、native cap {SPEC['max_tokens']}。使用既有 phase2/.venv（transformers {cfg['runtime']['transformers']}、torch {cfg['runtime']['torch']}），完整 runtime 與模型 snapshot checksum 在 config.json。",'',
        cache_note,'',
        f"主要評分前，以原始 length-sorted batch 及 companions fresh replay 所有抽中 A 和 queries，對使用中的 baseline 最大向量差 {replay.max_abs_error.max():g}，全部 bitwise 相同。C 也由原 batch 僅替換該 target 的文字編碼，未混用其他設定 embeddings。",'',
        f"每個 pair 都獨立把 target embedding 放回同一完整 catalog index，使用相同 shape 的 FP32 matmul；所有競爭商品分數 bitwise 不變，再以沿用的 target-only 公式排名。Smoke 先跑 5 pairs 並以完整 score array 排序核對 A/C；pilot 全部另以獨立計數公式核對。相同 query 下的 targets 從未同時錯填；結果不是一次 catalog-wide intervention。",'',
        f"{history_note}歷史結果採全 query-matrix matmul，本次為一個 query × full catalog 的同形 A/C matmul。與歷史逐筆 A scores 的最大差為 {hist.score_error.abs().max():.3g}；抽中 A ranks 差異 {rankhist}/{len(hist)}。詳見 qa/pilot/historical_rank_comparison.csv，A/C 主要比較使用完全一致的 baseline 及運算形式。",'',
        '時間紀錄限制：首次正式 pilot 的 time.time() 累計跨越 2026-10-30 至 2026-11-03，stdout 在 C 編碼期間出現大幅牆鐘時間跳躍。該次 complete.json 的 seconds 包含跨日間隔，不能當成純 GPU 計算耗時或成本估計；實際活躍 GPU 時間未獨立量測。原始時間觀察保存於 qa/timing_observation.json。','',
        '重跑命令與所有檔案索引見 README.md。selection_manifest.json 在計算新分數前凍結；完整實際 A/C 在 data/full_inputs.csv/.parquet。data/ 包含母體、排除、長度與抽樣清單；results/pilot/ 包含逐筆、summary、rank distribution 與案例；qa/ 保留 replay、排名及第一輪／論文未變檢查。','']
    if s['crossing_n']==0:
        lines.insert(lines.index('## 4. 抽樣、原始排名及限制'),'本次沒有 crossing；須結合下方原始排名分布解讀，不能直接說錯填不影響檢索。\n')
    (HERE/'report.md').write_text('\n'.join(lines),encoding='utf8')
    print('WROTE descriptive report',canonical(s),flush=True)

if __name__=='__main__':main()
