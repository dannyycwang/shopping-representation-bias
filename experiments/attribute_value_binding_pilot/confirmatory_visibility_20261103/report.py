"""Concise empirical report driven only by completed frozen-sample outputs."""
from core import *
from paper_artifacts import mdtable,cell

def main():
    c=read(HERE/'data/sample_counts.json');a=frame(HERE/'results/rq1_summary.csv');b=frame(HERE/'results/rq2_summary.csv')
    contrasts=frame(HERE/'results/rq1_C_minus_O.csv');conc1=frame(HERE/'results/rq1_query_concentration.csv');conc2=frame(HERE/'results/rq2_query_concentration.csv')
    main1=a[a.stratum.eq('all')];b5=b[b.budget.eq(5)];times=[];coverage=[];lengthrows=[]
    selected_ids=set(frame(HERE/'data/selected_pairs.csv',dtype={'product_id':str}).product_id)
    for model in ['gte','harrier']:
        plan=read(HERE/f'cache_plan_{model}.json');done=read(HERE/f'qa/formal/{model}_complete.json')
        timing_fields=['started_utc','finished_utc','monotonic_elapsed_seconds','wall_elapsed_seconds','awake_elapsed_seconds','possible_suspended_seconds','wall_clock_jump_seconds']
        history=HERE/f'qa/formal_full_matrix_history/{model}_complete.json'
        if history.exists():
            old=read(history);times.append(dict(model=model,stage='first_completed_formal',**{k:old[k] for k in timing_fields}))
        times.append(dict(model=model,stage='final_replay_and_rescore' if history.exists() else 'resumed_formal',**{k:done[k] for k in timing_fields}))
        packed=frame(HERE/f'qa/formal/{model}_packing_audit.csv')
        coverage.append([model,int(packed.eligible.sum()),len(packed),plan['catalog_native_truncated']])
        lengths=frame(HERE/f'data/token_audit_{model}.csv',dtype={'product_id':str})
        lengths=lengths[lengths.product_id.isin(selected_ids)]
        lengthrows.append([model,len(lengths),int(lengths.tokens.min()),int(lengths.tokens.max()),profiles()[model]['max_length'],int(lengths.groupby('product_id').tokens.nunique().gt(1).sum())])
    csv('results/execution_time_summary.csv',times)
    target_ci=b[b.evaluation.eq('target')&b.metric.eq('delta_inclusion')]
    covered=int((target_ci.ci_low.le(0)&target_ci.ci_high.ge(0)).sum())
    headline=f'錯填會改變部分原本 Exact 商品的 Top-20 曝光，但本次兩模型的錯填 crossing 點估計均低於保持配對正確的完整欄位換序。跨模型的 {covered}/{len(target_ci)} 個「方向 × 條件 × 預算」ΔInclusion@20 區間涵蓋 0；本次未得到穩定正向轉移的證據，這也不是效果等於零的等價性證明。'
    r1=[]
    for model in ['gte','harrier']:
        g=main1[main1.model.eq(model)&main1.condition.eq('C')].set_index('metric')
        r1.append(f"{model} 的隨機錯填 crossing 為 {cell(g.loc['crossing'])}%，原始事件 {int(g.loc['crossing'].raw_events)}/{int(g.loc['crossing'].raw_trial_denominator)} 次改動；entry {int(g.loc['entry'].raw_events)} 次、exit {int(g.loc['exit'].raw_events)} 次。")
        r1.append(f"其淨 ΔInclusion@20 為 {cell(g.loc['net'])} pp，少量進出不代表平均曝光改善。")
    observed=bool(main1[main1.condition.eq('C')&main1.metric.eq('crossing')].raw_events.gt(0).any())
    r1.append('在新的確認查詢上，重現了錯填可改變 Top-20 曝光的現象。' if observed else '本次確認樣本未觀察到錯填造成 Top-20 crossing；這不證明所有分數或排名都不受影響。')
    r1.append('前一輪是開發資料、每 pair 一次交換；本輪每 pair 五次取平均，不能把兩輪當作同一批獨立觀察合併，也不以原始事件總數比較效果大小。')
    diffrows=[]
    for r in contrasts.itertuples():
        diffrows.append([r.model,r.metric,f'{100*r.estimate:.2f}',f'[{100*r.ci_low:.2f}, {100*r.ci_high:.2f}]'])
    transfer=[];strong=[]
    for source,target in [('gte','harrier'),('harrier','gte')]:
        for cond in ['C','O']:
            g=b5[b5.source.eq(source)&b5.condition.eq(cond)].set_index(['evaluation','metric'])
            s=g.loc[('source','delta_inclusion')];d=g.loc[('target','delta_inclusion')]
            label='目標模型的區間下界高於 0，支持此設定下的正向平均曝光轉移' if d.ci_low>0 else '目標模型區間上界低於 0，呈現平均曝光下降' if d.ci_high<0 else '目標模型區間涵蓋 0，尚不足以確認正向平均曝光優勢'
            transfer.append(f"{source} → {target}、{cond}、b=5：來源 Δ {cell(s)} pp；目標 Δ {cell(d)} pp。{label}。")
            strong.append(bool(d.ci_low>0))
    type_rows=[]
    for r in a[a.metric.eq('crossing')&a.stratum.ne('all')].itertuples():
        type_rows.append([r.model,r.condition,r.stratum,int(r.pairs),f'{r.raw_events}/{r.raw_trial_denominator}',cell(r)])
    concentration_rows=[]
    for r in conc1[conc1.stratum.eq('all')].itertuples():
        concentration_rows.append(['random',r.model,r.condition,f'{int(r.event_queries)}/{c["confirmation_queries"]}',f'{100*float(r.top5_query_event_share):.1f}%' if pd.notna(r.top5_query_event_share) else 'NA'])
    for r in conc2.itertuples():
        concentration_rows.append(['transfer b=5',r.source+' → '+r.target,r.condition,f'{int(r.event_queries)}/{c["confirmation_queries"]}',f'{100*float(r.top5_query_event_share):.1f}%' if pd.notna(r.top5_query_event_share) else 'NA'])
    # Post-selection composition is descriptive only; retained A remains its own group.
    choices=frame(HERE/'results/rq2_choices.csv',dtype={'product_id':str});sm=frame(HERE/'data/selected_swaps.csv',dtype={'product_id':str}).set_index(['product_id','j'])
    choices['selected_type']=[('retained_A' if r.chosen_variant=='A' else 'yes_no' if sm.loc[(r.product_id,int(r.chosen_variant[1:]))].involves_yes_no else 'not_yes_no') for r in choices.itertuples()]
    typerows=[]
    for keys,g in choices.groupby(['source','target','condition','budget','selected_type']):
        typerows.append(dict(zip(['source','target','condition','budget','selected_type'],keys),pairs=len(g),
            target_entry=int(((~g.target_inclusion_A)&g.target_inclusion).sum()),target_exit=int((g.target_inclusion_A&(~g.target_inclusion)).sum()),
            target_net=int(g.target_net.sum()),interpretation='post-selection composition, not the primary estimand'))
    csv('results/rq2_selected_type_description.csv',typerows)
    t1=(HERE/'paper/table1.md').read_text(encoding='utf8');t2=(HERE/'paper/table2.md').read_text(encoding='utf8')
    ranks=frame(HERE/'results/baseline_rank_distribution.csv')
    rankrows=[[r.model,r.rank_bin,f'{r.numerator}/{r.denominator}',f'{100*r.numerator/r.denominator:.2f}%'] for r in ranks.itertuples()]
    interrupted=read(HERE/'qa/interrupted_attempt_measured_checkpoints.json')
    interrupted_harrier=read(HERE/'qa/interrupted_harrier_replay_checkpoints.json')
    recovered=read(HERE/'qa/interrupted_20261103_final_attempt.json')
    text=['# 確認實驗：商品欄位值錯填、換序對照與跨 encoder 曝光轉移','',
        f"本輪使用 {c['confirmation_queries']} 個與開發集隔離的新查詢、{c['sampled_pairs']} 個 Exact query–product pairs、{c['sampled_products']} 件商品。每商品固定五組配對、11 個完整版本；兩模型均以完整 {c['catalog_products']:,} 商品庫計算獨立 target-only 排名。未修改論文。",'',
        headline,'',
        '## RQ1：新查詢是否重現隨機錯填的曝光影響？','',*r1,'',t1,'',
        '表 1 的事件／trials 是原始五組改動計數；率先在 pair 內平均，再彙整。Entry 分母只含原本 Top-20 外 pairs，exit 只含原本內 pairs，crossing 為全部 pairs。trials 不是獨立樣本。主文優先解讀曝光門檻，分數與排名差的次要分布另存 results/secondary_deltas.csv。','',
        mdtable(['模型','A 原始排名','pairs / 全部','比例'],rankrows),'',
        '原始 rank 分布限制能觀察到的 crossing 機會；遠離第 20 名的改動即使改變排名，也可能不跨越曝光門檻。正式樣本按 hash 抽取，沒有只挑靠近門檻的商品。','',
        '## 與保持正確配對的換序有何不同？','',
        '下表是 C−O 的 paired query bootstrap 差異（百分點）。正 crossing 差表示錯填較常改變門檻狀態，不表示錯填較有利；entry 與 exit 必須分開看。兩條件都改動呈現方式，C−O 不能解讀成純粹的模型內部 binding 效應。','',
        mdtable(['模型','指標','C−O pp','95% CI pp'],diffrows),'',
        '兩模型的 C crossing 都低於 O：GTE 的 paired 差異區間仍涵蓋 0；Harrier 的區間為負，顯示這個樣本中完整換序更常改變曝光門檻。不能由此主張錯填具有較大的曝光優勢，也不能把單一模型、未校正多項比較的區間概括為普遍規律。','',
        '## RQ2：來源模型選出的優勢是否保留？','',*transfer,'',t2,'',
        'A 始終可選，完全同分保留 A、其次取較小 j；所有 pairs 都納入，包括保留 A。來源模型的最高分選擇使來源改善具有選擇偏差，不能單獨視為泛化證據。這是已知同一 query 且可取得來源分數的跨 encoder 測試，不是對未知查詢的轉移。','',
        '![Figure 1: target-model inclusion change by budget](paper/figure1.png)','',
        '圖 1：目標模型 ΔInclusion@20 與 95% query-cluster bootstrap CI。b=1,3,5 使用同一固定清單的巢狀前綴；A 不計入新增嘗試數。各預算、分層及多項差異為探索性比較，未作多重比較校正。完整選擇與來源／目標結果見 rq2_choices.csv、rq2_summary.csv。','',
        '## 是否集中於少數 query 或特定交換類型？','',
        mdtable(['分析','模型／方向','條件','有 crossing 的 query 數','事件最多 5 queries 的占比'],concentration_rows),'',
        '事件集中度有模型差異：隨機 C 的 GTE 事件分散於 32 個 queries，Harrier 僅 16 個；Harrier 的前五個 queries 占 55.2% 事件。b=5 的 GTE → Harrier 錯填轉移只有 3 次 crossing，分別來自 3 個 queries，對少量事件很敏感，不能描述為廣泛轉移。','',
        '集中度按 crossing 的原始事件計數，不能把高占比直接當作少數 query 的因果證明。rq1/rq2_by_query.csv 保留每 query 的分子與 pair 數；query_concentration.csv 另列最大單一 query 占比及逐一移除 query 的淨 inclusion 變化範圍。','',
        mdtable(['模型','條件','交換類型','有此類型的 pairs','原始 crossing/trials','pair-weighted % [95% CI]'],type_rows),'',
        '隨機 C 的 yes/no 原始事件為 GTE 32/65、Harrier 25/29，而這類改動占全部嘗試 3019/4380。Harrier 事件較集中於 yes/no，但 GTE 沒有同樣方向；兩模型並未呈現共同的單一交換類型解釋。','',
        'yes/no 是完整原值等於 yes 或 no 的預先指定分類。兩分層可能涵蓋相同 pair，但各自只平均該 pair 中屬於此類型的交換，因此 pair-weighted 率不一定等於原始 events/trials。分層母體不同，差異僅作描述，不能據此主張某種欄位具有因果弱點。RQ2 選中改動的類型組成另列 rq2_selected_type_description.csv，屬選擇後描述；主要結果仍使用全部 pairs。','',
        '## 支持與不支持的論文主張','',
        ('支持：在隔離的新查詢上，商品記錄干預可造成真實完整 catalog 的 Top-20 進出；效果大小與不確定性如上述表格。' if observed else '目前僅支持此樣本下的曝光估計與區間；未觀察到錯填 crossing，不能宣稱普遍無影響。'),
        ('跨模型正向優勢只可限定於上面區間下界高於零的具體方向、條件及預算；不能概括成普遍可利用的操控策略。' if any(strong) else '目前沒有 b=5 目標模型區間下界高於零的結果；不能把來源端的選分改善包裝成普遍可轉移的曝光優勢。'),
        '不支持：跨未知查詢或跨商品泛化、模型內部 binding 機制已被分離、曝光增加代表商品更相關、或搜尋品質必然惡化。所有 target 原本均為 Exact 商品，qrels 不重新認證被錯填文字。','',
        '適合四頁實證短論文的主張是「完整商品記錄的呈現干預會造成少量曝光變動，保持配對正確的換序同樣重要，而來源選分優勢的跨 encoder 轉移受限」。目前不宜包裝成通用操控策略。若擴大驗證，應另行預先凍結新樣本或模型以縮小轉移區間，保留 O 對照及所有保留 A 的 pairs；不為追求顯著而續加本輪樣本。','',
        '## 設計、相容性、依賴與計時限制','',
        f"使用 seed={SEED} 及獨立 purpose hash；{c['five_pair_excluded_products']} 個池內商品不足五組有效配對、長度排除 {c['length_excluded_products']} 個。沒有按新分數改選樣本。五組交換及所有實際輸入均先凍結。{c['products_repeated_across_confirm_queries']} 件商品跨確認 queries 重複（{c['pairs_on_repeated_products']} pairs）；{c['development_product_overlap']} 件與開發資料商品重疊（{c['pairs_with_development_product_overlap']} pairs）。query bootstrap 保留每 query 全部觀察，但未完全處理跨 query 重複商品及其他資料相關性。",'',
        '使用 10,000 次 paired query bootstrap，同一索引矩陣保留 targets、全部交換與模型條件；以分子／分母比重新估計，不把五組交換視為五倍樣本。報告 95% percentile CI，重點是效果量與事件數，不追加抽樣追求顯著。','',
        'GTE 與 Harrier 都沿用 pinned revision、prefix、pooling、normalization、precision 及 native cap。正式 A/query fresh replay 必須逐 bit 相同；競爭商品使用同一 source baseline，所有反事實獨立替換一個 target。Harrier 的歷史多表示 cache 僅取 C0 catalog slice。source/model/cache 詳細 checksum 在 manifest 與 cache_plan 中。','',
        '數值審查發現 Harrier 的舊 profile 指定完整 308-query 矩陣，而初版 smoke 逐 query 計分會出現浮點差異。在任何正式效果分數計算前已修正，並用原批次向量獨立重算、重新通過 smoke。早期不同計分的 Harrier smoke 保留為歷史 QA，不併入正式結果；樣本與變體沒有更動。GTE 保持前輪 pilot 的 query-row 計分方式。','',
        '最終 Harrier 另以原始全庫 baseline 的全部 13,242,152 個分數逐 bit 認證 256-column 計算區塊，保留完整 308-query 矩陣。164/168 區塊可用，另四區塊中的 23 件選中商品仍以原全庫矩陣計分；五個 smoke pairs 的全部 C/O 對所有 queries 也完全吻合。每次仍只替換一個 target，排名永遠使用完整 42,994 項 baseline array，絕非區塊／子庫排名。由此方法觀察到的每次 crossing 再以原完整矩陣驗分。','',
        '另在 native forward 回傳 CPU 向量後釋放未使用的 GPU allocator cache；沒有更動參數、精度、輸入或 batch。55 個原始 smoke 批次的所有向量逐 bit 一致。兩項執行調整與 gate 證據記錄在 execution_efficiency_amendment.json，並保留調整前 GTE 正式結果作完整 CSV 一致性核對。','',
        mdtable(['模型','不同商品版本數','最短 tokens','最長 tokens','native cap','版本 token 數不同的商品數'],lengthrows),'',
        'Token 數包含實際預處理與 special tokens，逐版本記錄，沒有假設交換必然保持 token 數相同。Harrier 為保留舊計分矩陣形狀而計算的開發 query rows 全數丟棄；選擇與分析只使用 208 個確認 queries。','',
        mdtable(['模型','通過 same-shape packing 的商品','全部商品','baseline native truncation 商品數'],coverage),'',
        '效率調整保持來源 batch 大小及 padded length，以原最長輸入作 guard；所有可用位置的 A 與原 baseline 必須逐 bit 相同。Smoke 的每個 packed C/O 向量亦須逐 bit 等於原方法。不同 token 長度、小 batch 或 A 不吻合時回到原 companions，沒有排除商品。這項計算調整在正式評分前獨立記錄於 execution_amendment.json。','',
        mdtable(['模型','執行段','monotonic 秒','排除休眠的 awake 秒','可能休眠秒','wall-clock jump 秒'],[
            [r['model'],r['stage'],f"{r['monotonic_elapsed_seconds']:.1f}",f"{r['awake_elapsed_seconds']:.1f}",f"{r['possible_suspended_seconds']:.1f}",f"{r['wall_clock_jump_seconds']:.1f}"] for r in times]),'',
        '計時使用 monotonic 與 Windows unbiased interrupt time；awake 時間包含 CPU、I/O、排程及 GPU 等待，不是純 GPU kernel 耗時。每個 forward checkpoint 亦保留建立時的計時及階段；中斷後的 cache 重用不重算為本次新 GPU 工作。','',
        f"上表是完成的正式執行段。數值審查期間曾中斷初次 GTE 段；其 {interrupted['forward_checkpoints']} 個已完成 forward checkpoints 有實測 awake {interrupted['awake_forward_seconds']:.1f} 秒，後續直接重用。該早期中斷段缺少完整 end-to-end 結束計時，不能把這個 checkpoint 合計冒充該段全部活躍時間；未記錄的 setup/I/O overhead 未納入。之後各次執行均保留開始與完成／失敗狀態計時。準備、tokenization、smoke 與統計的獨立計時在 qa/，不與包含它們的上層段重複加總。",'',
        f"啟用通過檢查的效率調整時，也中斷過 Harrier 的 A replay：保留 {interrupted_harrier['forward_checkpoints']} 個 forward checkpoints，實測 awake 合計 {interrupted_harrier['awake_forward_seconds']:.1f} 秒。此數字同樣只涵蓋已完成 checkpoint 的計時，未冒充完整中斷段耗時。GTE 表中的 first_completed_formal 與 final_replay_and_rescore 是兩段實際工作；後者重用向量並核對同一批結果，不能把第一次耗時隱去、只報快速重跑時間。",'',
        f"11 月 4 日接續時，前一天的 Harrier 程序已不在執行；終止時刻與原因沒有被捕捉。該段另保存 {recovered['forward_checkpoints']} 個 forward checkpoints，實測 awake 合計 {recovered['awake_forward_seconds']:.1f} 秒、checkpoint 內可辨識休眠約 {recovered['possible_suspended_forward_seconds']:.1f} 秒。這仍是已完成 forward 的下限，不能把整夜間隔算成活躍運算；逐筆計時與範圍見 qa/interrupted_20261103_final_attempt.json。接續程序僅在工作期間抑制 Windows 自動休眠，手動休眠或關機仍可能中斷。",'',
        '重跑命令、凍結設計與檔案索引見 README.md。paper/ 提供兩張表的 CSV/Markdown/LaTeX，以及圖 1 的 PNG/PDF/SVG；這些是獨立論文素材，未寫入既有 manuscript。','']
    main_text,method_text='\n'.join(text).split('## 設計、相容性、依賴與計時限制',1)
    (HERE/'method_notes.md').write_text('# 設計、相容性、依賴與計時限制'+method_text,encoding='utf8')
    main_text+='## 設計與執行紀錄\n\n'
    main_text+=f'五組交換與完整輸入在新分數計算前凍結。8 件池內商品不足五組有效配對，兩模型共同長度排除為 0。5 件商品跨確認查詢重複，3 件與開發集重疊；不能宣稱跨商品泛化。全部指標使用同一份 10,000 次 query bootstrap，保留每 query 的 targets、交換與模型條件。\n\n'
    main_text+='完成的 GTE 正式執行與重核各約 27.5、9.9 分鐘，Harrier 接續正式段約 120.6 分鐘；另有中斷段的實測 checkpoint 時間。這些是排除休眠的 awake 時間，包含 CPU、I/O 與 GPU 等待，並非純 GPU 耗時。完整 monotonic、休眠、中斷下限及 cache／replay 證據見 [method_notes.md](method_notes.md) 與 [README.md](README.md)。\n'
    (HERE/'report.md').write_text(main_text,encoding='utf8');print('REPORT READY',flush=True)

if __name__=='__main__':main()
