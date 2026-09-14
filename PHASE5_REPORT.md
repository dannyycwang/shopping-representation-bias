# Phase V — 最終優先實驗交付

**A. KEEP CURRENT PAPER STRUCTURE。** PI-FT 適配與 Set-attention 已完成。現有證據尚不足以升級為強 learned-solution 論文；保留診斷與簡單 invariant controls 主線。本輪未修改 manuscript。

## PI-FT：matched labeled representation track

| 方法 | Recall@100 | VI@20 | Always@100 | Never@100 |
|---|---:|---:|---:|---:|
| Labeled-ZeroShot |63.848%|10.590%|58.842%|31.183%|
| Standard-FT |62.394%|6.667%|59.369%|34.185%|
| Adapted-PI-FT |62.272%|11.565%|56.686%|32.146%|

七個 catalog schedules、308 Exact-eligible historical held-out queries、21,299 pairs、完整 42,994 商品。Recall 是平均入選；Always/VI/Never 分別為七版本全部、部分、均未入選。

PI 相對 Standard Recall 差 −0.122 百分點（95% CI −1.086 至 +0.829），VI 增加 4.899（+3.527 至 +6.370），Always 減少 2.683。Standard 降低 VI，但相對 zero-shot 的 Recall 點估計也較低，不是全面成功。

**正面子集保留：**共同 fully-fitting 6,473 pairs / 232 queries 的 PI/Standard/ZeroShot Recall 為 62.885% / 60.861% / 57.506%。PI 對 Standard 提升 2.024 百分點，描述性 CI 下界僅略高於零；但 VI 仍為 10.626% 對 6.278%。這不是全體成功，也不是截斷的因果分解。

Standard-FT 是同樣 supervised contrastive training 關閉 augmentation 的 control，不是新演算法。PI 使用 pinned author renderer、每次重排及 15% dropout；實測 15.0006%，title 保護違規零。只保護 title、不推測 facet。WANDS/BGE、407 training pairs、label-masked negatives、未縮短欄位等都是適配，不能称 faithful reproduction 或反駁作者原 benchmark 結論。

詳細 target-only、nDCG/cNDCG、來源、成本與 CI：[PI_FT_REPORT.md](phase5_pift/PI_FT_REPORT.md)。

## Set-attention：原 raw representation track

| 方法 | Recall@100 | VI@20 | Always@100 | Never@100 |
|---|---:|---:|---:|---:|
| Original |63.620%|11.932%|57.180%|30.515%|
| Pure canonical sort |63.237%|0%|63.237%|36.763%|
| Set-Mean |61.820%|0%|61.820%|38.180%|
| Set-attention |62.173%|0%|62.173%|37.827%|

Set-attention 對 Set-Mean 改善 **0.353 百分點，95% CI +0.066 至 +0.779**。這個小幅正面結果保留。但對 Original 為 −1.447（−3.014 至 +0.137），對 canonical 為 −1.064（−2.643 至 +0.577）。CI 包含零不等於等效。

實際 100 商品各重排七次，max L2 約 1.22e−7；七 schedules 重用 invariant cached vector，零 VI 是結構性結果，仍須考慮穩定漏失。Validation Recall：Set-attention 79.584%、Set-Mean 79.815%；未通過預先規定的優於 Set-Mean gate，不追加 seeds43/44。沒有依 test 調參或擴充。

詳細報告：[SET_ATTENTION_REPORT.md](phase5_mitigation/SET_ATTENTION_REPORT.md)。兩個 representation/training tracks 不完全 matched，不可混成單一排名。Target-only 的 competitors 固定為各方法自己的 C0，不是跨方法全部使用 Original competitors；inclusion 不是單一混合 index 的 Recall。

## 完成界線、成本與負面結果

三列 PI matched comparison、Set-attention seed42、actual invariance 與分析均完成。PI 計帳 12,902.61 秒（3.58 小時，含已記錄失敗 stages）。Set-attention 增量 16.391 秒，重用既有 field embeddings，不含原 embedding 或 torch import 成本。

舊 permutation-only Recall/VI：62.771%/10.487%；consistency λ=.05：63.085%/12.737%。負面及有限結果保留。λ=.1 訓練與 embeddings 保留至 C2s4；λ=.5 未訓練。因使用者改優先執行 PI-FT/Set-attention 而 deferred，不是實驗失敗；未選出 best lambda。舊 4-hour 報告保存在 phase5_mitigation/archive/budget4h/PHASE5_REPORT.md。

## 限制與重現

單資料集、單 retriever、小規模訓練、單 training seed；historical held-out 不是 pristine confirmation。10,000 draws paired query bootstrap 是描述性未校正 CI，不含 seed uncertainty，不逐格挑顯著性。不能從 loss 或零 VI 推論購買、點擊、fairness 效果。

分析命令（不重跑 GPU）：

```powershell
& phase2/.venv/Scripts/python.exe phase5_pift/analyze.py
& phase2/.venv/Scripts/python.exe phase5_mitigation/report_set_attention.py
& phase2/.venv/Scripts/python.exe phase5_pift/finalize_delivery.py
```

來源、config、IDs、augmentation traces、checkpoints、ranks 保留在對應資料夾。FINAL_DELIVERY_MANIFEST.json 核對原凍結 artifacts（包含 manuscript）與 checkpoint hashes，記錄本輪交付檔案。未新增付費 API 或修改論文。
