# Phase V 實驗與論文修訂報告

## 結論先行

### 已有的商品可見性改寫是否有用？

在本輪設定中，**沒有可靠證據顯示兩個 adapted E-GEO 改寫器能改善相關商品入選**。WANDS+BGE 的 heuristic 相對 Raw 只增加 0.25 個百分點，95% CI 為 [−9.82, +14.42]；technical prompt 減少 0.47 點，[−3.25, +2.19]。兩者區間都很寬並包含零。ESCI+BGE 則明確下降：heuristic −12.74 點，[−23.49, −4.45]；technical −4.89 點，[−9.43, −1.45]。

這些結果不能解讀為原始 E-GEO 無效。本輪把公開 prompt 適配到 Qwen2.5-0.5B-Instruct，評估 dense first-stage inclusion；原研究使用 GPT-4.1，在固定十個候選中的 LLM reranking 設定。模型、資料、候選階段與目標均不同。

### 改寫是否抵抗屬性排列？

**沒有。未防護的改寫反而擴大排列敏感性。**在 BGE 的 target-only Top-100 實驗中：

| 資料 | 方法 | 平均入選率 | VI@100 | robust coverage |
|---|---:|---:|---:|---:|
| WANDS | Raw | 32.1% | 14.8% | 23.3% |
| WANDS | Heuristic | 32.3% | 31.6% | 19.0% |
| WANDS | Technical | 31.6% | 23.2% | 20.2% |
| ESCI | Raw | 87.5% | 3.0% | 86.4% |
| ESCI | Heuristic | 74.7% | 17.2% | 66.0% |
| ESCI | Technical | 82.6% | 10.0% | 77.1% |

這裡每個最高相關商品都在自己的反事實索引中被替換；query、model、catalog 與所有 competitors 固定為 C0。平均入選率和 VI 是不同結果：WANDS heuristic 的平均值幾乎不變，但跨排列的 crossing 超過 Raw 的兩倍。

### Canonical preprocessing 是否足夠？

Canonical sorting 能讓七個輸入完全相同，因此在 deterministic cache 下 measured VI=0。它是目前最可靠、最便宜的輸入穩健性控制，但不能保證召回無損。

完整目錄的 Recall@100 結果為：

| 資料 | Encoder | Canonical − Raw | 95% CI |
|---|---:|---:|---:|
| WANDS | BGE | +0.07 個百分點 | [−1.50, +1.62] |
| WANDS | MiniLM | −1.83 個百分點 | [−3.17, −0.64] |
| ESCI | BGE | +0.07 個百分點 | [−0.04, +0.18] |
| ESCI | MiniLM | +0.28 個百分點 | [−0.05, +0.78] |

只有 WANDS+MiniLM 有清楚下降；其他三個區間包含零，但這不證明等效。Canonical 應被當成工程 baseline，而不是自動視為無代價解法。

### 控制事實偏移與生成隨機性後，結論是否成立？

嚴格欄位檢查使結論更保守。WANDS 的兩個方法沒有任何輸出通過；ESCI heuristic 的 raw-order 通過率為 7.6%，canonical-input 為 8.1%，technical 為 0%。Technical prompt 有 65.8%–80.3% 的輸出達到 512-token 上限；heuristic 為 9.9%–17.9%。失敗來源包括遺漏或新增數值、單位、顏色，以及沒有完整保留來源中的衝突欄位。這是保守的自動檢查，不是人工 ground truth；但共同七順序均通過的支持只有兩個 ESCI 商品，WANDS 和 technical prompt 都為零，無法支持 fact-preserving effectiveness claim。

Guarded pipeline 對截斷、空輸出或任何事實失敗回退 raw canonical text，不按排名選擇。結果幾乎完全等於 Canonical：WANDS 以及兩資料的 technical prompt 全部回退；ESCI heuristic+BGE 的 guarded mean inclusion@100 為 86.2%，Canonical 為 87.2%，VI=0。MiniLM 分別為 75.3% 與 75.6%，VI=1.0%。因此 guard 能阻止不可信文字進索引，卻沒有帶來超越 canonical 的收益。

獨立 stochastic control 也顯示不能把生成差異全部歸因於輸入排列。WANDS 在相同 C0 或 canonical 輸入的三次生成間，Top-100 crossing 為 16.7%–50.0%；七種順序在各 replicate 內的 VI 同樣為 16.7%–50.0%。ESCI 每格只有兩個商品，僅能視為診斷。主實驗使用 greedy decoding，但 temperature=0 類設定仍只是實作條件，不能證明一般生成穩定性。

## 與既有 Phase I–IV 證據的整合

既有主要發現未被更改：

- 六個一般 encoder×dataset cells 的完整目錄 highest-label pair-micro VI@20 為 4.42%–13.26%。
- Target-only VI@20：WANDS 為 8.84%–12.72%，ESCI 為 3.11%–6.68%。
- `turquoise chair` 的 Exact-relevant 商品，在相同 query、model 和 C0 competitors 下，只改 target attribute order 即從第 5 名降到第 1,527 名。
- 排除 encoder truncation 後，WANDS+BGE 的 fully-fitting Exact 子集仍有 8.69% query-macro VI@20，95% CI [7.15%, 10.39%]。
- WANDS+BGE Top-100 有 8,384 個 highest-label pairs always included、3,856 個 sometimes included、13,230 個 never included。零 VI 可能代表穩定成功，也可能代表穩定漏失。
- Hybrid、centroid/max、set-mean、appearance-first transfer 與 fixed-text reranker 的正負結果全部保留。Reranker 只能調整收到的候選，無法找回 first stage 已刪除的商品。

本輪新主線把問題從「屬性排列會不會影響排名」延伸為「公開 visibility rewriting 在排列干擾下是否穩健」。結果支持一篇 robustness/audit paper，而不是宣稱提出成功的新 optimizer。

## Encoder 與 schema transfer

同一批生成文字直接交給 MiniLM，沒有重新生成或調參。WANDS heuristic 相對 Raw 為 −3.77 點，[−14.36, +6.28]；technical 為 +1.62，[−2.68, +6.37]。ESCI 分別為 +0.72，[−3.78, +5.72] 與 −0.89，[−4.30, +2.49]。沒有一致方向或可靠跨 encoder 改善。既有 appearance-first 規則在 WANDS MiniLM 也下降 1.75 點，[−3.00, −0.63]。這些結果共同說明不能在一個 encoder 或 schema 上挑策略後直接假設轉移。

## 成本與 protocol deviation

主要生成含 5,872 個唯一 cache keys，物化為 6,736 個 task rows；噪音控制另有 342 個唯一生成、384 個 task rows。內容雜湊包含 actual input、完整 prompt、model revision、decoding 與 replicate ID。多次 GPU OOM 後只降低 batch size，所有已完成輸出由 checkpoint 重用；沒有因 rank 或 factuality 選擇較好的輸出。

主要生成累計約 10.1 個 GPU inference hours，noise control 約 1.4 小時，金錢 API 成本為零。這超過 protocol 原定八小時停止目標。完整 frozen matrix 最終被完成；超時期間沒有看 rank、修改 sample、prompt、model、decoding 或 method selection。這仍是 protocol deviation，已寫入論文、provenance 與本報告，不會隱藏。

Technical prompt 的輸出接近 token cap，佔用大部分成本。Pure canonical sorting 不需生成模型，是本輪性價比最好的穩健性控制。若未來要測 GPT-4.1 或更大型本地模型，應另做成本核准與新 confirmation sample，不能把本輪 test set 再當開發資料。

## 論文定位與剩餘 reviewer 風險

目前最合適的主張是：神經 Web retrieval 的 visibility optimization 必須同時評估 relevance、representation robustness、factuality、generation randomness 與 candidate access。商品屬性排列提供乾淨的 raw-fact intervention；free-text rewriting 則暴露 semantic drift 與部署 failure handling。

主要風險如下：

1. Phase V 每資料集只有八個 query clusters，CI 很寬；它適合當受控 audit，不適合宣稱普遍 optimizer ranking。
2. Adapted rewriter 僅 0.5B，負面結果不能推廣到 GPT-4.1 E-GEO。
3. 事實檢查很嚴格且沒有人工驗證；零通過率可能混合真實遺漏與 checker conservatism。
4. ESCI 判斷不完整；target-intervention inclusion 不是單一目錄的 Recall。
5. 沒有 clicks、sales、purchase、agent success 或 user exposure 證據。
6. 八小時停止目標被超過，必須保留為透明的執行偏差。

## 主要檔案

- 方法原文、公開 prompt、版本與適配：`METHOD_SOURCE_AUDIT.md`
- 凍結設計、樣本、estimands 與 contrasts：`OPTIMIZATION_ROBUSTNESS_PROTOCOL.md`
- 所有生成與 hash：`phase5/results/generations.jsonl`
- 主要 summary：`phase5/results/optimization_robustness_summary.csv`
- Paired contrasts：`phase5/results/primary_contrasts.csv`
- 事實與成本：`phase5/results/factuality_cost.csv`
- 隨機生成控制：`phase5/results/generation_noise_order_effect.csv`、`generation_noise_same_input.csv`
- 結果來源映射：`paper_www2027/RESULT_PROVENANCE.md`
- 最終論文：`paper_www2027/main.pdf`
