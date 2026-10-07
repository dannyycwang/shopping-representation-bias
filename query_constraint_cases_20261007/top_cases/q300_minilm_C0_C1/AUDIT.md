# Query 300: animal print handmade tufted wool beige/black area rug by allmodern

最終等級 **B**，限 `wool` / `whole` 子條件；minilm_C0_C1，catalog-wide，K=20。原始 query 不變。完整集合包括所有 qrels；沒有修改官方 relevance label。

## 來源逐件稽核

逐件讀取 25 件聯集的所有來源。11497、22340、25537、27973 有明確 wool 欄位而退出；6201 以 polypropylene 為材質而进入。24511/25569/32608 的 wool 進入也完整计入，40603 的 wool + viscose 是 AMBIGUOUS。15534/15535 保留的混紡亦為 AMBIGUOUS。

關鍵修正：退出的 15536 雖只有 `material : wool`，完整描述說 Thornbury collection 許多設計帶 viscose。依不能忽略 collection/variant 衝突的規則降為 AMBIGUOUS。因此最初只靠欄位/局部文字的 18→16、界限 [-2,-1] 不能作為最終結果；最終 17→16、界限 [-2,0]，降為 B。未改選其他 query 補出 A 級結果。

整個前後 Top-20 全是 Partial，H=2 而 Exact recall 為 0/2。兩個 20 維 gain 序列逐位相同，所以 nDCG 真正相等，並非四捨五入或容差判等。這仍只是一個 wool 子條件案例：beige/black、多色，以及 animal print、handmade、tufted、AllModern 等限制未完全核定，不能稱為整句 query 滿足數。三個入選案例均有截斷，這例不是 fully-fitting 的替代例子。

## 全部條件與集合（M/C/U/A）

M=MATCH、C=CONTRADICTION、U=UNKNOWN、A=AMBIGUOUS；U/A 不當作不符合。joint 額外包含 product type、用途、所有其他文字限制及部位縮限。

| 子條件 | 範圍 | C0 M/C/U/A | C1 M/C/U/A | tight Δ |
| --- | --- | --- | --- | --- |
| wool | whole | 17/0/0/3 | 16/1/0/3 | [-2, 0] |
| beige/black | whole | 3/0/0/17 | 2/0/0/18 | [-5, 4] |

| 集合 | C0 M/C/U/A | C1 M/C/U/A | L M/C/U/A | G M/C/U/A | loose Δ | tight Δ | 消失已證明 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full | 17/0/0/3 | 16/1/0/3 | 4/0/0/1 | 3/1/0/1 | [-4, 2] | [-2, 0] | False |
| exact | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | [0, 0] | [0, 0] | False |
| joint | 0/0/2/18 | 0/1/1/18 | 0/0/1/4 | 0/1/0/4 | [-20, 19] | [-5, 4] | False |

## 完整進出清單

| ID | 名稱 | qrel | 進出 | 歷史 C0→C1 | 條件 | 原始對應欄位 | 類型 | 用途 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11497 | lyly animal print tufted wool beige/black area rug | Partial | lost | 2→23 | MATCH | material : wool | MATCH | {} |
| 15536 | thornbury animal print tufted wool gold/black area rug | Partial | lost | 18→22 | AMBIGUOUS | material : wool | MATCH | {} |
| 22340 | tampico hand-tufted wool ivory/black area rug | Partial | lost | 11→195 | MATCH | material : wool | MATCH | {} |
| 24511 | wilfredo abstract handloomed wool pink area rug | Partial | gained | 62→15 | MATCH | material : wool | MATCH | {} |
| 25537 | animal print tufted wool brown/black area rug | Partial | lost | 6→21 | MATCH | material : wool | MATCH | {} |
| 25569 | roloff hand-tufted wool ivory/black area rug | Partial | gained | 23→20 | MATCH | material : wool | MATCH | {} |
| 27973 | casual tufted wool beige area rug | Partial | lost | 19→25 | MATCH | material : wool | MATCH | {} |
| 32608 | confidence hand knotted wool black/beige area rug | Partial | gained | 24→9 | MATCH | material : wool | MATCH | {} |
| 40603 | penley animal print tufted brown/black/ivory area rug | Partial | gained | 50→17 | AMBIGUOUS | material : viscose; material : wool | MATCH | {} |
| 6201 | power loom red/beige/gray area rug | Partial | gained | 99→18 | CONTRADICTION | material : polypropylene | MATCH | {} |

留在 Top-20 的全部商品同樣列於 `all_products.csv` 與 `audit.json`。`source_products` 保存原始完整欄位、標題、描述；`all_clause_evidence` 保留逐條件判定。歷史名次若未保存，僅標 `>20`；新重算精確名次在另一張表，沒有回填冒充歷史。

## 歷史指標

| 來源 | Schedule | Exact count | Recall | DCG | nDCG | qrel E/P/I/unjudged |
| --- | --- | --- | --- | --- | --- | --- |
| historical | before | 0 | 0.000000000000000 | 7.040268381923513 | 0.683380021849204 | {'Exact': 0, 'Partial': 20, 'Irrelevant': 0, 'Unjudged': 0} |
| historical | after | 0 | 0.000000000000000 | 7.040268381923513 | 0.683380021849204 | {'Exact': 0, 'Partial': 20, 'Irrelevant': 0, 'Unjudged': 0} |

nDCG equality = `exact_gain_sequence`；Δ = 0.000000000000000。nonzero Exact-cancellation = False。

## 相同 FP32 快取計分重現（分表）

| 來源 | Schedule | Exact count | Recall | DCG | nDCG | Top20 順序相符 | 20−21 score margin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| matched cached scoring | C0 | 0 | 0.000000000000000 | 7.040268381923513 | 0.683380021849204 | True | 0.00185418128967 |
| matched cached scoring | C1 | 0 | 0.000000000000000 | 7.040268381923513 | 0.683380021849204 | True | 0.00711208581924 |

兩邊均使用同一 pinned 模型 revision、同一 query cache、來源雜湊相符的 product caches、單 query × 42,994 full catalog 的 CPU FP32 dot product，以及 stable catalog-index tie-breaking。三例都逐位重現 Top20 順序。只重用向量，沒有重新 forward；歷史 CUDA fp16 encoder 的實際 batch padding/OOM subdivision/GPU 型號仍 unknown。不能據此聲稱所有歷史浮點路徑逐 bit 一致。每件與 rank20/21 的分差及 union 精確名次均保存於 `matched_scoring.json`；margin 不用來淘汰案例。

## 真實模板與 token 稽核

tokenizer revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`；BertTokenizerFast，含 2 特殊 tokens；padding/truncation side = right/right。已核對保存的 full-catalog representation 順序与 SHA；本案例每件文字逐字等於原模板產生值。query 18 tokens，完整輸入。

| ID | 進出 | C0 tokens 含特殊 | C1 tokens 含特殊 | 上限 | 兩序列均完整 |
| --- | --- | --- | --- | --- | --- |
| 11497 | lost | 411 | 411 | 256 | False |
| 15536 | lost | 672 | 672 | 256 | False |
| 22340 | lost | 491 | 491 | 256 | False |
| 24511 | gained | 740 | 740 | 256 | False |
| 25537 | lost | 475 | 475 | 256 | False |
| 25569 | gained | 838 | 838 | 256 | False |
| 27973 | lost | 422 | 422 | 256 | False |
| 32608 | gained | 203 | 203 | 256 | True |
| 40603 | gained | 477 | 477 | 256 | False |
| 6201 | gained | 490 | 490 | 256 | False |

進出商品 10 件、C0/C1 共 20 序列，只有 2 序列完整輸入，長度 203–838。聯集 25 件、50 序列，完整 4，長度 203–838。不能推論全 catalog 無截斷。全部 raw attributes 的重數、title/class/category/description、query 均不變；encoder 可見前綴會因截斷及次序改變，因此這是 native 截斷 pipeline 的表示干預案例，不能隔離為完整輸入中的純位置效應。

`input_sequences.jsonl` 保存實際 C0/C1 字串與截斷後 input IDs；`token_audit.csv` 保存所有聯集商品而非僅進出者。沒有使用外部商品頁或圖片補值。
