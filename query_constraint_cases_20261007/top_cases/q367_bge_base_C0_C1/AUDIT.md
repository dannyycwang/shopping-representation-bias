# Query 367: wooden chair outdoor

最終等級 **A**，限 `wood` / `frame` 子條件；bge_base_C0_C1，catalog-wide，K=20。原始 query 不變。完整集合包括所有 qrels；沒有修改官方 relevance label。

## 來源逐件稽核

逐件讀取 24 件聯集商品的全部 attributes、title 與 description。退出的 Fordyce (1817)、Joetta (26196)、Fitzhugh (3604)、Vandergriff (36317) 都有 solid-wood 框架紀錄；進入的 Jettie (26195)、Traditional rocking chair (35827) 亦有木框架；Lewallen (32126) 記錄 plastic/resin 框架，Keating (4355) 記錄 metal 框架。這是四出四進的同時集合變化，沒有指定一件因果排擠另一件。

保留商品也完整納入。41122/41124/41125 雖記錄 solid wood，描述另提 aluminum rocking base，因此保守列為框架部位歧義；三件都前後保留，會改變 MATCH 水準而不改變淨差。15057 有 wood/metal 多值，21267 有 wood/wicker 多值；14696 的 wicker/rattan 外框與金屬框描述有部位問題；18010 的 manufactured-wood 欄位與 pine/fir 描述不能直接當 solid wood；5701 有金屬椅身與再生木扶手，依保守規則保留歧義。6641 雖名含 wood、描述也說木構造，沒有本次映射的 frame 欄位，故 frame-clause 為 UNKNOWN，不偷偷由標題補成 MATCH。

Lewallen 的完整紀錄是 `framematerial : plastic/resin`，details 為 `polywood ( hdps )`；`weather wood` / `textured wood grain` 描述外觀。另保留 `dswoodtone : white wood`、`hdpematerial : no`，不將外觀詞當木材證明，也不聲稱特定聚合物種類已確認。未找到足以核定座面材質的獨立欄位；結論僅限框架。Keating 的木製扶手裝飾不推翻明示 aluminum frame，但也說明不能把「非木框架」改寫成整張椅子無木。所有 24 件 chair 類型有來源支持；outdoor 為 23 件 MATCH、保留的 15057 為 UNKNOWN，不能只由 weatherproofed 推定用途。8 件進出商品的 chair/outdoor 均為 MATCH。chair 商品可為套組，query 未指定單件數量。frame 子條件仍不代表整句 wooden 的唯一解釋。

四件 Exact 進出商品與題目線索均吻合；Lewallen 仍照原始 benchmark 計為 Exact。這揭露 label granularity / annotation mismatch 的可能性，並不證明官方標籤錯誤。

## 全部條件與集合（M/C/U/A）

M=MATCH、C=CONTRADICTION、U=UNKNOWN、A=AMBIGUOUS；U/A 不當作不符合。joint 額外包含 product type、用途、所有其他文字限制及部位縮限。

| 子條件 | 範圍 | C0 M/C/U/A | C1 M/C/U/A | tight Δ |
| --- | --- | --- | --- | --- |
| wood | frame | 8/3/1/8 | 6/5/1/8 | [-2, -2] |

| 集合 | C0 M/C/U/A | C1 M/C/U/A | L M/C/U/A | G M/C/U/A | loose Δ | tight Δ | 消失已證明 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full | 8/3/1/8 | 6/5/1/8 | 4/0/0/0 | 2/2/0/0 | [-11, 7] | [-2, -2] | False |
| exact | 5/0/1/1 | 4/1/1/1 | 2/0/0/0 | 1/1/0/0 | [-3, 1] | [-1, -1] | False |
| joint | 0/3/0/17 | 0/5/0/15 | 0/0/0/4 | 0/2/0/2 | [-17, 15] | [-4, 2] | False |

## 完整進出清單

| ID | 名稱 | qrel | 進出 | 歷史 C0→C1 | 條件 | 原始對應欄位 | 類型 | 用途 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1817 | fordyce rocking chair | Unjudged | lost | 9→>20 | MATCH | outerframematerial : solid wood | MATCH | {'outdoor': 'MATCH'} |
| 26195 | jettie modern outdoor wood patio chair with cushions | Exact | gained | 30→20 | MATCH | outerframematerial : solid wood | MATCH | {'outdoor': 'MATCH'} |
| 26196 | joetta outdoor wood patio chair with cushions | Exact | lost | 6→23 | MATCH | outerframematerial : solid wood | MATCH | {'outdoor': 'MATCH'} |
| 32126 | lewallen weather wood adirondack chair | Exact | gained | 38→17 | CONTRADICTION | framematerial : plastic/resin | MATCH | {'outdoor': 'MATCH'} |
| 35827 | traditional rocking chair | Unjudged | gained | 21→13 | MATCH | outerframematerial : solid wood | MATCH | {'outdoor': 'MATCH'} |
| 3604 | fitzhugh patio chair | Unjudged | lost | 18→28 | MATCH | outerframematerial : solid wood | MATCH | {'outdoor': 'MATCH'} |
| 36317 | vandergriff lounge patio chair with cushions | Exact | lost | 12→69 | MATCH | outerframematerial : solid wood | MATCH | {'outdoor': 'MATCH'} |
| 4355 | keating 28 '' wide outdoor chair with cushions | Partial | gained | 37→5 | CONTRADICTION | outerframematerial : metal | MATCH | {'outdoor': 'MATCH'} |

留在 Top-20 的全部商品同樣列於 `all_products.csv` 與 `audit.json`。`source_products` 保存原始完整欄位、標題、描述；`all_clause_evidence` 保留逐條件判定。歷史名次若未保存，僅標 `>20`；新重算精確名次在另一張表，沒有回填冒充歷史。

## 歷史指標

| 來源 | Schedule | Exact count | Recall | DCG | nDCG | qrel E/P/I/unjudged |
| --- | --- | --- | --- | --- | --- | --- |
| historical | before | 7 | 0.166666666666667 | 9.174026565841174 | 0.434359699004102 | {'Exact': 7, 'Partial': 3, 'Irrelevant': 0, 'Unjudged': 10} |
| historical | after | 7 | 0.166666666666667 | 9.061091192078836 | 0.429012584015687 | {'Exact': 7, 'Partial': 4, 'Irrelevant': 0, 'Unjudged': 9} |

nDCG equality = `decreased`；Δ = -0.005347114988415。nonzero Exact-cancellation = True。

## 相同 FP32 快取計分重現（分表）

| 來源 | Schedule | Exact count | Recall | DCG | nDCG | Top20 順序相符 | 20−21 score margin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| matched cached scoring | C0 | 7 | 0.166666666666667 | 9.174026565841174 | 0.434359699004102 | True | 0.000113725662231 |
| matched cached scoring | C1 | 7 | 0.166666666666667 | 9.061091192078836 | 0.429012584015687 | True | 0.000523686408997 |

兩邊均使用同一 pinned 模型 revision、同一 query cache、來源雜湊相符的 product caches、單 query × 42,994 full catalog 的 CPU FP32 dot product，以及 stable catalog-index tie-breaking。三例都逐位重現 Top20 順序。只重用向量，沒有重新 forward；歷史 CUDA fp16 encoder 的實際 batch padding/OOM subdivision/GPU 型號仍 unknown。不能據此聲稱所有歷史浮點路徑逐 bit 一致。每件與 rank20/21 的分差及 union 精確名次均保存於 `matched_scoring.json`；margin 不用來淘汰案例。

## 真實模板與 token 稽核

tokenizer revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`；BertTokenizerFast，含 2 特殊 tokens；padding/truncation side = right/right。已核對保存的 full-catalog representation 順序与 SHA；本案例每件文字逐字等於原模板產生值。query 5 tokens，完整輸入。

| ID | 進出 | C0 tokens 含特殊 | C1 tokens 含特殊 | 上限 | 兩序列均完整 |
| --- | --- | --- | --- | --- | --- |
| 1817 | lost | 770 | 770 | 512 | False |
| 26195 | gained | 780 | 780 | 512 | False |
| 26196 | lost | 870 | 870 | 512 | False |
| 32126 | gained | 568 | 568 | 512 | False |
| 35827 | gained | 555 | 555 | 512 | False |
| 3604 | lost | 759 | 759 | 512 | False |
| 36317 | lost | 775 | 775 | 512 | False |
| 4355 | gained | 793 | 793 | 512 | False |

進出商品 8 件、C0/C1 共 16 序列，只有 0 序列完整輸入，長度 555–870。聯集 24 件、48 序列，完整 10，長度 333–1036。不能推論全 catalog 無截斷。全部 raw attributes 的重數、title/class/category/description、query 均不變；encoder 可見前綴會因截斷及次序改變，因此這是 native 截斷 pipeline 的表示干預案例，不能隔離為完整輸入中的純位置效應。

`input_sequences.jsonl` 保存實際 C0/C1 字串與截斷後 input IDs；`token_audit.csv` 保存所有聯集商品而非僅進出者。沒有使用外部商品頁或圖片補值。
