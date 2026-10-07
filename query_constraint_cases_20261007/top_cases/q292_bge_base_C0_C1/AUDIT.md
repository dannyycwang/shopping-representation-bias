# Query 292: decorative white pillow

最終等級 **C**，限 `white` / `whole` 子條件；bge_base_C0_C1，catalog-wide，K=20。原始 query 不變。完整集合包括所有 qrels；沒有修改官方 relevance label。

## 來源逐件稽核

逐件讀取 27 件聯集的所有來源。兩件白色 MATCH 退出為 13387（完整 cover & insert，但 qrel 是 Irrelevant）與 2901（pillow insert、Partial，完整裝飾抱枕類型仍有歧義）。藍色 13395（Partial）进入；另一件藍色 13421 同時退出，必須同時計入。13375 的 `purple/black` 依固定複合值規則是 AMBIGUOUS，不能為了案例把它轉成單色 CONTRADICTION。

14721 與 40077 是 pillow cover；保留的白色 21561 也是 cover，39176 是 insert。21241 名稱含 white angels，但其完整紀錄是 yellow/teal/blue，描述包含彩色套組及白色腰枕，不能當一致白色 item。21427 有 mustard/gray/blush/blue 變體與白色底布描述；9538 有大量不同色值；其他混色值逐筆保留。這不是乾淨的純顏色替換。nDCG 上升，Exact 前後皆 0/12；不存在非零 Exact-cancellation。joint-MATCH 1→0 也不能稱為選項消失，因 after 還有 UNKNOWN/AMBIGUOUS。

## 全部條件與集合（M/C/U/A）

M=MATCH、C=CONTRADICTION、U=UNKNOWN、A=AMBIGUOUS；U/A 不當作不符合。joint 額外包含 product type、用途、所有其他文字限制及部位縮限。

| 子條件 | 範圍 | C0 M/C/U/A | C1 M/C/U/A | tight Δ |
| --- | --- | --- | --- | --- |
| white | whole | 4/1/2/13 | 2/1/2/15 | [-6, 4] |

| 集合 | C0 M/C/U/A | C1 M/C/U/A | L M/C/U/A | G M/C/U/A | loose Δ | tight Δ | 消失已證明 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full | 4/1/2/13 | 2/1/2/15 | 2/1/1/3 | 0/1/1/5 | [-17, 15] | [-6, 4] | False |
| exact | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | [0, 0] | [0, 0] | False |
| joint | 1/2/2/15 | 0/4/1/15 | 1/1/1/4 | 0/3/0/4 | [-18, 15] | [-6, 3] | False |

## 完整進出清單

| ID | 名稱 | qrel | 進出 | 歷史 C0→C1 | 條件 | 原始對應欄位 | 類型 | 用途 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13375 | exotic light decorative square pillow cover & insert | Partial | gained | 80→7 | AMBIGUOUS | color : purple/black | MATCH | {} |
| 13377 | kint fun inspire decorative square pillow cover & insert | Irrelevant | lost | 14→22 | AMBIGUOUS | color : beige/brown | MATCH | {} |
| 13387 | slick quality decorative square pillow cover & insert | Irrelevant | lost | 18→31 | MATCH | color : white | MATCH | {} |
| 13389 | colorful relax decorative square pillow cover & insert | Partial | gained | 48→15 | AMBIGUOUS | color : graylight green | MATCH | {} |
| 13395 | pamper bright decorative square pillow cover & insert | Partial | gained | 33→19 | CONTRADICTION | color : blue | MATCH | {} |
| 13417 | appealing effervescent decorative square pillow cover & insert | Partial | lost | 7→32 | AMBIGUOUS | color : yellow/beige | MATCH | {} |
| 13418 | lingering decorative square pillow cover & insert | Partial | gained | 31→17 | AMBIGUOUS | color : yellow/blue | MATCH | {} |
| 13421 | real tangy decorative square pillow cover & insert | Partial | lost | 17→21 | CONTRADICTION | color : blue | MATCH | {} |
| 14721 | square pillow cover | Unjudged | gained | 44→20 | AMBIGUOUS | color : beige/gray | CONTRADICTION | {} |
| 21427 | square cotton pillow cover and insert | Partial | gained | 28→16 | AMBIGUOUS | color : mustard; color : gray; color : blush; color : blue | MATCH | {} |
| 2901 | pillow insert | Partial | lost | 15→29 | MATCH | color : white | AMBIGUOUS | {} |
| 40077 | dominga cotton throw pillow cover | Unjudged | gained | 21→14 | UNKNOWN |  | CONTRADICTION | {} |
| 5536 | hyatt round throw pillow | Unjudged | lost | 20→24 | UNKNOWN |  | MATCH | {} |
| 9538 | mcgee relax thin outdoor rectangular pillow cover & insert | Partial | lost | 11→28 | AMBIGUOUS | color : off white/black; color : orange; color : dark gray/white; color : mint green; color : navy; color : charcoal; color : white/gray; color : coral; color : yellow | MATCH | {} |

留在 Top-20 的全部商品同樣列於 `all_products.csv` 與 `audit.json`。`source_products` 保存原始完整欄位、標題、描述；`all_clause_evidence` 保留逐條件判定。歷史名次若未保存，僅標 `>20`；新重算精確名次在另一張表，沒有回填冒充歷史。

## 歷史指標

| 來源 | Schedule | Exact count | Recall | DCG | nDCG | qrel E/P/I/unjudged |
| --- | --- | --- | --- | --- | --- | --- |
| historical | before | 0 | 0.000000000000000 | 4.167101255677560 | 0.241911175717368 | {'Exact': 0, 'Partial': 11, 'Irrelevant': 3, 'Unjudged': 6} |
| historical | after | 0 | 0.000000000000000 | 4.563005929577276 | 0.264894482159609 | {'Exact': 0, 'Partial': 12, 'Irrelevant': 1, 'Unjudged': 7} |

nDCG equality = `increased`；Δ = 0.022983306442240。nonzero Exact-cancellation = False。

## 相同 FP32 快取計分重現（分表）

| 來源 | Schedule | Exact count | Recall | DCG | nDCG | Top20 順序相符 | 20−21 score margin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| matched cached scoring | C0 | 0 | 0.000000000000000 | 4.167101255677560 | 0.241911175717368 | True | 6.79492950439e-06 |
| matched cached scoring | C1 | 0 | 0.000000000000000 | 4.563005929577276 | 0.264894482159609 | True | 0.000457167625427 |

兩邊均使用同一 pinned 模型 revision、同一 query cache、來源雜湊相符的 product caches、單 query × 42,994 full catalog 的 CPU FP32 dot product，以及 stable catalog-index tie-breaking。三例都逐位重現 Top20 順序。只重用向量，沒有重新 forward；歷史 CUDA fp16 encoder 的實際 batch padding/OOM subdivision/GPU 型號仍 unknown。不能據此聲稱所有歷史浮點路徑逐 bit 一致。每件與 rank20/21 的分差及 union 精確名次均保存於 `matched_scoring.json`；margin 不用來淘汰案例。

## 真實模板與 token 稽核

tokenizer revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`；BertTokenizerFast，含 2 特殊 tokens；padding/truncation side = right/right。已核對保存的 full-catalog representation 順序与 SHA；本案例每件文字逐字等於原模板產生值。query 5 tokens，完整輸入。

| ID | 進出 | C0 tokens 含特殊 | C1 tokens 含特殊 | 上限 | 兩序列均完整 |
| --- | --- | --- | --- | --- | --- |
| 13375 | gained | 629 | 629 | 512 | False |
| 13377 | lost | 631 | 631 | 512 | False |
| 13387 | lost | 638 | 638 | 512 | False |
| 13389 | gained | 640 | 640 | 512 | False |
| 13395 | gained | 639 | 639 | 512 | False |
| 13417 | lost | 637 | 637 | 512 | False |
| 13418 | gained | 637 | 637 | 512 | False |
| 13421 | lost | 642 | 642 | 512 | False |
| 14721 | gained | 616 | 616 | 512 | False |
| 21427 | gained | 649 | 649 | 512 | False |
| 2901 | lost | 702 | 702 | 512 | False |
| 40077 | gained | 517 | 517 | 512 | False |
| 5536 | lost | 616 | 616 | 512 | False |
| 9538 | lost | 494 | 494 | 512 | True |

進出商品 14 件、C0/C1 共 28 序列，只有 2 序列完整輸入，長度 494–702。聯集 27 件、54 序列，完整 8，長度 366–787。不能推論全 catalog 無截斷。全部 raw attributes 的重數、title/class/category/description、query 均不變；encoder 可見前綴會因截斷及次序改變，因此這是 native 截斷 pipeline 的表示干預案例，不能隔離為完整輸入中的純位置效應。

`input_sequences.jsonl` 保存實際 C0/C1 字串與截斷後 input IDs；`token_audit.csv` 保存所有聯集商品而非僅進出者。沒有使用外部商品頁或圖片補值。
