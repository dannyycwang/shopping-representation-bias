"""Render reports/tables from validated results, keeping explicit source caveats."""
import re
from common import *

def write(path,text):(HERE/path).write_text(text.strip()+'\n',encoding='utf8')
def table(headers,rows):
    def cell(x):return str(x).replace('|',' / ').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(cell,r))+' |' for r in rows])
def countstr(d):return '/'.join(str(d[k]) for k in ['MATCH','CONTRADICTION','UNKNOWN','AMBIGUOUS'])
def n(x):return f'{x:.15f}'

NOTES={
367:"""逐件讀取 24 件聯集商品的全部 attributes、title 與 description。退出的 Fordyce (1817)、Joetta (26196)、Fitzhugh (3604)、Vandergriff (36317) 都有 solid-wood 框架紀錄；進入的 Jettie (26195)、Traditional rocking chair (35827) 亦有木框架；Lewallen (32126) 記錄 plastic/resin 框架，Keating (4355) 記錄 metal 框架。這是四出四進的同時集合變化，沒有指定一件因果排擠另一件。

保留商品也完整納入。41122/41124/41125 雖記錄 solid wood，描述另提 aluminum rocking base，因此保守列為框架部位歧義；三件都前後保留，會改變 MATCH 水準而不改變淨差。15057 有 wood/metal 多值，21267 有 wood/wicker 多值；14696 的 wicker/rattan 外框與金屬框描述有部位問題；18010 的 manufactured-wood 欄位與 pine/fir 描述不能直接當 solid wood；5701 有金屬椅身與再生木扶手，依保守規則保留歧義。6641 雖名含 wood、描述也說木構造，沒有本次映射的 frame 欄位，故 frame-clause 為 UNKNOWN，不偷偷由標題補成 MATCH。

Lewallen 的完整紀錄是 `framematerial : plastic/resin`，details 為 `polywood ( hdps )`；`weather wood` / `textured wood grain` 描述外觀。另保留 `dswoodtone : white wood`、`hdpematerial : no`，不將外觀詞當木材證明，也不聲稱特定聚合物種類已確認。未找到足以核定座面材質的獨立欄位；結論僅限框架。Keating 的木製扶手裝飾不推翻明示 aluminum frame，但也說明不能把「非木框架」改寫成整張椅子無木。所有 24 件 chair 類型有來源支持；outdoor 為 23 件 MATCH、保留的 15057 為 UNKNOWN，不能只由 weatherproofed 推定用途。8 件進出商品的 chair/outdoor 均為 MATCH。chair 商品可為套組，query 未指定單件數量。frame 子條件仍不代表整句 wooden 的唯一解釋。

四件 Exact 進出商品與題目線索均吻合；Lewallen 仍照原始 benchmark 計為 Exact。這揭露 label granularity / annotation mismatch 的可能性，並不證明官方標籤錯誤。""",
292:"""逐件讀取 27 件聯集的所有來源。兩件白色 MATCH 退出為 13387（完整 cover & insert，但 qrel 是 Irrelevant）與 2901（pillow insert、Partial，完整裝飾抱枕類型仍有歧義）。藍色 13395（Partial）进入；另一件藍色 13421 同時退出，必須同時計入。13375 的 `purple/black` 依固定複合值規則是 AMBIGUOUS，不能為了案例把它轉成單色 CONTRADICTION。

14721 與 40077 是 pillow cover；保留的白色 21561 也是 cover，39176 是 insert。21241 名稱含 white angels，但其完整紀錄是 yellow/teal/blue，描述包含彩色套組及白色腰枕，不能當一致白色 item。21427 有 mustard/gray/blush/blue 變體與白色底布描述；9538 有大量不同色值；其他混色值逐筆保留。這不是乾淨的純顏色替換。nDCG 上升，Exact 前後皆 0/12；不存在非零 Exact-cancellation。joint-MATCH 1→0 也不能稱為選項消失，因 after 還有 UNKNOWN/AMBIGUOUS。""",
300:"""逐件讀取 25 件聯集的所有來源。11497、22340、25537、27973 有明確 wool 欄位而退出；6201 以 polypropylene 為材質而进入。24511/25569/32608 的 wool 進入也完整计入，40603 的 wool + viscose 是 AMBIGUOUS。15534/15535 保留的混紡亦為 AMBIGUOUS。

關鍵修正：退出的 15536 雖只有 `material : wool`，完整描述說 Thornbury collection 許多設計帶 viscose。依不能忽略 collection/variant 衝突的規則降為 AMBIGUOUS。因此最初只靠欄位/局部文字的 18→16、界限 [-2,-1] 不能作為最終結果；最終 17→16、界限 [-2,0]，降為 B。未改選其他 query 補出 A 級結果。

整個前後 Top-20 全是 Partial，H=2 而 Exact recall 為 0/2。兩個 20 維 gain 序列逐位相同，所以 nDCG 真正相等，並非四捨五入或容差判等。這仍只是一個 wool 子條件案例：beige/black、多色，以及 animal print、handmade、tufted、AllModern 等限制未完全核定，不能稱為整句 query 滿足數。三個入選案例均有截斷，這例不是 fully-fitting 的替代例子。"""}

def main():
    protocol=read('query_constraint_cases_20261007/protocol.json');qa=read('query_constraint_cases_20261007/qa/independent_verification.json')
    index=read('query_constraint_cases_20261007/case_index.json')
    cases={i['query_id']:read('query_constraint_cases_20261007/top_cases/'+i['case_name']+'/audit.json') for i in index}
    comparisons=pd.read_csv(HERE/'comparison_summary.csv');rows=lines('data/query_comparisons.jsonl');scanned=[r for r in rows if r['scanned']]
    comparison_rows=[]
    for qid in [367,292,300]:
        c=cases[qid];s=c['summary'];t=c['token_summary']
        comparison_rows.append([qid,s['model'],c['final_grade']+'（子條件）',f"{s['relevant_before']}/{s['highest_n']} → {s['relevant_after']}/{s['highest_n']}",
            f"{s['full']['before']['MATCH']}→{s['full']['after']['MATCH']}",str(s['full']['tight_bounds']),f"{s['exact']['before']['MATCH']}→{s['exact']['after']['MATCH']}",
            n(s['delta_ndcg']),f"{t['changed_fitting_sequences']}/{t['changed_schedule_sequences']}"])
    comparison_table=table(['Query','Model','最終等級','Exact Recall','完整 T MATCH','抵消共有項 Δ 界限','Exact R MATCH','Δ nDCG','進出序列完整輸入'],comparison_rows)
    for qid,c in cases.items():
        s=c['summary'];t=c['token_summary'];folder='top_cases/'+c['case_name'];products=c['products']
        ct=table(['集合','C0 M/C/U/A','C1 M/C/U/A','L M/C/U/A','G M/C/U/A','loose Δ','tight Δ','消失已證明'],
            [[pop,countstr(s[pop]['before']),countstr(s[pop]['after']),countstr(s[pop]['lost']),countstr(s[pop]['gained']),s[pop]['loose_bounds'],s[pop]['tight_bounds'],s[pop]['disappearance']] for pop in ['full','exact','joint']])
        changes=table(['ID','名稱','qrel','進出','歷史 C0→C1','條件','原始對應欄位','類型','用途'],[[p['product_id'],p['title'],p['qrel'],p['membership'],f"{p['before_rank']}→{p['after_rank']}",p['constraint_assessment']['status'], '; '.join(a['raw_entry'] for a in p['constraint_assessment']['raw_fields']),p['type_assessment']['status'],str({k:v['status'] for k,v in p['use_assessments'].items()})] for p in products if p['membership']!='retained'])
        metrics=table(['來源','Schedule','Exact count','Recall','DCG','nDCG','qrel E/P/I/unjudged'],
            [['historical',side,s['relevant_'+side],n(s['recall_'+side]),n(s['dcg_'+side]),n(s['ndcg_'+side]),str(s['qrel_counts_'+side])] for side in ['before','after']])
        newmetrics=table(['來源','Schedule','Exact count','Recall','DCG','nDCG','Top20 順序相符','20−21 score margin'],
            [['matched cached scoring',sch,x['metrics']['relevant_count'],n(x['metrics']['recall']),n(x['metrics']['dcg']),n(x['metrics']['ndcg']),x['top20_order_matches_history'],f"{x['cutoff_margin']:.12g}"] for sch,x in c['matched_scoring'].items()])
        tokens=pd.read_csv(HERE/folder/'token_audit.csv');tt=[]
        for p in products:
            if p['membership']=='retained':continue
            r=tokens[tokens.product_id.astype(str).eq(p['product_id'])].set_index('schedule')
            tt.append([p['product_id'],p['membership'],int(r.loc['C0','tokens_including_special']),int(r.loc['C1','tokens_including_special']),t['max_length'],bool(r.fully_fitting.all())])
        token_table=table(['ID','進出','C0 tokens 含特殊','C1 tokens 含特殊','上限','兩序列均完整'],tt)
        clause_table=table(['子條件','範圍','C0 M/C/U/A','C1 M/C/U/A','tight Δ'],[[x['constraint'],x['scope'],countstr(x['full']['before']),countstr(x['full']['after']),x['full']['tight_bounds']] for x in c['all_clauses']])
        text=f"""# Query {qid}: {s['query']}

最終等級 **{c['final_grade']}**，限 `{s['constraint']}` / `{s['scope']}` 子條件；{s['comparison']}，catalog-wide，K=20。原始 query 不變。完整集合包括所有 qrels；沒有修改官方 relevance label。

## 來源逐件稽核

{NOTES[qid]}

## 全部條件與集合（M/C/U/A）

M=MATCH、C=CONTRADICTION、U=UNKNOWN、A=AMBIGUOUS；U/A 不當作不符合。joint 額外包含 product type、用途、所有其他文字限制及部位縮限。

{clause_table}

{ct}

## 完整進出清單

{changes}

留在 Top-20 的全部商品同樣列於 `all_products.csv` 與 `audit.json`。`source_products` 保存原始完整欄位、標題、描述；`all_clause_evidence` 保留逐條件判定。歷史名次若未保存，僅標 `>20`；新重算精確名次在另一張表，沒有回填冒充歷史。

## 歷史指標

{metrics}

nDCG equality = `{s['ndcg_equality']}`；Δ = {n(s['delta_ndcg'])}。nonzero Exact-cancellation = {s['exact_cancellation']}。

## 相同 FP32 快取計分重現（分表）

{newmetrics}

兩邊均使用同一 pinned 模型 revision、同一 query cache、來源雜湊相符的 product caches、單 query × 42,994 full catalog 的 CPU FP32 dot product，以及 stable catalog-index tie-breaking。三例都逐位重現 Top20 順序。只重用向量，沒有重新 forward；歷史 CUDA fp16 encoder 的實際 batch padding/OOM subdivision/GPU 型號仍 unknown。不能據此聲稱所有歷史浮點路徑逐 bit 一致。每件與 rank20/21 的分差及 union 精確名次均保存於 `matched_scoring.json`；margin 不用來淘汰案例。

## 真實模板與 token 稽核

tokenizer revision `{t['revision']}`；{t['tokenizer_class']}，含 {2} 特殊 tokens；padding/truncation side = {t['padding_side']}/{t['truncation_side']}。已核對保存的 full-catalog representation 順序与 SHA；本案例每件文字逐字等於原模板產生值。query {t['query_tokens_with_special']} tokens，完整輸入。

{token_table}

進出商品 {t['changed_products']} 件、C0/C1 共 {t['changed_schedule_sequences']} 序列，只有 {t['changed_fitting_sequences']} 序列完整輸入，長度 {t['changed_min_tokens']}–{t['changed_max_tokens']}。聯集 {t['union_products']} 件、{t['union_schedule_sequences']} 序列，完整 {t['union_fitting_sequences']}，長度 {t['union_min_tokens']}–{t['union_max_tokens']}。不能推論全 catalog 無截斷。全部 raw attributes 的重數、title/class/category/description、query 均不變；encoder 可見前綴會因截斷及次序改變，因此這是 native 截斷 pipeline 的表示干預案例，不能隔離為完整輸入中的純位置效應。

`input_sequences.jsonl` 保存實際 C0/C1 字串與截斷後 input IDs；`token_audit.csv` 保存所有聯集商品而非僅進出者。沒有使用外部商品頁或圖片補值。
"""
        write(folder+'/AUDIT.md',text)
        if qid==367:write('wooden_chair_outdoor_audit.md',text.replace('`all_products.csv`','`top_cases/q367_bge_base_C0_C1/all_products.csv`').replace('`audit.json`','`top_cases/q367_bge_base_C0_C1/audit.json`'))
    cohort_table=table(['Comparison','Query contrasts','有條件 queries / clauses','Exact cancellation','Equal Recall','Candidates','Screen A/B/C','證實下降/上升','未定下降/上升','confirmed 不變'],
        [[r.comparison,r.evaluation_queries,f'{r.attribute_queries}/{r.clause_rows}',r.cancellation_queries,r.equal_recall_queries,r.candidates,f'{r.grade_A}/{r.grade_B}/{r.grade_C}',f'{r.proved_decrease}/{r.proved_increase}',f'{r.confirmed_decrease_uncertain}/{r.confirmed_increase_uncertain}',r.confirmed_unchanged] for r in comparisons.itertuples()])
    sums=comparisons.iloc[:,2:].sum();primary=scanned[:0];primary=[r for r in scanned if r['primary']]
    no_match=sum(r['full']['before']['MATCH']==r['full']['after']['MATCH']==0 for r in primary)
    proven_vanish=sum(r['full']['disappearance'] for r in scanned)
    unique_candidates=len({r['query_id'] for r in scanned if r['candidate']})
    known_ranks_missing=sum(p['before_rank']=='>20' or p['after_rank']=='>20' for c in cases.values() for p in c['products'])
    mis=pd.read_csv(HERE/'missingness_summary.csv');mm=mis[mis.comparison.eq('bge_base_C0_C1')&mis.population.eq('full')]
    missing_table=table(['主分析所有 68 子條件×Top20（非獨立商品）','MATCH','CONTRADICTION','UNKNOWN','AMBIGUOUS','總數'],[[r.side,r.MATCH,r.CONTRADICTION,r.UNKNOWN,r.AMBIGUOUS,r.total_product_clause_instances] for r in mm.itertuples()])
    bar=next(r for r in primary if r['query_id']==440)
    doc=f"""# Query-explicit constraint case search — 2026-10-07

**三個完整稽核案例中，沒有找到比 query 367 更清楚的替代例子。** 木椅可作 **A 級、frame-material 子條件** 的探索性 illustration：完整 Top20 的確認木框架數 8→6，Exact 子集 5→4；相同商品的不確定性抵消後淨差分別固定為 −2、−1。Recall 完全相同，但 nDCG 下降。白色抱枕為 C；MiniLM 羊毛地毯最初看似 A，完整描述揭露混紡變體後降為 B，沒有重選第四例。

重要限制：木椅 8 件進出商品的 16 個實際序列全部超過 512 tokens。因此可以用於 native catalog-wide pipeline 的案例，**不能作 fully-fitting 或已排除截斷的證據**。本次只證實既有向量在同一計分實作重現；未重新執行模型 forward。

{comparison_table}

A/B/C 等級針對明確標出的子條件。白色案例產品類型及混色歧義較多；羊毛案例的 0/2 Recall 及多重未核定限制，不宜用來取代木椅。其餘篩檢 A 級（如 q275/q417 的 secondary contrasts）沒有在三例上限之外做完整來源/token/重現稽核，**不能聲稱搜尋空間不存在更佳案例**。

## 設定、來源與有效分母

實際起點 `{protocol['actual_starting_commit']}`，歷史參考 `{protocol['historical_reference_commit']}`。兩者 tree 相同，參考是 merge commit；來源檔逐一 SHA 保存，沒有靜默拼接不同版本。主分析 WANDS/BGE native C0→C1，K=20，所有 42,994 競爭商品都套用 attribute ordering。它不是 target-only/fixed-competitor。

原始 480 queries；既有 heldout {qa['heldout_queries']}；其中 {qa['no_exact_heldout']} 沒有 Exact，排除後 eligible {qa['eligible_queries']}，Exact query-product pairs {qa['highest_pairs']}。H_q 固定為原 processed qrels 的 Exact；Recall=|Top20∩H_q|/|H_q|，gain 為 3/1/0，unjudged 評分 0、但不等於屬性 CONTRADICTION。獨立重核 raw labels 清理：{qa['raw_qrel_conflicting_pairs_removed']} conflicting pairs、{qa['raw_duplicate_rows_removed']} duplicate rows；清理結果與 {qa['clean_judged_pairs']} 筆 processed qrels 完全一致。原始 query 逐字相同。

依 query/schema 預先保存 62 queries、68 個可執行屬性子條件；其餘 246 queries 也有排除/延後理由。numeric dimension 缺可靠單位/方向而延後；直接記錄的 nominal bedding sizes 有納入。所有多條件原文與其他限制保留，不由單條件 MATCH 推論整句滿足。

14 個 schedule runs、12 個預定 contrasts 全可用；總 {qa['query_comparisons']} query-contrast、{qa['assessed_clause_comparisons']} assessed clause-contrast，另有未執行 clause 的 placeholders，合計 {qa['all_summary_rows']} summary rows。744 個 attribute-query-contrast 其實是同 62 queries 重複 12 次，816 clauses 亦非獨立樣本。76 candidate clause-contrast，來自 {unique_candidates} 個 distinct queries；同 query 不同模型/seed 重複保留。主分析 73 nonzero Exact-cancellation、194 equal-count Recall queries，均由整數集合重算，不用先前常數強制結果。

## 全範圍結果：包括相反方向與零結果

{cohort_table}

表中 A/B/C 是保守自動 screen 的級別，不能視為全部已人工核實的案例。12 組共 {int(sums['proved_decrease'])} 個 clause-contrast 有 tightened bound 支持下降、{int(sums['proved_increase'])} 個支持上升、{int(sums['confirmed_decrease_uncertain'])} 個確認 MATCH 下降但淨差不定、{int(sums['confirmed_increase_uncertain'])} 個確認 MATCH 上升但淨差不定、{int(sums['confirmed_unchanged'])} 個確認 MATCH 不變。最後一類也可能仍有不確定性，並非證明真正符合數完全不變。沒有 post-selection 顯著性檢定，這些不是一般發生率估計。

主分析 68 個 clauses 中：1 下降（木椅）、1 上升（q440 `wood bar stools`，MATCH 9→12、tight [1,3]，Recall {bar['relevant_before']}/{bar['highest_n']}→{bar['relevant_after']}/{bar['highest_n']}，故不滿足 equal-Recall candidate）。另有 11 未定下降、14 未定上升、41 confirmed 不變；其中 {no_match} 個前後皆零確認 MATCH。全範圍通過完整證據的 MATCH 消失數為 {proven_vanish}，不能用零確認 MATCH 代替消失證明。

## 缺值與歧義

{missing_table}

`missingness_summary.csv` 列出全部 12 comparisons 的 full/Exact/joint、前後四狀態與分母。這些計數是 product-clause instances，可重複同商品；不偽裝成獨立樣本。每個 query 的具體缺值、變體和 scoped uncertainty 皆在完整 summary/evidence。

木椅最終 full 為 8/3/1/8→6/5/1/8；loose [-11,7]，抵消前後相同商品後 tight [-2,-2]。Exact 為 5/0/1/1→4/1/1/1；loose [-3,1]、tight [-1,-1]。共同的 1 UNKNOWN、8 AMBIGUOUS 使絕對符合數不確定，但都未進出，不能任意在兩邊換真值，因此不影響淨差。這依賴屬性/商品身分固定；全原始紀錄重數與非排序文字均核對。

白色 full 4/1/2/13→2/1/2/15，tight [-6,4]；羊毛 full 17/0/0/3→16/1/0/3，tight [-2,0]。二者都不能證實完整候選的真實淨損失；Exact 子集則各為空集合。三例 joint-MATCH 均不能證明全部 query 選項消失。

## 木椅線索的核對與更正

- H=42、Exact 7→7、Recall 7/42 完全重現。Joetta 26196 是 6→23；Vandergriff 36317 是 12→69；Jettie 26195 是 30→20；Lewallen 32126 是 38→17；四件都為 Exact，完整涵蓋 Exact 進出。
- nDCG {n(cases[367]['summary']['ndcg_before'])}→{n(cases[367]['summary']['ndcg_after'])}，Δ {n(cases[367]['summary']['delta_ndcg'])}；不能寫兩個 effectiveness metrics 不變。
- 全 Top20 另有 1817/3604 退出、35827/4355 進入；納入後木框架淨差仍 −2。新增 4355 是 Partial，其餘三件 unjudged。
- 完整描述審閱使共同的三件木欄位/鋁底座商品變成 AMBIGUOUS，所以初次 screen 的 11→9 修正為 **8→6**，而不是忽略冲突。Exact 的 5→4 不變。所有進出者對該框架子條件均已判清，淨差不因共同歧義而反轉。
- 名稱 wood 與塑膠框架不能互相代換；wood grain 是外觀，座面材質仍未核定。Lewallen 仍為官方 Exact，不改標籤、不宣稱是假標。
- 全部 8 進出商品的 16 序列為 **555–870 tokens**，包含 [CLS]/[SEP]，沒有一個 ≤512。不可使用 fully-fitting 說法。

完整逐件來源、全部留存者、所有進出者的 token 數及正反表請見 [wooden_chair_outdoor_audit.md](wooden_chair_outdoor_audit.md)。

## 設計凍結、修正與選擇透明度

這是 exploratory、非 preregistered；367/292 都是已知 seed。v1.0 到 v1.1 的 nominal-size / bamboo 修訂在任何本輪排名方向計算前，原檔位於 `design_history/`。已凍結 protocol/constraints/source SHA 持續檢查。

兩次執行語意修正發生在結果可见後，完整列於 `qa/semantic_corrections.json`：先修正類型分類不能把未命中 class 字串當成矛盾；再於三例完整描述審閱後，補足 collection 纖維變體及 structural-base 衝突的保守檢查。都全量重跑原 12 comparisons，沒有改動 query、support、K、field maps 或 schedule family。不能把最終所有 adjudication 稱為排名前完全固定。初次 screen 與修正前 ranking 有留存，可重建修訂歷程。

在第二次修正前，固定兩個 seed 與最高排序的另一 query 300，沒有取最大 rank drop。full-source audit 將 300 從 A 降 B 後，仍保留三例，不補換 q275/q417。`candidate_selection.csv` 保存全部 76 records 的排序欄位與未稽核理由；`selection.json` 鎖定三例。

## 模型、重現與尚缺的 artifacts

BGE `{protocol['execution_profiles'][-1]['model_revision']}` 的實際 pinned revision 以 protocol 對應列為準；BGE name `BAAI/bge-base-en-v1.5`，revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`，CLS、512 tokens、configured batch 48。MiniLM `sentence-transformers/all-MiniLM-L6-v2` revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，attention-mask mean、256、configured batch 128。tokenizer 與 model revision 相同，BertTokenizerFast；empty query prefix；所有 cached execution metadata 同為 phase2-encoder-v1、CUDA fp16 model/FP32 pooling、L2 normalize、FP32 dot product、stable original-catalog tie。當前 encoding.py 的 v3 實作與 cache 的 v1 不混稱為同次 forward。

實際 template：依原 section_order 非空段落以 newline 串接 title/class/category/description/pipe-joined raw attributes；C1 只反轉 entries。保存的 representation 檔案雜湊、解壓文字 hash、全部 42,994 IDs 次序與入選模板均核對。三例各 C0/C1 的 full-catalog cache 計分都重現 Top20 順序，歷史/new matched cached scoring 分表；沒有只在聯集重新排序。

所需排名、原始商品/query/qrels、14 product caches、兩個 query caches 與 pinned local tokenizers 都存在，沒有因此跳過任何預定 contrast。仍缺：歷史每個 batch 的實際分組與 padding/OOM subdivision、每 artifact GPU 型號及完整 forward telemetry；無新 matched encoder forward，所以這些不能補成已知。缺 unit/axis schema 的 numeric dimension 未做；部分來源本身少描述、少映射欄位或有變體冲突，明列 U/A。入選聯集 {known_ranks_missing} 件有至少一邊無保存的 exact outside rank，歷史欄只寫 >20；新 cache 計分提供分開的 exact ranks。未另行標註原始圖片、未用外部網頁補資料。没有預測 satisfaction/sales/exposure/agent success 的資料。

## 可支持的稿件措辭

建議主文若需一例，使用木椅的 **native pipeline / recorded frame clause**，並保留截斷限制；若段落主張完整輸入的純位置效應，這例不適用。它支持：在 catalog-wide attribute reversal 下，benchmark-label Recall 相同可以伴隨有明確記錄的木框架候選減少；這個例子的 nDCG 有小幅下降。它不支持 preference/utility/satisfaction 已下降，也不支持一個商品直接因果排擠另一個、官方 qrel 已被證偽或兩項指標皆無變化。

`paper_case_table.tex` 列出所有四件 relevant 進出與兩種集合的完整四狀態分母；`RQ2_illustrative_case.txt` 提供 80–120-word 英文草稿。都留在本分析目錄，沒有編輯 manuscript。

## 驗證與交付

獨立於 core 的 `verify.py` 已重算 {qa['query_comparisons']} 組集合/指標、{qa['assessed_clause_comparisons']} 組子條件 full/Exact/joint bounds，驗證原始 qrels/query、來源 SHA、三例重現與 token 一致性。14 個語意測試涵蓋缺值、複合/變體、共同未知抵消、full vs Exact、scope、>20 與描述衝突。{qa['protected_files']} 個既有檔案（含 manuscript、舊 analysis 和已修改檔）SHA 未變。算術驗證不是第二位人工 annotator；保守字面映射可能漏掉可由更深入理解支持的 MATCH。

一命令與環境依賴見 [README.md](README.md)；資料/執行設定見 `input_manifest.json`、`protocol.json`、`qa/cache_tokenizer_manifest.json`、`qa/reproduction_runtime.json`；可核對的最終數據在 `query_comparison_summary.csv`、`comparison_summary.csv`、`candidate_evidence.jsonl`、`catalog_evidence.jsonl` 與 `top_cases/`。沒有新 canonical、hybrid、ESCI、模型、K 或額外 schedule；完成既定搜尋與三例後停止。
"""
    # Avoid a derived model-order reference in prose: revisions are named explicitly.
    doc=doc.replace(f"BGE `{protocol['execution_profiles'][-1]['model_revision']}` 的實際 pinned revision 以 protocol 對應列為準；BGE name",'BGE name')
    write('REPORT.md',doc)
    draft="""In an exploratory WANDS example, catalog-wide attribute reversal for “wooden chair outdoor” preserved Recall@20 computed from benchmark relevance labels (7/42), while nDCG@20 decreased from 0.43436 to 0.42901. Under conservative recorded frame-material adjudication, confirmed wooden-frame products fell from eight to six in the full Top-20 and from five to four among Exact products. Shared uncertain items cancel, fixing these differences at −2 and −1. The entering plastic/resin-frame chair is nevertheless labeled Exact, suggesting possible label granularity or annotation mismatch, not a proven labeling error. All changed products exceed BGE’s native token limit; this illustrates the truncated retrieval pipeline, not a fully-fitting order effect or measured user utility."""
    assert 80<=len(draft.split())<=120,len(draft.split())
    write('RQ2_illustrative_case.txt',draft)
    write('paper_case_table.tex',r"""% Standalone table fragment; requires booktabs in the eventual manuscript.
% Kept in analysis directory: manuscript is not edited.
\begin{table*}[t]
\centering
\small
\begin{tabular}{llrrl}
\toprule
Product ID & Name & C0 rank & C1 rank & Recorded frame \\
\midrule
26196 & Joetta & 6 & 23 & Solid wood (acacia) \\
36317 & Vandergriff & 12 & 69 & Solid wood (mahogany) \\
26195 & Jettie & 30 & 20 & Solid wood (acacia) \\
32126 & Lewallen & 38 & 17 & Plastic/resin \\
\midrule
\multicolumn{5}{l}{Full Top-20 M/C/U/A: $8/3/1/8 \to 6/5/1/8$} \\
\multicolumn{5}{l}{Exact subset M/C/U/A: $5/0/1/1 \to 4/1/1/1$} \\
\bottomrule
\end{tabular}
\caption{Exploratory WANDS/BGE native, catalog-wide C0-to-C1 case for
\texttt{wooden chair outdoor}. All four Exact-labeled membership changes
are shown, including the plastic/resin-frame item. There are four total
departures and four arrivals; the remaining changes are documented in the
analysis. M/C/U/A denote recorded frame MATCH/CONTRADICTION/UNKNOWN/AMBIGUOUS.
Recall remains $7/42$; nDCG changes $0.43436\to0.42901$.
After cancelling shared uncertain items, full-set and Exact-subset
differences are $-2$ and $-1$. All eight changed products exceed the
512-token limit. No claim of whole-query satisfaction or direct displacement.}
\label{tab:query-constraint-illustration}
\end{table*}""")
    req=read('query_constraint_cases_20261007/input_manifest.json')['packages']
    write('requirements.txt','\n'.join(f'{k}=={v}' for k,v in req.items() if k in ['numpy','pandas','pyarrow','transformers','tokenizers']))
    readme=f"""# Query constraint case analysis (2026-10-07)

從 repository root 執行一個命令：

```powershell
python query_constraint_cases_20261007/run.py
```

依序保留/驗證 frozen protocol → semantic tests → 固定 12 contrasts → 固定 3 cases 的 token/cache audit → independent verification → report/table/draft → output hashes。只寫本目錄，不訓練、不 download、不改 query/模型/support、不改 manuscript 或舊 option analysis。使用者後續已明確授權完成後 commit/push；runner 本身不執行 Git 寫入。

初始來源 HEAD `{protocol['actual_starting_commit']}`；已核對 historical reference `{protocol['historical_reference_commit']}` 的 tree 相同。後續 checkout commit 會變，但 frozen input SHA 必須仍匹配。`prepare.py` 不會根據結果重建 protocol。selection 同樣不可在修正後改選；已完整稽核 367、292、300，停止擴張。

環境 Python {sys.version.split()[0] if 'sys' in globals() else read('query_constraint_cases_20261007/input_manifest.json')['python'].split()[0]}；套件精確版本見 `requirements.txt` / `input_manifest.json`。不需 GPU；cache rescoring 用 CPU numpy FP32。需已有本地 pinned Hugging Face tokenizer snapshots；使用 local_files_only 且 HF_HUB_OFFLINE。tokenizer 會提示 uncapped 序列超過 max length，這是稽核刻意計數，沒有將過長序列送入 model forward。

必要本地 artifacts：

- 原始 `data/raw/product.csv`, `query.csv`, `label.csv`；processed WANDS products/queries/judgments。
- `phase4/config/splits.json`、`revision_graded_20260922/PROTOCOL.json` 與 per-query metrics；已有 option_coverage provenance / validated lists。
- 兩模型 × C0/C1/五 shuffles 的 top50 與 pair-rank parquet、metadata；selected C0/C1 product/query `.npy` payloads。
- 保存的 C0/C1 `phase2/data/representations/wands/*.jsonl.gz`；BGE/MiniLM pinned tokenizers。

完整確切路徑、bytes、SHA 位於 `input_manifest.json`、`qa/cache_tokenizer_manifest.json`。大 cache/data 在原 repo 的忽略範圍，這次不重新 push 原始大型資料；fresh clone 若沒有這些既有本地 artifacts，不能僅憑新分析目錄重跑來源核對。可直接讀已保存 CSV/JSONL/報告。缺資料時 runner fail-fast，不改用別的設定。當前本機所需 artifacts 全在；缺失的 per-batch/GPU/forward telemetry 與不可靠 numeric units 在 REPORT 明列。

實際最終執行命令、exit code 與 stdout/stderr 會保存 `qa/execution_log.json` / `qa/run_logs/`。準備/修正過程見 `qa/semantic_corrections.json` 與 `design_history/`，不是 preregistration。起始 dirty status 与 165 個保護檔 hash 見 `qa/preexisting_state.json`；任何既有資料或稿件變化都會使驗證失敗，不會還原/覆寫使用者檔案。

交付重點：

- `REPORT.md`：中文結論、正向/零/相反結果、分母、缺值與限制。
- `wooden_chair_outdoor_audit.md` / `top_cases/*/AUDIT.md`：來源逐件審閱、歷史與 matched-cache 分表、全進出與 token。
- `query_constraints.csv` / `protocol.json`：308 queries 與凍結條件；`query_comparison_summary.csv` 保留每組各方向。
- `candidate_evidence.jsonl`：4493 份 query-product-clause evidence，含 scope、qrel、rank、原始對應欄位與完整描述；`catalog_evidence.jsonl` 去重存原始全欄位，以 product_id 連結。
- `candidate_ranking.csv` / `candidate_selection.csv`：全部候選及未稽核原因。screen grade 不等於所有案例人工驗證。
- `missingness_summary.csv`：所有集合的 M/C/U/A 分母。
- `paper_case_table.tex`：表格 fragment；`RQ2_illustrative_case.txt`：{len(draft.split())} words，未置入論文。
- `qa/independent_verification.json` / `qa/output_manifest.json`：檢核結果與輸出 hash。

本任務不要求新增大型 figure；以可逐項核對的表格呈現，沒有修改或新增論文圖。
"""
    write('README.md',readme)
    dump('qa/missing_artifacts.json',dict(required_ranking_data_missing=[],required_selected_caches_tokenizers_missing=[],
        unknown_historical_metadata=['per-batch actual grouping and padding','OOM subdivisions','per-artifact GPU model','complete model-forward telemetry'],
        unavailable_semantic_schema=['reliable global units and axes for numeric dimensions'],
        not_performed=['fresh matched encoder forward (comparable cached vectors reused)','external product-page/image adjudication','fourth detailed case after source-audit downgrade'],
        missingness_table='missingness_summary.csv',historical_union_items_with_unknown_exact_outside_rank=known_ranks_missing))
    print('Reports, all three case audits, LaTeX table fragment and',len(draft.split()),'word English draft written.')

if __name__=='__main__':main()
