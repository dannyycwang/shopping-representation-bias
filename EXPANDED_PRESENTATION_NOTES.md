# 擴充呈現版本

主文件：`paper_www2027/main.pdf`，內文 10 頁，參考文獻 1 頁，共 11 頁。詳細附錄另編成 `paper_www2027/supplementary.pdf`，不計入主 PDF。

本次沒有新增實驗、模型推論或資料標註，也沒有改動原有 Phase II/III 與 target-only 數值。主要新增內容如下。

- **更完整的方法解說**：VI 的含義與限制、micro/macro 分母差異、Top-K 並非累積失敗率、target-only 與 catalog-wide 的不同因果問題，以及不變性方法可能犧牲跨屬性互動的原因。
- **更多既有數據**：encoding controls 表、turquoise chair 全部七種順序的排名與 C0-relative delta、候選池 coverage 表，以及 C0 reranking 前後的 recall 點估計。
- **新增兩張圖**：target-only 在 Top-10/20/50 的變化；所有 mitigation 的 paired effectiveness CI，包括 late-aggregation 的負面結果。
- **更完整的解讀**：說明 WANDS 與 ESCI 的差異、common-pool reranking 的條件限制、不能將不顯著當成等價，以及如何設計 structured Web retrieval 的 regression evaluation。
- **保留第一頁設計**：Web-level abstract 與 target-only Figure 1 保留；Section 7.2 維持 Scope of the Visibility Claim。

主文目前有 6 張圖與 5 張表。舊的完整 catalog-wide sensitivity 表仍放在補充文件，沒有重新加入與 Figure 2 重複的主文表格。新增數字與圖表來源已寫入 `paper_www2027/RESULT_PROVENANCE.md`。

這是依照指定閱讀篇幅整理的版本，不表示已核對 WWW 2027 最終投稿頁數規定。先前修訂紀錄中的「8 頁」描述屬於上一版；本文件記錄目前的 11 頁擴充版本。既有 citation metadata 與 conference-header placeholders 仍需在投稿前做最後整理。
