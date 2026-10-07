# Case audit

All cases use WANDS / BGE native / catalog-wide C0 versus C1 / K=20 / Exact judgments. Additional cases use the frozen query-ID hash and are field-specific. They were not chosen for maximum effect. Full source attributes, all relevant candidate IDs and ranks are in `data/cases.jsonl`; candidate pools and hash selection are in `data/case_selection.json`.

## Query 231: anti fatigue mat

Selection: existing_anti_fatigue_mat; illustrating field: **shape**.

Relevant candidates 20 -> 20; lost/gained products 5/5; Exact pool 307. Recall@20 0.065146580 -> 0.065146580. Full-ranking nDCG@20 1.000000000 -> 1.000000000 (3/1/0 gains; equality is reported only where observed).

Before IDs: 19170, 20172, 20175, 20185, 25796, 27622, 7798, 7801, 7802, 7803, 7805, 7809, 7810, 7811, 7813, 7815, 7816, 7817, 7818, 7842.

After IDs: 20172, 25796, 27622, 37018, 5540, 5543, 7798, 7799, 7801, 7802, 7803, 7805, 7808, 7809, 7810, 7811, 7815, 7816, 7817, 7842.

Departing IDs: 19170, 20175, 20185, 7813, 7818. Entering IDs: 37018, 5540, 5543, 7799, 7808.

| Field | Strict eligible | Before options | After options | Lost | Gained |
|---|---|---|---|---|---|
| material | yes | foam, memory foam, plastic, rubber, synthetics | foam, memory foam, plastic, rubber, synthetics | none | none |
| shape | yes | rectangle, semi-circle | rectangle | semi-circle | none |
| color | no: conflicting_values, missing_key | unknown | unknown | not assessed | not assessed |

| Product ID | Source title | C0 rank | C1 rank | Status |
|---|---|---:|---:|---|
| 19170 | anette salon anti-fatigue mat | 20 | 32 | departing |
| 20172 | theta anti-fatigue mat | 7 | 11 | retained |
| 20175 | wausau anti-fatigue mat | 19 | 29 | departing |
| 20185 | abtal anti-fatigue mat | 17 | 28 | departing |
| 25796 | rozella anti-fatigue mat | 14 | 16 | retained |
| 27622 | comfort floor anti-fatigue mat | 1 | 1 | retained |
| 37018 | thurber anti-fatigue mat | 37 | 12 | entering |
| 5540 | kyng lemon drop anti-fatigue mat | 26 | 14 | entering |
| 5543 | sirna anti-fatigue mat | 44 | 15 | entering |
| 7798 | rutter life anti-fatigue mat | 4 | 3 | retained |
| 7799 | ruvalcaba sea life serenade anti-fatigue mat | 32 | 20 | entering |
| 7801 | sadye oceana anti-fatigue mat | 12 | 13 | retained |
| 7802 | salomon ocean anti-fatigue mat | 16 | 17 | retained |
| 7803 | salsbury tide pool shells anti-fatigue mat | 11 | 18 | retained |
| 7805 | salamone living on beach time anti-fatigue mat | 13 | 6 | retained |
| 7808 | lee good to be home anti-fatigue mat | 22 | 19 | entering |
| 7809 | kole family time anti-fatigue mat | 9 | 9 | retained |
| 7810 | crowl cotton boll anti-fatigue mat | 3 | 8 | retained |
| 7811 | cline bless this home anti-fatigue mat | 2 | 2 | retained |
| 7813 | nicholson sunflowers anti-fatigue mat | 18 | 21 | departing |
| 7815 | kerry fluer de lis anti-fatigue mat | 5 | 10 | retained |
| 7816 | mccart anti-fatigue mat | 6 | 5 | retained |
| 7817 | tressa boho tile anti-fatigue mat | 8 | 7 | retained |
| 7818 | mccardle morroccan tiles anti-fatigue mat | 10 | 26 | departing |
| 7842 | bohemian anti-fatigue mat | 15 | 4 | retained |

The departing Anette Salon product is ID 19170, recorded as `material : foam`, `shape : semi-circle`, `color : black`. The full relevant Top-20 comparison verifies loss of the recorded **semi-circle shape**: all after-products have valid shape metadata and none supplies it. Foam remains supplied by 27622, 5540, 5543, 7808, 7811. Black remains explicitly supplied by 37018.

Color is strictly ineligible: ID 20172 has conflicting colors, and departing ID 20175 has no generic color key. The known black suppliers establish that black is still represented, but incomplete color metadata do not establish any complete color-set comparison. Both full nDCG@20 values are 1 (within floating representation) because all 20 retrieved candidates are Exact; this is a verified fact about this case, not a cancellation requirement.

Thus the existing two-row substitution illustration alone was insufficient, but this full-list audit supports a narrow recorded-shape disappearance for query 231. It does not establish a utility loss, physical preference satisfaction, or loss of foam/black options.

## Query 341: canvas map art

Selection: additional_cancellation_option_loss; illustrating field: **color**.

Relevant candidates 16 -> 16; lost/gained products 1/1; Exact pool 22. Recall@20 0.727272727 -> 0.727272727. Full-ranking nDCG@20 0.895476621 -> 0.896298862 (3/1/0 gains; equality is reported only where observed).

Before IDs: 12198, 13582, 17378, 19427, 25684, 28263, 31876, 36931, 38961, 39147, 41681, 41794, 4181, 4360, 729, 9573.

After IDs: 12198, 13582, 13583, 17378, 19427, 25684, 28263, 31876, 36931, 38961, 39147, 41681, 41794, 4181, 729, 9573.

Departing IDs: 4360. Entering IDs: 13583.

| Field | Strict eligible | Before options | After options | Lost | Gained |
|---|---|---|---|---|---|
| material | no: missing_key | unknown | unknown | not assessed | not assessed |
| shape | no: missing_key | unknown | unknown | not assessed | not assessed |
| color | yes | black, blue, blue , white , brown , red , gray, blue/purple/orange, brown/green/yellow, gold/white, gray, multicolor, pink, rustic/black, silver/olivine/turquoise, white ; santa fe ; valencia ; red ; tan, white/gray/red | black, blue, blue , white , brown , red , gray, blue/purple/orange, brown/green/yellow, gold/pink, gold/white, gray, multicolor, pink, silver/olivine/turquoise, white ; santa fe ; valencia ; red ; tan, white/gray/red | rustic/black | gold/pink |

| Product ID | Source title | C0 rank | C1 rank | Status |
|---|---|---:|---:|---|
| 12198 | madallions map - wrapped canvas graphic art print | 12 | 11 | retained |
| 13582 | world map 2 canvas art | 19 | 16 | retained |
| 13583 | world map 4 canvas art | 21 | 20 | entering |
| 17378 | historix 1881 oahu hawaii vintage map - 24x36 inch vintage map of oahu hawaii wall art - map of hawaii oahu poster - survey of oahu hawaiian islands - old map oahu - historic oahu print ( 2 sizes ) | 15 | 14 | retained |
| 19427 | watercolour map of the world framed graphic art print on canvas | 11 | 9 | retained |
| 25684 | ocean eye i - 3 piece picture frame print set on canvas | 20 | 17 | retained |
| 28263 | world map - wrapped canvas painting print | 5 | 2 | retained |
| 31876 | usa map 2 - graphic art print on canvas | 7 | 6 | retained |
| 36931 | world map on wood by jamie macdowell - wrapped canvas print | 3 | 3 | retained |
| 38961 | world map painting print on wrapped canvas | 1 | 1 | retained |
| 39147 | `` champaign gold map '' graphic art on wrapped canvas | 10 | 10 | retained |
| 41681 | textural world map - multi-piece image print on canvas | 9 | 7 | retained |
| 41794 | world map - wrapped canvas graphic art print | 4 | 4 | retained |
| 4181 | colorful world map - graphic art print on canvas | 2 | 5 | retained |
| 4360 | world map by fireside home - picture frame graphic art print on wood | 13 | 36 | departing |
| 729 | albania country in the balkans on world map - wrapped canvas graphic art print | 6 | 12 | retained |
| 9573 | classic world map - wrapped canvas graphic art print | 8 | 8 | retained |

## Query 453: midcentury tv unit

Selection: additional_cancellation_no_field_option_change; illustrating field: **material**.

Relevant candidates 5 -> 5; lost/gained products 1/1; Exact pool 25. Recall@20 0.200000000 -> 0.200000000. Full-ranking nDCG@20 0.390181570 -> 0.361287096 (3/1/0 gains; equality is reported only where observed).

Before IDs: 16673, 16734, 37160, 37161, 39866.

After IDs: 16734, 16736, 37160, 37161, 39866.

Departing IDs: 16673. Entering IDs: 16736.

| Field | Strict eligible | Before options | After options | Lost | Gained |
|---|---|---|---|---|---|
| material | yes | manufactured wood, solid + manufactured wood | manufactured wood, solid + manufactured wood | none | none |
| shape | no: missing_key | unknown | unknown | not assessed | not assessed |
| color | no: conflicting_values, missing_key | unknown | unknown | not assessed | not assessed |

| Product ID | Source title | C0 rank | C1 rank | Status |
|---|---|---:|---:|---|
| 16673 | allegra tv stand for tvs up to 60 '' | 14 | 25 | departing |
| 16734 | oglethorpe tv stand for tvs up to 70 '' | 6 | 4 | retained |
| 16736 | oglethorpe tv stand for tvs up to 49 '' | 25 | 16 | entering |
| 37160 | george oliver mid-century tv stand with 2 storage shelves & door for living room , tv console cabinet , retro entertainment center for flat screen tv cable box gaming consoles | 1 | 1 | retained |
| 37161 | george oliver mid-century tv stand with 2 storage shelves & door for christmas gift , tv console cabinet , retro entertainment center for flat screen tv cable box gaming consoles , rustic brown | 2 | 5 | retained |
| 39866 | glenn tv stand for tvs up to 65 '' | 19 | 19 | retained |

