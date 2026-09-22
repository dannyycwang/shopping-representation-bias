| Dataset / encoder / rule | Dense lost | gained | crossing in | out | Hybrid lost | gained | crossing in | out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wands / minilm / asc | 0.23 | 0.43 | 7.57 | 12.01 | 0.21 | 0.42 | 6.77 | 6.93 |
| wands / minilm / desc | 0.42 | 0.60 | 8.43 | 11.15 | 0.27 | 0.50 | 6.48 | 7.23 |
| wands / minilm / field | 0.63 | 0.57 | 8.33 | 11.25 | 0.13 | 0.46 | 6.78 | 6.92 |
| wands / bge_base / asc | 0.17 | 0.40 | 5.37 | 6.56 | 0.09 | 0.13 | 3.11 | 4.41 |
| wands / bge_base / desc | 0.20 | 0.22 | 5.79 | 6.15 | 0.14 | 0.34 | 3.88 | 3.65 |
| wands / bge_base / field | 0.18 | 0.48 | 5.70 | 6.23 | 0.13 | 0.12 | 3.33 | 4.19 |
| esci / minilm / asc | 0.03 | 0.01 | 2.04 | 2.00 | 0.03 | 0.01 | 1.65 | 1.62 |
| esci / minilm / desc | 0.02 | 0.04 | 2.03 | 2.00 | 0.02 | 0.06 | 1.67 | 1.60 |
| esci / minilm / field | 0.03 | 0.02 | 2.02 | 2.02 | 0.03 | 0.01 | 1.66 | 1.60 |
| esci / bge_base / asc | 0.09 | 0.08 | 1.95 | 1.71 | 0.06 | 0.03 | 1.34 | 0.97 |
| esci / bge_base / desc | 0.13 | 0.08 | 1.75 | 1.91 | 0.01 | 0.01 | 1.08 | 1.23 |
| esci / bge_base / field | 0.09 | 0.08 | 1.93 | 1.73 | 0.06 | 0.03 | 1.36 | 0.94 |

All cells are percentage points of highest-label support, query macro, K=20. Lost = reference always-in but fixed-control omitted; gained = reference always-out but fixed-control included. Crossing columns partition reference crossing mass. Dense and hybrid use their respective raw catalog-wide seven-schedule references. Full six-cell tables, denominators, conditional rates and uncorrected query-cluster intervals are in control_transitions.csv.
