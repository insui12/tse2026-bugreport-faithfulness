# TSE Supplementary Analysis — Deliverables

Generated: 2026-04-04

The accompanying analysis scripts live at `../../code/analysis/` (one level up from
`results/`); only their pre-computed outputs are stored here.

## Structure

```
results/analysis/
├── s1_contrast/                    # 필수 1: Per-instance + Contrast Statistics
│   ├── per_instance_all.csv.gz         16,813 rows (391 test × 43 conditions)
│   ├── contrast_paired_stats.csv       728 tests (52 contrasts × 14 metrics)
│   ├── contrast_paired_stats_full.csv  1,196 tests (+9 metrics, +BH FDR q)
│   └── direction_consistency.csv       345 contrast-groups, 64.1% consistent
│
├── s2_entity_validation/           # 필수 2: Hard-entity Extraction Validation
│   ├── entity_level_annotA.csv         439 entities, Annotator A labeled
│   ├── entity_level_blank.csv          same 439 entities, blank for Annotator B
│   ├── entity_extraction_stats.csv     type-level aggregate stats
│   └── annotation_sheet.csv            100-row legacy sheet (row-level)
│
├── s3_rq2_tau/                     # 필수 3: RQ2 Kendall tau-b
│   ├── rq2_tau_with_bh.csv             108 tests, Holm + BH FDR dual correction
│   ├── rq2_tau_per_instance.csv        108 tests (original, Holm only)
│   └── rq2_tau_summary.csv             cross-model aggregation
│
└── s4_mining_audit/                # 필수 4: DPO Pair-mining Audit
    ├── mining_audit_summary.csv        8 adapters: pairs, pass-rate, reject reasons
    ├── candidate_pool_stats.csv        8 pools: score distribution
    └── threshold_sensitivity.csv       7,200 rows (8 pools × 900 threshold combos)
```

## Key Findings

### S1: 728 paired contrasts
- 404/728 (55.5%) significant at Holm q < 0.05
- 66.6% of significant tests have large effect (|r| >= 0.5)
- 38.1% of 3-model groups show direction inconsistency (model-dependent effects)

### S2: Annotator A extraction precision
- URL: 100%, FILE: 100%, FLAG: 94.4%, HASH: 86.2%, Overall: 92.9%
- HASH FP: 20/20 are Gecko epoch constant "20100101"
- FLAG FP: 11/11 are abbreviations (e.g., i.e.) or too-short tokens
- FP impact on UAHE: 9/180 = 5.0% overestimation

### S3: RQ2 tau-b
- k -> UAHE_per_1kTok: tau = -0.32, all 3 models significant (RAG reduces hallucination)
- k -> ContextSupportRate: tau = +0.38, all 3 models significant (RAG improves grounding)
- k -> ROUGE/SBERT/CTQRS: not significant (RAG does not improve content similarity)

### S4: Mining audit
- hard profile: 2.3% pass rate (135 pairs) — severe data scarcity
- balanced profile: 44-60% pass rate
- Gecko epoch "20100101" inflates RAHE counts in retrieval conditions

## Remaining Human Work
- Annotator B: fill entity_level_blank.csv independently
- Compute inter-annotator Cohen's kappa from A vs B labels
