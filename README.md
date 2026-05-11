# Replication Package

**Hallucination, Context Faithfulness, and Template Fidelity in Bug Report Generation with Open-Source LLMs: A Comparative Study of Prompting, Retrieval Augmentation, and DPO Fine-Tuning**

## Directory Structure

```
replication_package/
├── README.md
├── requirements.txt
├── code/
│   ├── bugreport_pipeline_tse.py    # Main generation & evaluation pipeline
│   ├── remine_dpo_pairs.py          # DPO preference pair mining
│   ├── downsample.py                # Data downsampling utility
│   ├── analysis/                    # 8 analysis scripts
│   │   ├── s1_contrast_stats.py     # Paired bootstrap contrast tests
│   │   ├── s1b_supplement_stats.py  # Supplementary statistics
│   │   ├── s2_hard_entity_sheet.py  # Hard entity extraction validation
│   │   ├── s2b_entity_level_sheet.py
│   │   ├── s2c_refine_annotA.py
│   │   ├── s2d_compute_agreement.py # Inter-annotator agreement
│   │   ├── s3_rq2_tau.py           # Kendall tau-b for RQ2
│   │   └── s4_mining_audit.py       # DPO mining audit
│   └── evaluation/                  # CTQRS evaluation module
│       ├── perfect_ctqrs.py
│       └── *.csv                    # Lookup tables
├── data/
│   ├── splits/                      # Train/val/test splits (seed=42)
│   │   ├── train_seed42.csv
│   │   ├── val_seed42.csv
│   │   ├── test_seed42.csv
│   │   └── split_seed42.meta.json   # Split metadata
│   └── dpo_pairs/                   # DPO preference pairs (balanced profile)
│       └── *.jsonl                  # 6 files (3 models × aware/free)
├── results/
│   ├── summary/
│   │   └── INTEGRATED_summary.csv   # 43 conditions, all metrics
│   └── analysis/
│       ├── README.md
│       ├── s1_contrast/             # 728 paired bootstrap contrasts
│       ├── s2_entity_validation/    # Entity extraction precision
│       ├── s3_rq2_tau/              # Kendall tau-b (k effect)
│       └── s4_mining_audit/         # DPO pair mining audit
├── scripts/                         # Execution shell scripts (hyperparameters inline)
│   ├── run_all.sh                   # End-to-end pipeline (data → SFT → eval)
│   ├── run_all_v2.sh                # Updated pipeline driver
│   ├── run_after_sft.sh             # DPO training + post-SFT evaluation
│   └── run_remine_dpo_pairs.sh      # DPO preference-pair mining (balanced/structure/hard profiles)
└── adapters/                        # (not redistributed — see "LoRA Adapters" below)
```

## Data Source

The dataset is taken directly from:
- Acharya and Ginde, "Can we enhance bug report quality using LLMs?", EASE 2025
- Repository: https://github.com/GindeLab/Ease_2025_AI_model

## Base Models

Base model weights are not redistributed. All three are the instruction-tuned variants
used throughout the paper; obtain from Hugging Face:
- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `meta-llama/Llama-3.2-3B-Instruct` (gated)

## LoRA Adapters

Trained LoRA adapter weights (SFT and DPO) are available from the authors upon request.

## Environment

- Python 3.10.19
- CUDA: 13.0 (Driver 580.159.03)
- GPU: NVIDIA RTX PRO 6000 Blackwell (97 GB VRAM)
- Framework: Unsloth + Hugging Face Transformers + TRL
- OS: Linux (GCC 11.2.0)

## How to Reproduce

### What is reproducible without a GPU

All of the **aggregated numbers reported in the paper** are shipped in:
- `results/summary/INTEGRATED_summary.csv` — 43 conditions × all metrics with bootstrap CIs.
- `results/analysis/s1_contrast/` — 728 paired-contrast tests with Holm/BH-FDR q-values
  (plus `per_instance_all.csv.gz`, the long-format 16,813-row matrix used to derive them).
- `results/analysis/s2_entity_validation/` — Annotator-A entity labels and type-level stats.
- `results/analysis/s3_rq2_tau/` — Kendall τ-b per RQ2 (per-instance + summary + BH-FDR).
- `results/analysis/s4_mining_audit/` — DPO pair-mining audit and threshold sensitivity.

These files can be opened directly with pandas/Excel; no execution is required. The full
test set (`data/splits/test_seed42.csv`) and DPO preference pairs
(`data/dpo_pairs/*.jsonl`) are also shipped for direct inspection.

### What requires re-running the GPU pipeline

The analysis scripts in `code/analysis/` consume per-instance `scored_*.jsonl.gz` files,
which are intermediate artifacts produced by the `score` step of the GPU pipeline and are
**not redistributed in this package** due to size. To regenerate them from raw splits:

```bash
# 1. Uncomment the ML dependencies in requirements.txt and install
# 2. Download base models from Hugging Face (see "Base Models" above)
# 3. Obtain LoRA adapters from the authors and place under adapters/
# 4. Run the end-to-end driver (data prep → SFT → DPO → gen → score → aggregate)
bash scripts/run_all.sh
# 5. After SFT-only runs, continue with DPO pair remining and post-SFT evaluation
bash scripts/run_remine_dpo_pairs.sh
bash scripts/run_after_sft.sh
# 6. The analysis scripts can then be re-run against the freshly produced scored dir
python code/analysis/s1_contrast_stats.py \
    --scored-dir results/runs/<timestamp>_<tag>_seed42/scored \
    --out-dir    results/analysis/s1_contrast_rerun
```

Scripts `s2c_refine_annotA.py` and `s2d_compute_agreement.py` operate on shipped
annotator-level files and can be invoked without the GPU pipeline; consult each script's
`--help` for the exact arguments.

## License

- **Code** (this replication package): MIT License.
- **Data splits** (`data/splits/*.csv`): derived by re-splitting the dataset distributed by
  Acharya & Ginde [5] at https://github.com/GindeLab/Ease_2025_AI_model. Their README declares
  the project under the MIT License, but at the time of this submission their repository does
  not contain an SPDX-identifiable LICENSE file; redistribution terms should therefore be
  confirmed with the original authors. The underlying bug reports were originally collected
  from Mozilla Bugzilla, which is distributed under the Mozilla Public License; any
  re-use of the underlying report text should additionally respect MPL terms.
- **DPO preference pairs** (`data/dpo_pairs/*.jsonl`): newly mined in this study from the
  splits above; released under MIT alongside the code.
- **Trained LoRA adapter weights** (SFT and DPO): not redistributed in this package;
  available from the authors upon request.
