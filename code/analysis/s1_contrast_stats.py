#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s1_contrast_stats.py  —  필수 1: Per-instance export + contrast-level supplementary statistics

Outputs:
  1) per_instance_all.csv.gz        — 391 rows × 43 conditions (long format)
  2) contrast_paired_stats.csv      — mean_A, mean_B, paired_delta, bootstrap CI,
                                       Wilcoxon p, Holm q, rank-biserial r, n_total, n_nonzero
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ────────────────────────────────────────────
# Config
# ────────────────────────────────────────────

COND_COLS = [
    "model", "adapter_type", "pref_profile", "align_context",
    "retriever_type", "retriever_mode", "k", "template", "retrieval_mask",
]

TARGET_METRICS = [
    "ROUGE1_R", "ROUGE1_F1", "SBERT", "CTQRS",
    "ROUGE1_R_NORM", "ROUGE1_F1_NORM", "SBERT_NORM",
    "SecFilled", "SecPresence", "CondFilledRate",
    "UAHE_per_1kTok", "UAHE_Total",
    "ContextSupportRate", "ContextUnattributedRate",
]

BOOT_ITERS = 10000
CI_ALPHA = 0.05
RNG_SEED = 42

# ────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────

def load_scored_dir(scored_dir: Path) -> pd.DataFrame:
    """Load all scored JSONL.gz into one long-format DataFrame."""
    frames = []
    for p in sorted(scored_dir.glob("scored_*.jsonl.gz")):
        rows = []
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        sys.exit(f"[ERROR] No scored JSONL files found in {scored_dir}")
    df = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(df)} rows from {len(frames)} scored files")
    return df


def cond_key(row: pd.Series) -> str:
    parts = [str(row[c]) for c in COND_COLS]
    return "|".join(parts)

# ────────────────────────────────────────────
# Contrast definitions
# ────────────────────────────────────────────

def _m(model, adapter, profile, align, rtype, rmode, k, template, mask):
    """Helper: build condition dict."""
    return dict(model=model, adapter_type=adapter, pref_profile=profile,
                align_context=align, retriever_type=rtype,
                retriever_mode=rmode, k=k, template=template,
                retrieval_mask=mask)


def build_contrasts(models: Sequence[str]) -> List[Dict[str, Any]]:
    """Return list of {name, cond_A, cond_B} dicts for all paper contrasts."""
    cs = []

    for m in models:
        # --- Base → SFT ---
        cs.append(dict(name=f"Base→SFT|k0|{m}",
            cond_A=_m(m,"base","none","none","none","none",0,"on","none"),
            cond_B=_m(m,"sft","none","none","none","none",0,"on","none")))
        cs.append(dict(name=f"Base→SFT|dense_k1|{m}",
            cond_A=_m(m,"base","none","none","dense","similar",1,"on","none"),
            cond_B=_m(m,"sft","none","none","dense","similar",1,"on","none")))

        # --- SFT → DPO (balanced, free) ---
        cs.append(dict(name=f"SFT→DPO_free_bal|k0|{m}",
            cond_A=_m(m,"sft","none","none","none","none",0,"on","none"),
            cond_B=_m(m,"dpo","balanced","retrieval_free","none","none",0,"on","none")))
        cs.append(dict(name=f"SFT→DPO_free_bal|dense_k1|{m}",
            cond_A=_m(m,"sft","none","none","dense","similar",1,"on","none"),
            cond_B=_m(m,"dpo","balanced","retrieval_free","dense","similar",1,"on","none")))

        # --- SFT → DPO (balanced, aware) ---
        cs.append(dict(name=f"SFT→DPO_aware_bal|k0|{m}",
            cond_A=_m(m,"sft","none","none","none","none",0,"on","none"),
            cond_B=_m(m,"dpo","balanced","retrieval_aware","none","none",0,"on","none")))
        cs.append(dict(name=f"SFT→DPO_aware_bal|dense_k1|{m}",
            cond_A=_m(m,"sft","none","none","dense","similar",1,"on","none"),
            cond_B=_m(m,"dpo","balanced","retrieval_aware","dense","similar",1,"on","none")))

        # --- RAG effect (k=0 → k=1) ---
        for adp, prof, alc in [("sft","none","none"),
                                ("dpo","balanced","retrieval_free"),
                                ("dpo","balanced","retrieval_aware")]:
            label = f"{adp}" if adp == "sft" else f"dpo_{alc[:4]}_bal"
            cs.append(dict(name=f"NoRAG→RAG|{label}|{m}",
                cond_A=_m(m,adp,prof,alc,"none","none",0,"on","none"),
                cond_B=_m(m,adp,prof,alc,"dense","similar",1,"on","none")))

        # --- k=1 → k=2 (SFT only) ---
        cs.append(dict(name=f"k1→k2|sft|{m}",
            cond_A=_m(m,"sft","none","none","dense","similar",1,"on","none"),
            cond_B=_m(m,"sft","none","none","dense","similar",2,"on","none")))

        # --- Dense vs Lexical (SFT, k=1) ---
        cs.append(dict(name=f"Dense→Lex|sft_k1|{m}",
            cond_A=_m(m,"sft","none","none","dense","similar",1,"on","none"),
            cond_B=_m(m,"sft","none","none","lexical","similar",1,"on","none")))

        # --- Similar vs Random (SFT, dense k=1) ---
        cs.append(dict(name=f"Sim→Rand|sft_dense_k1|{m}",
            cond_A=_m(m,"sft","none","none","dense","similar",1,"on","none"),
            cond_B=_m(m,"sft","none","none","dense","random",1,"on","none")))

        # --- Template on → off (SFT, dense k=1) ---
        cs.append(dict(name=f"TplOn→Off|sft_dense_k1|{m}",
            cond_A=_m(m,"sft","none","none","dense","similar",1,"on","none"),
            cond_B=_m(m,"sft","none","none","dense","similar",1,"off","none")))

        # --- No mask → Hardmask (SFT, dense k=1) ---
        cs.append(dict(name=f"NoMask→HardMask|sft_dense_k1|{m}",
            cond_A=_m(m,"sft","none","none","dense","similar",1,"on","none"),
            cond_B=_m(m,"sft","none","none","dense","similar",1,"on","hardmask")))

        # --- DPO free vs aware (balanced) ---
        cs.append(dict(name=f"DPO_free→aware|bal_k0|{m}",
            cond_A=_m(m,"dpo","balanced","retrieval_free","none","none",0,"on","none"),
            cond_B=_m(m,"dpo","balanced","retrieval_aware","none","none",0,"on","none")))
        cs.append(dict(name=f"DPO_free→aware|bal_dense_k1|{m}",
            cond_A=_m(m,"dpo","balanced","retrieval_free","dense","similar",1,"on","none"),
            cond_B=_m(m,"dpo","balanced","retrieval_aware","dense","similar",1,"on","none")))

    # --- Qwen-only DPO profile contrasts ---
    m = "qwen2.5-7b"
    for k_val, rtype, rmode, ksuf in [(0,"none","none","k0"), (1,"dense","similar","dense_k1")]:
        cs.append(dict(name=f"DPO_bal→hard|free_{ksuf}|{m}",
            cond_A=_m(m,"dpo","balanced","retrieval_free",rtype,rmode,k_val,"on","none"),
            cond_B=_m(m,"dpo","hard","retrieval_free",rtype,rmode,k_val,"on","none")))
        cs.append(dict(name=f"DPO_bal→struct|free_{ksuf}|{m}",
            cond_A=_m(m,"dpo","balanced","retrieval_free",rtype,rmode,k_val,"on","none"),
            cond_B=_m(m,"dpo","structure","retrieval_free",rtype,rmode,k_val,"on","none")))

    return cs

# ────────────────────────────────────────────
# Statistical tests
# ────────────────────────────────────────────

def paired_bootstrap_ci(delta: np.ndarray, n_boot: int, alpha: float,
                        rng: np.random.Generator) -> Tuple[float, float]:
    n = len(delta)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = delta[idx].mean()
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def rank_biserial_from_wilcoxon(delta: np.ndarray) -> float:
    """Rank-biserial r as effect size for Wilcoxon signed-rank test."""
    nonzero = delta[delta != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = sp_stats.rankdata(np.abs(nonzero))
    r_plus = ranks[nonzero > 0].sum()
    r_minus = ranks[nonzero < 0].sum()
    n = len(nonzero)
    return float((r_plus - r_minus) / (n * (n + 1) / 2))


def holm_correction(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni correction. Returns q-values."""
    n = len(pvals)
    if n == 0:
        return np.array([])
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    q = np.empty(n)
    cummax = 0.0
    for i in range(n):
        adjusted = sorted_p[i] * (n - i)
        cummax = max(cummax, adjusted)
        q[order[i]] = min(cummax, 1.0)
    return q

# ────────────────────────────────────────────
# Main
# ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="필수1: contrast-level paired statistics")
    parser.add_argument("--scored-dir", type=str, required=True,
                        help="Path to scored/ directory")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory for CSVs")
    parser.add_argument("--boot-iters", type=int, default=BOOT_ITERS)
    parser.add_argument("--ci-alpha", type=float, default=CI_ALPHA)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    scored_dir = Path(args.scored_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 1) Load all scored data
    print("[1/4] Loading scored data...")
    df = load_scored_dir(scored_dir)

    # Ensure k is int for matching
    df["k"] = df["k"].astype(int)

    # 2) Export per-instance long-format table
    print("[2/4] Exporting per-instance data...")
    export_cols = ["row_id"] + COND_COLS + ["ConditionID"] + TARGET_METRICS + [
        "gen_tok_count", "UAHE_Total", "IUHE_Total", "RAHE_Total",
        "TransferRate", "PlaceholderRate", "UnknownRate",
        "SignalFilledRate", "NoSignalFilledRate",
    ]
    exist_cols = [c for c in export_cols if c in df.columns]
    per_inst_path = out_dir / "per_instance_all.csv.gz"
    df[exist_cols].to_csv(per_inst_path, index=False, compression="gzip")
    print(f"  → {per_inst_path}  ({len(df)} rows)")

    # 3) Build condition lookup
    print("[3/4] Computing paired contrasts...")
    df["_ckey"] = df.apply(cond_key, axis=1)
    cond_groups = {k: g.set_index("row_id") for k, g in df.groupby("_ckey")}

    models = sorted(df["model"].unique())
    contrasts = build_contrasts(models)
    print(f"  Defined {len(contrasts)} contrasts × {len(TARGET_METRICS)} metrics")

    # Compute per-contrast, per-metric stats
    rows_out = []
    pvals_all = []  # for Holm correction across ALL tests

    for cdef in contrasts:
        key_a = "|".join(str(cdef["cond_A"].get(c, "")) for c in COND_COLS)
        key_b = "|".join(str(cdef["cond_B"].get(c, "")) for c in COND_COLS)
        if key_a not in cond_groups or key_b not in cond_groups:
            # This contrast doesn't exist in the data (e.g., condition missing)
            continue
        ga = cond_groups[key_a]
        gb = cond_groups[key_b]
        common_ids = ga.index.intersection(gb.index)
        if len(common_ids) < 5:
            continue

        for metric in TARGET_METRICS:
            if metric not in ga.columns or metric not in gb.columns:
                continue
            va = ga.loc[common_ids, metric].astype(float).values
            vb = gb.loc[common_ids, metric].astype(float).values
            delta = vb - va  # positive = B is higher

            mean_a = float(np.mean(va))
            mean_b = float(np.mean(vb))
            mean_delta = float(np.mean(delta))
            ci_lo, ci_hi = paired_bootstrap_ci(delta, args.boot_iters,
                                                args.ci_alpha, rng)

            nonzero_mask = delta != 0
            n_nonzero = int(nonzero_mask.sum())
            if n_nonzero >= 5:
                try:
                    stat_w, p_w = sp_stats.wilcoxon(delta[nonzero_mask])
                except Exception:
                    stat_w, p_w = np.nan, 1.0
                r_rb = rank_biserial_from_wilcoxon(delta)
            else:
                stat_w, p_w = np.nan, 1.0
                r_rb = 0.0

            rows_out.append(dict(
                contrast=cdef["name"],
                metric=metric,
                mean_A=mean_a,
                mean_B=mean_b,
                paired_delta=mean_delta,
                boot_ci_lo=ci_lo,
                boot_ci_hi=ci_hi,
                wilcoxon_p=p_w,
                rank_biserial_r=r_rb,
                n_total=len(common_ids),
                n_nonzero=n_nonzero,
            ))
            pvals_all.append(p_w)

    # 4) Holm correction across all tests
    print("[4/4] Applying Holm correction & saving...")
    pvals_arr = np.array(pvals_all, dtype=float)
    # Replace NaN with 1.0 for Holm
    pvals_clean = np.where(np.isnan(pvals_arr), 1.0, pvals_arr)
    qvals = holm_correction(pvals_clean)

    for i, row in enumerate(rows_out):
        row["holm_q"] = float(qvals[i])

    result_df = pd.DataFrame(rows_out)
    col_order = [
        "contrast", "metric",
        "mean_A", "mean_B", "paired_delta",
        "boot_ci_lo", "boot_ci_hi",
        "wilcoxon_p", "holm_q", "rank_biserial_r",
        "n_total", "n_nonzero",
    ]
    result_df = result_df[col_order]
    out_path = out_dir / "contrast_paired_stats.csv"
    result_df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"  → {out_path}  ({len(result_df)} rows)")

    # Summary
    sig = result_df[result_df["holm_q"] < 0.05]
    print(f"\n  Significant contrasts (Holm q < 0.05): {len(sig)} / {len(result_df)}")
    print("Done.")


if __name__ == "__main__":
    main()
