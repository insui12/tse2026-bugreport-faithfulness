#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s1b_supplement_stats.py — Supplement: RAHE/TransferRate contrasts + direction consistency + BH FDR

Outputs:
  1) contrast_paired_stats_full.csv — original + RAHE_per_1kTok, TransferRate added
  2) direction_consistency.csv      — per contrast base: direction across models
  3) rq2_tau_with_bh.csv            — tau-b with BH FDR q-values
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# Import contrast builder from s1
sys.path.insert(0, str(Path(__file__).parent))
from s1_contrast_stats import (
    load_scored_dir, cond_key, build_contrasts,
    paired_bootstrap_ci, rank_biserial_from_wilcoxon, holm_correction,
    COND_COLS,
)

EXTRA_METRICS = [
    "RAHE_per_1kTok", "TransferRate", "RAHE_Total",
    "IUHE_per_1kTok", "IUHE_Total",
    "PlaceholderRate", "UnknownRate",
    "SignalFilledRate", "NoSignalFilledRate",
]

BOOT_ITERS = 10000
CI_ALPHA = 0.05
RNG_SEED = 42


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    if n == 0:
        return np.array([])
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    q = np.empty(n)
    cummin = 1.0
    for i in range(n - 1, -1, -1):
        adjusted = sorted_p[i] * n / (i + 1)
        cummin = min(cummin, adjusted)
        q[order[i]] = min(cummin, 1.0)
    return q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--existing-contrast", type=str, default=None,
                        help="Path to existing contrast_paired_stats.csv to merge with")
    parser.add_argument("--existing-tau", type=str, default=None,
                        help="Path to existing rq2_tau_per_instance.csv")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    # ── Part 1: Extra contrast metrics ──
    print("[1/3] Computing extra contrast metrics (RAHE, TransferRate, etc.)...")
    df = load_scored_dir(Path(args.scored_dir))
    df["k"] = df["k"].astype(int)
    df["_ckey"] = df.apply(cond_key, axis=1)
    cond_groups = {k: g.set_index("row_id") for k, g in df.groupby("_ckey")}

    models = sorted(df["model"].unique())
    contrasts = build_contrasts(models)

    rows_out = []
    pvals_all = []

    for cdef in contrasts:
        key_a = "|".join(str(cdef["cond_A"].get(c, "")) for c in COND_COLS)
        key_b = "|".join(str(cdef["cond_B"].get(c, "")) for c in COND_COLS)
        if key_a not in cond_groups or key_b not in cond_groups:
            continue
        ga = cond_groups[key_a]
        gb = cond_groups[key_b]
        common_ids = ga.index.intersection(gb.index)
        if len(common_ids) < 5:
            continue

        for metric in EXTRA_METRICS:
            if metric not in ga.columns or metric not in gb.columns:
                continue
            va = ga.loc[common_ids, metric].astype(float).values
            vb = gb.loc[common_ids, metric].astype(float).values
            delta = vb - va

            mean_a = float(np.mean(va))
            mean_b = float(np.mean(vb))
            mean_delta = float(np.mean(delta))
            ci_lo, ci_hi = paired_bootstrap_ci(delta, BOOT_ITERS, CI_ALPHA, rng)

            nonzero_mask = delta != 0
            n_nonzero = int(nonzero_mask.sum())
            if n_nonzero >= 5:
                try:
                    _, p_w = sp_stats.wilcoxon(delta[nonzero_mask])
                except Exception:
                    p_w = 1.0
                r_rb = rank_biserial_from_wilcoxon(delta)
            else:
                p_w = 1.0
                r_rb = 0.0

            rows_out.append(dict(
                contrast=cdef["name"], metric=metric,
                mean_A=mean_a, mean_B=mean_b, paired_delta=mean_delta,
                boot_ci_lo=ci_lo, boot_ci_hi=ci_hi,
                wilcoxon_p=p_w, rank_biserial_r=r_rb,
                n_total=len(common_ids), n_nonzero=n_nonzero,
            ))
            pvals_all.append(p_w)

    # Merge with existing
    extra_df = pd.DataFrame(rows_out)
    if args.existing_contrast:
        existing = pd.read_csv(args.existing_contrast)
        combined = pd.concat([existing, extra_df], ignore_index=True)
    else:
        combined = extra_df

    # Re-compute Holm across ALL tests
    all_p = combined["wilcoxon_p"].fillna(1.0).values
    combined["holm_q"] = holm_correction(all_p)
    combined["bh_q"] = bh_fdr(all_p)

    col_order = [
        "contrast", "metric", "mean_A", "mean_B", "paired_delta",
        "boot_ci_lo", "boot_ci_hi", "wilcoxon_p", "holm_q", "bh_q",
        "rank_biserial_r", "n_total", "n_nonzero",
    ]
    combined = combined[[c for c in col_order if c in combined.columns]]
    full_path = out_dir / "contrast_paired_stats_full.csv"
    combined.to_csv(full_path, index=False, float_format="%.6f")
    print(f"  → {full_path}  ({len(combined)} rows)")

    # ── Part 2: Direction consistency ──
    print("[2/3] Direction consistency summary...")
    combined["base_contrast"] = combined["contrast"].str.rsplit("|", n=1).str[0]
    combined["contrast_model"] = combined["contrast"].str.rsplit("|", n=1).str[1]

    dir_records = []
    for (base, metric), g in combined.groupby(["base_contrast", "metric"]):
        if len(g) < 2:
            continue
        deltas = g["paired_delta"].values
        signs = np.sign(deltas)
        nonzero_signs = signs[signs != 0]
        n_pos = int((nonzero_signs > 0).sum())
        n_neg = int((nonzero_signs < 0).sum())
        n_zero = int((signs == 0).sum())
        n_models = len(g)
        consistent = len(set(nonzero_signs)) <= 1 if len(nonzero_signs) > 0 else True
        all_sig = (g["holm_q"] < 0.05).all()
        any_sig = (g["holm_q"] < 0.05).any()
        median_r = float(g["rank_biserial_r"].median())

        dir_records.append(dict(
            base_contrast=base, metric=metric, n_models=n_models,
            n_positive=n_pos, n_negative=n_neg, n_zero=n_zero,
            direction_consistent=consistent,
            all_significant=all_sig, any_significant=any_sig,
            median_rank_biserial=median_r,
            deltas=";".join(f"{d:.4f}" for d in deltas),
            models=";".join(g["contrast_model"].values),
        ))

    dir_df = pd.DataFrame(dir_records)
    dir_path = out_dir / "direction_consistency.csv"
    dir_df.to_csv(dir_path, index=False)
    n_consistent = dir_df["direction_consistent"].sum()
    n_total = len(dir_df)
    print(f"  → {dir_path}")
    print(f"  Consistent: {n_consistent}/{n_total} ({n_consistent/n_total*100:.1f}%)")

    # ── Part 3: RQ2 tau with BH FDR ──
    print("[3/3] Adding BH FDR to RQ2 tau-b...")
    if args.existing_tau:
        tau_df = pd.read_csv(args.existing_tau)
        tau_p = tau_df["tau_p"].fillna(1.0).values
        tau_df["bh_q"] = bh_fdr(tau_p)
        tau_df["holm_q"] = holm_correction(tau_p)
        tau_path = out_dir / "rq2_tau_with_bh.csv"
        tau_df.to_csv(tau_path, index=False, float_format="%.6f")
        print(f"  → {tau_path}")
        sig_bh = (tau_df["bh_q"] < 0.05).sum()
        sig_holm = (tau_df["holm_q"] < 0.05).sum()
        print(f"  BH FDR q<0.05: {sig_bh}/{len(tau_df)}")
        print(f"  Holm q<0.05:   {sig_holm}/{len(tau_df)}")

    print("Done.")


if __name__ == "__main__":
    main()
