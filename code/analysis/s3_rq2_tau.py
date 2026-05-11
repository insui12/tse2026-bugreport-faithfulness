#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s3_rq2_tau.py  —  필수 3: RQ2 Kendall tau-b + bootstrap CI + Holm q-value

Computes ordinal associations between retrieval configuration axes and metrics.

Ordinal axes examined:
  A) k-value effect:  k ∈ {0, 1, 2}  (SFT adapter, dense similar, per model)
  B) adapter progression:  base(0) < sft(1) < dpo(2)  (per model, per retrieval config)

Per-instance (N=391) tau-b is computed for each (axis × model × metric) combination.

Outputs:
  1) rq2_tau_per_instance.csv   — tau-b, p-value, bootstrap CI, Holm q per test
  2) rq2_tau_summary.csv        — aggregated across models (mean tau, combined p)
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

# ────────────────────────────────────────────
# Config
# ────────────────────────────────────────────

TARGET_METRICS = [
    "ROUGE1_R", "ROUGE1_F1", "SBERT", "CTQRS",
    "ROUGE1_R_NORM", "ROUGE1_F1_NORM", "SBERT_NORM",
    "SecFilled", "CondFilledRate",
    "UAHE_per_1kTok", "UAHE_Total",
    "ContextSupportRate",
]

BOOT_ITERS = 10000
CI_ALPHA = 0.05
RNG_SEED = 42

COND_COLS = [
    "model", "adapter_type", "pref_profile", "align_context",
    "retriever_type", "retriever_mode", "k", "template", "retrieval_mask",
]

# ────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────

def load_scored_dir(scored_dir: Path) -> pd.DataFrame:
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
        sys.exit(f"[ERROR] No scored files in {scored_dir}")
    return pd.concat(frames, ignore_index=True)

# ────────────────────────────────────────────
# Axis extractors: return (ordinal_value, per-instance metric arrays)
# ────────────────────────────────────────────

def extract_k_axis(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Axis A: k ∈ {0, 1, 2} for SFT + dense similar + template on + no mask."""
    tests = []
    df_sft = df[
        (df["adapter_type"] == "sft") &
        (df["template"] == "on") &
        (df["retrieval_mask"] == "none")
    ].copy()

    for model in sorted(df_sft["model"].unique()):
        dm = df_sft[df_sft["model"] == model]

        # k=0: no retriever
        k0 = dm[(dm["k"] == 0) & (dm["retriever_type"] == "none")]
        # k=1: dense similar
        k1 = dm[(dm["k"] == 1) & (dm["retriever_type"] == "dense") & (dm["retriever_mode"] == "similar")]
        # k=2: dense similar
        k2 = dm[(dm["k"] == 2) & (dm["retriever_type"] == "dense") & (dm["retriever_mode"] == "similar")]

        if k0.empty or k1.empty or k2.empty:
            continue

        # Build per-instance arrays: each row_id gets ordinal level repeated
        frames = []
        for kval, sub in [(0, k0), (1, k1), (2, k2)]:
            sub = sub.copy()
            sub["_ordinal"] = kval
            frames.append(sub)
        combined = pd.concat(frames, ignore_index=True)

        tests.append(dict(
            axis="k_value",
            model=model,
            context="sft_dense_similar",
            data=combined,
        ))
    return tests


def extract_adapter_axis(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Axis B: base(0) → sft(1) → dpo(2), per model × retrieval config."""
    tests = []
    adapter_map = {"base": 0, "sft": 1, "dpo": 2}

    configs = [
        ("k0", {"k": 0, "retriever_type": "none", "retriever_mode": "none",
                "template": "on", "retrieval_mask": "none"}),
        ("dense_k1", {"k": 1, "retriever_type": "dense", "retriever_mode": "similar",
                      "template": "on", "retrieval_mask": "none"}),
    ]

    for model in sorted(df["model"].unique()):
        dm = df[df["model"] == model]
        for cfg_name, cfg_filter in configs:
            frames = []
            for adp, ordval in [("base", 0), ("sft", 1)]:
                sub = dm[dm["adapter_type"] == adp]
                for col, val in cfg_filter.items():
                    sub = sub[sub[col] == val] if col != "k" else sub[sub[col] == int(val)]
                if sub.empty:
                    continue
                sub = sub.copy()
                sub["_ordinal"] = ordval
                # SFT has pref_profile=none, align_context=none
                frames.append(sub)

            # DPO: use balanced + retrieval_free as canonical
            dpo_sub = dm[
                (dm["adapter_type"] == "dpo") &
                (dm["pref_profile"] == "balanced") &
                (dm["align_context"] == "retrieval_free")
            ]
            for col, val in cfg_filter.items():
                dpo_sub = dpo_sub[dpo_sub[col] == val] if col != "k" else dpo_sub[dpo_sub[col] == int(val)]
            if not dpo_sub.empty:
                dpo_sub = dpo_sub.copy()
                dpo_sub["_ordinal"] = 2
                frames.append(dpo_sub)

            if len(frames) < 2:
                continue
            combined = pd.concat(frames, ignore_index=True)
            tests.append(dict(
                axis="adapter_progression",
                model=model,
                context=cfg_name,
                data=combined,
            ))
    return tests

# ────────────────────────────────────────────
# Statistical computation
# ────────────────────────────────────────────

def bootstrap_tau_ci(x: np.ndarray, y: np.ndarray, n_boot: int, alpha: float,
                     rng: np.random.Generator) -> Tuple[float, float]:
    n = len(x)
    taus = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        tau_b, _ = sp_stats.kendalltau(x[idx], y[idx])
        taus[i] = tau_b if not np.isnan(tau_b) else 0.0
    lo = np.percentile(taus, 100 * alpha / 2)
    hi = np.percentile(taus, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def holm_correction(pvals: np.ndarray) -> np.ndarray:
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
    parser = argparse.ArgumentParser(description="필수3: RQ2 tau-b analysis")
    parser.add_argument("--scored-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--boot-iters", type=int, default=BOOT_ITERS)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    scored_dir = Path(args.scored_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("[1/3] Loading scored data...")
    df = load_scored_dir(scored_dir)
    df["k"] = df["k"].astype(int)
    print(f"  {len(df)} rows loaded")

    print("[2/3] Computing per-instance tau-b...")
    all_tests = extract_k_axis(df) + extract_adapter_axis(df)
    print(f"  {len(all_tests)} axis-model combinations found")

    rows_out = []
    pvals_all = []

    for test in all_tests:
        data = test["data"]
        ordinal = data["_ordinal"].values.astype(float)

        for metric in TARGET_METRICS:
            if metric not in data.columns:
                continue
            vals = data[metric].astype(float).values
            mask = ~(np.isnan(ordinal) | np.isnan(vals))
            x, y = ordinal[mask], vals[mask]
            if len(x) < 10:
                continue

            tau_b, p_val = sp_stats.kendalltau(x, y)
            if np.isnan(tau_b):
                tau_b, p_val = 0.0, 1.0
            ci_lo, ci_hi = bootstrap_tau_ci(x, y, args.boot_iters, CI_ALPHA, rng)

            rows_out.append(dict(
                axis=test["axis"],
                model=test["model"],
                context=test["context"],
                metric=metric,
                tau_b=tau_b,
                tau_p=p_val,
                boot_ci_lo=ci_lo,
                boot_ci_hi=ci_hi,
                n=len(x),
                n_levels=int(len(np.unique(x))),
            ))
            pvals_all.append(p_val)

    # Holm correction
    print("[3/3] Applying Holm correction & saving...")
    pvals_arr = np.array(pvals_all, dtype=float)
    pvals_clean = np.where(np.isnan(pvals_arr), 1.0, pvals_arr)
    qvals = holm_correction(pvals_clean)

    for i, row in enumerate(rows_out):
        row["holm_q"] = float(qvals[i])

    result_df = pd.DataFrame(rows_out)
    col_order = [
        "axis", "model", "context", "metric",
        "tau_b", "tau_p", "holm_q",
        "boot_ci_lo", "boot_ci_hi",
        "n", "n_levels",
    ]
    result_df = result_df[[c for c in col_order if c in result_df.columns]]
    out_path = out_dir / "rq2_tau_per_instance.csv"
    result_df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"  → {out_path}  ({len(result_df)} rows)")

    # Summary across models
    if not result_df.empty:
        summary = result_df.groupby(["axis", "context", "metric"]).agg(
            mean_tau=("tau_b", "mean"),
            median_tau=("tau_b", "median"),
            min_tau=("tau_b", "min"),
            max_tau=("tau_b", "max"),
            mean_q=("holm_q", "mean"),
            n_models=("model", "nunique"),
            n_sig=("holm_q", lambda x: (x < 0.05).sum()),
        ).reset_index()
        summary_path = out_dir / "rq2_tau_summary.csv"
        summary.to_csv(summary_path, index=False, float_format="%.6f")
        print(f"  → {summary_path}")

    sig = result_df[result_df["holm_q"] < 0.05] if not result_df.empty else result_df
    print(f"\n  Significant (Holm q < 0.05): {len(sig)} / {len(result_df)}")
    print("Done.")


if __name__ == "__main__":
    main()
