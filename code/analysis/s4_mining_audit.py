#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s4_mining_audit.py  —  필수 4: DPO pair-mining audit

Reads existing meta.json + candidate pools to produce:
  1) mining_audit_summary.csv       — per-adapter: pairs, pass-rate, reject reasons
  2) threshold_sensitivity.csv      — grid sweep of margin/quality_floor/uahe_cap
  3) candidate_pool_stats.csv       — per-prompt score distribution from candidate pools
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ────────────────────────────────────────────
# Config
# ────────────────────────────────────────────

# Sensitivity grid for threshold sweep
MARGINS = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
QUALITY_FLOORS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
UAHE_CAPS = [-1.0, 3.0, 5.0, 10.0, 15.0]  # -1 = disabled
MIN_COND_RATES = [0.0, 0.3, 0.5, 0.6, 0.7]

# ────────────────────────────────────────────
# Data loading helpers
# ────────────────────────────────────────────

def find_meta_files(remine_dir: Path) -> List[Path]:
    return sorted(remine_dir.rglob("*.meta.json"))


def find_candidate_pools(cache_dir: Path) -> List[Path]:
    return sorted(cache_dir.rglob("*_candidates_full.jsonl.gz"))


def load_candidate_pool(pool_path: Path) -> List[Dict[str, Any]]:
    """Load candidate pool, grouped by row_id."""
    rows = []
    with gzip.open(pool_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

# ────────────────────────────────────────────
# Part 1: Mining audit from meta.json
# ────────────────────────────────────────────

def audit_from_meta(meta_files: List[Path]) -> pd.DataFrame:
    records = []
    for mp in meta_files:
        with open(mp, "r", encoding="utf-8") as f:
            meta = json.load(f)

        model = meta.get("model_key", "unknown")
        profile = meta.get("pref_profile", "unknown")
        align = meta.get("align_context", "unknown")
        policy = meta.get("policy", {})

        prompts_total = meta.get("prompts_total", 0)
        prompts_kept = meta.get("prompts_kept", 0)
        pairs = meta.get("pairs", 0)
        reject_reasons = meta.get("reject_reason_counts", {})

        pass_rate = prompts_kept / max(1, prompts_total)

        rec = {
            "model": model,
            "pref_profile": profile,
            "align_context": align,
            "quality_floor": policy.get("quality_floor", 0.0),
            "margin": policy.get("margin", 0.05),
            "uahe_cap": policy.get("uahe_cap", -1.0),
            "min_cond_filled_rate": policy.get("min_cond_filled_rate", 0.0),
            "enforce_uahe_order": policy.get("enforce_uahe_order", False),
            "pairs_per_prompt": policy.get("pairs_per_prompt", 1),
            "prompts_total": prompts_total,
            "prompts_kept": prompts_kept,
            "pairs": pairs,
            "pass_rate": round(pass_rate, 4),
        }
        # Add reject reason columns
        for reason, count in reject_reasons.items():
            rec[f"reject_{reason}"] = count

        rec["meta_path"] = str(mp)
        records.append(rec)

    return pd.DataFrame(records)

# ────────────────────────────────────────────
# Part 2: Threshold sensitivity sweep
# ────────────────────────────────────────────

def select_pairs_sweep(
    candidates_by_prompt: Dict[int, List[Dict]],
    margin: float,
    quality_floor: float,
    uahe_cap: float,
    min_cond: float,
    enforce_uahe_order: bool = True,
) -> Dict[str, int]:
    """Simulate pair selection with given thresholds. Returns counts."""
    prompts_total = len(candidates_by_prompt)
    prompts_kept = 0
    pairs = 0
    reasons = Counter()

    for row_id, cands in candidates_by_prompt.items():
        if not cands:
            reasons["no_candidates"] += 1
            continue

        # Sort by pref_score descending
        sorted_cands = sorted(cands, key=lambda c: c.get("pref_score", 0.0), reverse=True)
        best = sorted_cands[0]
        best_sc = best.get("pref_score", 0.0)
        best_uahe = best.get("UAHE_per_1kTok", 0.0)
        best_cond = best.get("CondFilledRate", 0.0)

        # Quality floor
        if best_sc < quality_floor:
            reasons["below_quality_floor"] += 1
            continue

        # UAHE cap
        if uahe_cap >= 0 and best_uahe > uahe_cap:
            reasons["uahe_cap"] += 1
            continue

        # Min cond filled
        if min_cond > 0 and best_cond < min_cond:
            reasons["min_cond"] += 1
            continue

        # Find reject candidate
        found_pair = False
        for worst in reversed(sorted_cands):
            worst_sc = worst.get("pref_score", 0.0)
            if (best_sc - worst_sc) < margin:
                continue
            if enforce_uahe_order:
                worst_uahe = worst.get("UAHE_per_1kTok", 0.0)
                if best_uahe > worst_uahe:
                    continue
            best_text = best.get("gen_text", "").strip()
            worst_text = worst.get("gen_text", "").strip()
            if best_text == worst_text:
                continue
            found_pair = True
            pairs += 1
            break

        if found_pair:
            prompts_kept += 1
        else:
            reasons["no_valid_reject"] += 1

    return {
        "prompts_total": prompts_total,
        "prompts_kept": prompts_kept,
        "pairs": pairs,
        "pass_rate": round(prompts_kept / max(1, prompts_total), 4),
        **{f"reject_{k}": v for k, v in reasons.items()},
    }


def run_sensitivity_sweep(pool_path: Path) -> List[Dict[str, Any]]:
    """Run threshold grid sweep on one candidate pool."""
    print(f"  Loading {pool_path.name}...")
    raw = load_candidate_pool(pool_path)

    # Group by row_id
    by_prompt: Dict[int, List[Dict]] = defaultdict(list)
    for rec in raw:
        row_id = rec.get("row_id", rec.get("prompt_idx", -1))
        # Flatten: extract pref_score and components
        comps = rec.get("components", rec)
        entry = {
            "gen_text": rec.get("gen_text", rec.get("candidate_text", "")),
            "pref_score": float(comps.get("PrefScore", comps.get("pref_score", 0.0))),
            "UAHE_per_1kTok": float(comps.get("UAHE_per_1kTok", 0.0)),
            "CondFilledRate": float(comps.get("CondFilledRate", 0.0)),
        }
        by_prompt[row_id].append(entry)

    # Extract pool identity from filename
    pool_id = pool_path.stem.replace("_candidates_full.jsonl", "")

    results = []
    for margin in MARGINS:
        for qf in QUALITY_FLOORS:
            for uahe in UAHE_CAPS:
                for mc in MIN_COND_RATES:
                    stats = select_pairs_sweep(
                        by_prompt, margin=margin, quality_floor=qf,
                        uahe_cap=uahe, min_cond=mc)
                    results.append({
                        "pool": pool_id,
                        "margin": margin,
                        "quality_floor": qf,
                        "uahe_cap": uahe,
                        "min_cond_filled": mc,
                        **stats,
                    })
    return results

# ────────────────────────────────────────────
# Part 3: Candidate pool score statistics
# ────────────────────────────────────────────

def candidate_pool_stats(pool_path: Path) -> Dict[str, Any]:
    raw = load_candidate_pool(pool_path)
    pool_id = pool_path.stem.replace("_candidates_full.jsonl", "")

    by_prompt: Dict[int, List[float]] = defaultdict(list)
    all_scores = []
    all_uahe = []
    all_cond = []

    for rec in raw:
        row_id = rec.get("row_id", rec.get("prompt_idx", -1))
        comps = rec.get("components", rec)
        sc = float(comps.get("PrefScore", comps.get("pref_score", 0.0)))
        by_prompt[row_id].append(sc)
        all_scores.append(sc)
        all_uahe.append(float(comps.get("UAHE_per_1kTok", 0.0)))
        all_cond.append(float(comps.get("CondFilledRate", 0.0)))

    scores_arr = np.array(all_scores)
    uahe_arr = np.array(all_uahe)
    cond_arr = np.array(all_cond)

    # Per-prompt spread
    spreads = [max(scs) - min(scs) for scs in by_prompt.values() if len(scs) > 1]

    return {
        "pool": pool_id,
        "n_prompts": len(by_prompt),
        "n_candidates": len(all_scores),
        "candidates_per_prompt": round(len(all_scores) / max(1, len(by_prompt)), 1),
        "score_mean": round(float(scores_arr.mean()), 4),
        "score_std": round(float(scores_arr.std()), 4),
        "score_median": round(float(np.median(scores_arr)), 4),
        "score_min": round(float(scores_arr.min()), 4),
        "score_max": round(float(scores_arr.max()), 4),
        "uahe_mean": round(float(uahe_arr.mean()), 4),
        "uahe_median": round(float(np.median(uahe_arr)), 4),
        "cond_filled_mean": round(float(cond_arr.mean()), 4),
        "per_prompt_spread_mean": round(float(np.mean(spreads)), 4) if spreads else 0.0,
        "per_prompt_spread_median": round(float(np.median(spreads)), 4) if spreads else 0.0,
    }

# ────────────────────────────────────────────
# Main
# ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="필수4: DPO pair-mining audit")
    parser.add_argument("--remine-dir", type=str, required=True,
                        help="Path to data/dpo_pairs_remine/")
    parser.add_argument("--candidates-dir", type=str, required=True,
                        help="Path to results/cache/dpo_candidates/")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="Skip threshold sweep (faster)")
    args = parser.parse_args()

    remine_dir = Path(args.remine_dir)
    candidates_dir = Path(args.candidates_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Part 1: Audit from remine meta
    print("[1/3] Mining audit from meta.json...")
    meta_files = find_meta_files(remine_dir)
    if meta_files:
        audit_df = audit_from_meta(meta_files)
        audit_path = out_dir / "mining_audit_summary.csv"
        audit_df.to_csv(audit_path, index=False)
        print(f"  → {audit_path}  ({len(audit_df)} entries)")
        # Print summary
        print(audit_df[[
            "model", "pref_profile", "align_context",
            "prompts_total", "prompts_kept", "pairs", "pass_rate"
        ]].to_string(index=False))
    else:
        print("  [WARN] No meta.json files found in remine dir")

    # Part 2: Candidate pool stats
    print("\n[2/3] Candidate pool statistics...")
    pool_paths = find_candidate_pools(candidates_dir)
    if pool_paths:
        pool_records = []
        for pp in pool_paths:
            ps = candidate_pool_stats(pp)
            pool_records.append(ps)
        pool_df = pd.DataFrame(pool_records)
        pool_path_out = out_dir / "candidate_pool_stats.csv"
        pool_df.to_csv(pool_path_out, index=False)
        print(f"  → {pool_path_out}")
        print(pool_df.to_string(index=False))
    else:
        print("  [WARN] No candidate pool files found")

    # Part 3: Threshold sensitivity
    if not args.skip_sensitivity:
        print(f"\n[3/3] Threshold sensitivity sweep ({len(MARGINS)}×{len(QUALITY_FLOORS)}×{len(UAHE_CAPS)}×{len(MIN_COND_RATES)} = "
              f"{len(MARGINS)*len(QUALITY_FLOORS)*len(UAHE_CAPS)*len(MIN_COND_RATES)} combos per pool)...")
        all_sensitivity = []
        for pp in pool_paths:
            results = run_sensitivity_sweep(pp)
            all_sensitivity.extend(results)
        if all_sensitivity:
            sens_df = pd.DataFrame(all_sensitivity)
            sens_path = out_dir / "threshold_sensitivity.csv"
            sens_df.to_csv(sens_path, index=False)
            print(f"  → {sens_path}  ({len(sens_df)} rows)")

            # Compact summary: for each pool, show pairs at key thresholds
            print("\n  Key threshold combinations:")
            for pool_id in sens_df["pool"].unique():
                sub = sens_df[sens_df["pool"] == pool_id]
                default = sub[
                    (sub["margin"] == 0.05) &
                    (sub["quality_floor"] == 0.0) &
                    (sub["uahe_cap"] == -1.0) &
                    (sub["min_cond_filled"] == 0.0)
                ]
                if not default.empty:
                    row = default.iloc[0]
                    print(f"    {pool_id}: default → {int(row['pairs'])} pairs "
                          f"({row['pass_rate']:.1%})")
    else:
        print("\n[3/3] Threshold sensitivity: SKIPPED (--skip-sensitivity)")

    print("\nDone.")


if __name__ == "__main__":
    main()
