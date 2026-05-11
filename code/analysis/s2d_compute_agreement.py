#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s2d_compute_agreement.py — Compute inter-annotator agreement (κ) and consensus

Usage:
  python s2d_compute_agreement.py \
    --annot-a  analysis/output/entity_level_annotA.csv \
    --annot-b  result_gpt_merged.tsv \
    --annot-c  result_gemini_merged.tsv \
    --out-dir  analysis/output

Outputs:
  agreement_summary.csv     — pairwise κ + overall stats
  consensus_labels.csv      — majority vote (≥2/3) consensus
  disagreements.csv         — rows where annotators disagree
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def load_annotA(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"annotA_extraction": "A_ext", "annotA_attribution": "A_attr"})
    # Normalize to TP/FP
    df["A_ext_bin"] = df["A_ext"].apply(lambda x: "TP" if str(x).startswith("TP") else "FP")
    # Normalize attribution
    df["A_attr_bin"] = df.apply(
        lambda r: "input" if "input" in str(r.get("A_attr","")).lower()
        else "retrieval" if "retrieval" in str(r.get("A_attr","")).lower()
        else "unattributed" if "unattributed" in str(r.get("A_attr","")).lower()
        else "NA", axis=1)
    return df


def load_external(path: Path, label: str) -> pd.DataFrame:
    """Load GPT/Gemini TSV output."""
    sep = "\t" if path.suffix == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)
    # Normalize column names
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("extraction", "ext", "annotb_extraction"):
            col_map[c] = f"{label}_ext"
        elif cl in ("attribution", "attr", "annotb_attribution"):
            col_map[c] = f"{label}_attr"
        elif cl in ("row_id",):
            col_map[c] = "row_id"
        elif cl in ("entity_type",):
            col_map[c] = "entity_type"
        elif cl in ("entity_surface",):
            col_map[c] = "entity_surface"
    df = df.rename(columns=col_map)

    ext_col = f"{label}_ext"
    attr_col = f"{label}_attr"

    if ext_col in df.columns:
        df[f"{label}_ext_bin"] = df[ext_col].apply(
            lambda x: "TP" if str(x).strip().upper().startswith("TP") else "FP")
    if attr_col in df.columns:
        df[f"{label}_attr_bin"] = df[attr_col].apply(
            lambda x: "input" if "input" in str(x).lower()
            else "retrieval" if "retrieval" in str(x).lower()
            else "unattributed" if "unattributed" in str(x).lower()
            else "NA")
    return df


def compute_kappa_and_stats(labels1, labels2, name1, name2):
    """Compute Cohen's κ and agreement stats."""
    mask = pd.notna(labels1) & pd.notna(labels2)
    l1 = labels1[mask].values
    l2 = labels2[mask].values
    if len(l1) < 5:
        return {"pair": f"{name1}-{name2}", "n": len(l1), "kappa": np.nan,
                "agreement": np.nan, "note": "too few"}
    agree = (l1 == l2).sum()
    kappa = cohen_kappa_score(l1, l2)
    return {
        "pair": f"{name1}-{name2}",
        "n": int(len(l1)),
        "agree": int(agree),
        "agreement": round(agree / len(l1), 4),
        "kappa": round(kappa, 4),
    }


def majority_vote(row, columns):
    """Return majority label if ≥2/3 agree, else 'no_consensus'."""
    vals = [row[c] for c in columns if pd.notna(row.get(c))]
    if not vals:
        return "no_consensus"
    from collections import Counter
    counts = Counter(vals)
    top, top_count = counts.most_common(1)[0]
    if top_count >= 2:
        return top
    return "no_consensus"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annot-a", required=True)
    parser.add_argument("--annot-b", required=True, help="GPT-4 result TSV/CSV")
    parser.add_argument("--annot-c", default=None, help="Gemini result TSV/CSV (optional)")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load annotators
    print("[1/4] Loading annotations...")
    a_df = load_annotA(Path(args.annot_a))
    b_df = load_external(Path(args.annot_b), "B")

    # Merge on row_id + entity_type + entity_surface
    merge_keys = ["row_id", "entity_type", "entity_surface"]
    merged = a_df.merge(b_df[["row_id", "entity_type", "entity_surface",
                                "B_ext_bin", "B_attr_bin"]],
                         on=merge_keys, how="left")

    if args.annot_c:
        c_df = load_external(Path(args.annot_c), "C")
        merged = merged.merge(c_df[["row_id", "entity_type", "entity_surface",
                                     "C_ext_bin", "C_attr_bin"]],
                               on=merge_keys, how="left")

    print(f"  Merged: {len(merged)} entities")
    print(f"  B matched: {merged['B_ext_bin'].notna().sum()}")
    if args.annot_c:
        print(f"  C matched: {merged['C_ext_bin'].notna().sum()}")

    # ── Extraction agreement ──
    print("\n[2/4] Extraction agreement (TP/FP)...")
    agreement_rows = []

    # A vs B
    stats_ab = compute_kappa_and_stats(merged["A_ext_bin"], merged["B_ext_bin"], "Claude", "GPT-4")
    stats_ab["task"] = "extraction"
    agreement_rows.append(stats_ab)
    print(f"  Claude-GPT4: κ={stats_ab['kappa']}, agreement={stats_ab.get('agreement')}")

    if args.annot_c and "C_ext_bin" in merged.columns:
        stats_ac = compute_kappa_and_stats(merged["A_ext_bin"], merged["C_ext_bin"], "Claude", "Gemini")
        stats_ac["task"] = "extraction"
        agreement_rows.append(stats_ac)
        print(f"  Claude-Gemini: κ={stats_ac['kappa']}, agreement={stats_ac.get('agreement')}")

        stats_bc = compute_kappa_and_stats(merged["B_ext_bin"], merged["C_ext_bin"], "GPT-4", "Gemini")
        stats_bc["task"] = "extraction"
        agreement_rows.append(stats_bc)
        print(f"  GPT4-Gemini: κ={stats_bc['kappa']}, agreement={stats_bc.get('agreement')}")

    # ── Attribution agreement (TP only) ──
    print("\n[3/4] Attribution agreement (TP entities only)...")
    tp_mask = merged["A_ext_bin"] == "TP"
    if "B_ext_bin" in merged.columns:
        tp_mask = tp_mask & (merged["B_ext_bin"] == "TP")

    tp_merged = merged[tp_mask].copy()
    print(f"  TP entities for attribution: {len(tp_merged)}")

    stats_ab_attr = compute_kappa_and_stats(
        tp_merged["A_attr_bin"], tp_merged["B_attr_bin"], "Claude", "GPT-4")
    stats_ab_attr["task"] = "attribution"
    agreement_rows.append(stats_ab_attr)
    print(f"  Claude-GPT4 attribution: κ={stats_ab_attr['kappa']}")

    if args.annot_c and "C_attr_bin" in tp_merged.columns:
        stats_ac_attr = compute_kappa_and_stats(
            tp_merged["A_attr_bin"], tp_merged["C_attr_bin"], "Claude", "Gemini")
        stats_ac_attr["task"] = "attribution"
        agreement_rows.append(stats_ac_attr)

        stats_bc_attr = compute_kappa_and_stats(
            tp_merged["B_attr_bin"], tp_merged["C_attr_bin"], "GPT-4", "Gemini")
        stats_bc_attr["task"] = "attribution"
        agreement_rows.append(stats_bc_attr)

    # ── Consensus & output ──
    print("\n[4/4] Computing consensus and saving...")

    # Majority vote
    ext_cols = [c for c in ["A_ext_bin", "B_ext_bin", "C_ext_bin"] if c in merged.columns]
    merged["consensus_ext"] = merged.apply(lambda r: majority_vote(r, ext_cols), axis=1)

    attr_cols = [c for c in ["A_attr_bin", "B_attr_bin", "C_attr_bin"] if c in merged.columns]
    merged["consensus_attr"] = merged.apply(lambda r: majority_vote(r, attr_cols), axis=1)

    # Regex vs consensus precision
    consensus_tp = (merged["consensus_ext"] == "TP").sum()
    consensus_fp = (merged["consensus_ext"] == "FP").sum()
    consensus_prec = consensus_tp / max(1, consensus_tp + consensus_fp)
    print(f"\n  Regex precision vs consensus: {consensus_prec:.3f} "
          f"({consensus_tp} TP, {consensus_fp} FP)")

    # Per-type precision
    for et in ["URL", "HASH", "FILE", "FLAG"]:
        sub = merged[merged["entity_type"] == et]
        tp = (sub["consensus_ext"] == "TP").sum()
        fp = (sub["consensus_ext"] == "FP").sum()
        print(f"  {et:5s}: precision={tp/max(1,tp+fp):.3f} ({tp} TP, {fp} FP)")

    # Save
    agree_df = pd.DataFrame(agreement_rows)
    agree_df.to_csv(out_dir / "agreement_summary.csv", index=False)
    print(f"\n  → {out_dir / 'agreement_summary.csv'}")

    # Consensus labels
    out_cols = merge_keys + ext_cols + ["consensus_ext"] + attr_cols + ["consensus_attr"]
    out_cols = [c for c in out_cols if c in merged.columns]
    merged[out_cols].to_csv(out_dir / "consensus_labels.csv", index=False)
    print(f"  → {out_dir / 'consensus_labels.csv'}")

    # Disagreements
    disagree_mask = pd.Series(False, index=merged.index)
    for i in range(len(ext_cols)):
        for j in range(i+1, len(ext_cols)):
            disagree_mask |= (merged[ext_cols[i]] != merged[ext_cols[j]])
    disagree = merged[disagree_mask]
    disagree.to_csv(out_dir / "disagreements.csv", index=False)
    print(f"  → {out_dir / 'disagreements.csv'} ({len(disagree)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
