#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s2_hard_entity_sheet.py  —  필수 2 (자동 부분): Hard-entity extraction annotation sheet

Outputs:
  1) annotation_sheet.csv         — stratified sample with regex extraction results
                                     (annotators fill "human_*" columns)
  2) entity_extraction_stats.csv  — full-set regex extraction summary (type × count)
  3) auto_precision_recall.csv    — IF human annotations are provided, computes P/R/F1/κ
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ────────────────────────────────────────────
# Hard-entity regex (replicated from bugreport_pipeline_tse.py)
# ────────────────────────────────────────────

_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_HASH = re.compile(r"\b[a-f0-9]{7,40}\b", re.IGNORECASE)
_RE_FILENAME = re.compile(
    r"\b[\w\-.]+\.(?:gif|png|jpg|jpeg|webp|mp4|mov|avi|log|txt|csv|json|xml)\b",
    re.IGNORECASE,
)
_RE_SETTING = re.compile(
    r"\b[a-zA-Z_]+\.[a-zA-Z0-9_.]+\b(?::(true|false))?", re.IGNORECASE
)

ENTITY_TYPES = ["URL", "HASH", "FILE", "FLAG"]


def extract_hard_entities(text: str) -> Dict[str, List[str]]:
    urls = _RE_URL.findall(text)
    hashes = _RE_HASH.findall(text)
    files = _RE_FILENAME.findall(text)
    file_set = set(f.lower() for f in files)
    flags_raw = _RE_SETTING.findall(text)
    # _RE_SETTING has a capture group for (true|false), need to re-find with finditer
    flags = []
    for m in _RE_SETTING.finditer(text):
        tok = m.group(0)
        if tok.lower() not in file_set:
            flags.append(tok)
    return {"URL": urls, "HASH": hashes, "FILE": files, "FLAG": flags}


def hard_entity_sets(text: str) -> Dict[str, set]:
    ents = extract_hard_entities(text)
    return {k: set(v) for k, v in ents.items()}


def attribute_entities(
    input_sets: Dict[str, set],
    retrieval_sets: Dict[str, set],
    output_sets: Dict[str, set],
) -> Dict[str, Dict[str, set]]:
    """Per-type attribution: input / retrieval / unattributed."""
    result = {}
    for etype in ENTITY_TYPES:
        h_in = input_sets.get(etype, set())
        h_ret = retrieval_sets.get(etype, set())
        h_out = output_sets.get(etype, set())
        ia = h_out & h_in
        ra = (h_out - h_in) & h_ret
        ua = h_out - (h_in | h_ret)
        result[etype] = {"input_attr": ia, "retrieval_attr": ra, "unattributed": ua}
    return result

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
# Stratified sampling for annotation
# ────────────────────────────────────────────

def stratified_sample(df: pd.DataFrame, n_total: int, seed: int) -> pd.DataFrame:
    """Sample rows stratified by entity presence (has entities vs not)."""
    rng = random.Random(seed)

    # Compute entity counts per row
    df = df.copy()
    def _count_ents(r):
        gt = r.get("gen_text")
        if gt is None or (isinstance(gt, float) and pd.isna(gt)):
            return 0
        ents = extract_hard_entities(str(gt))
        return int(sum(len(ents[et]) for et in ENTITY_TYPES) > 0)

    df["_has_entity"] = df.apply(_count_ents, axis=1)

    with_ent = df[df["_has_entity"] == 1]
    without_ent = df[df["_has_entity"] == 0]

    # Target: 70% with entities, 30% without (or all available)
    n_with = min(len(with_ent), int(n_total * 0.7))
    n_without = min(len(without_ent), n_total - n_with)
    n_with = min(len(with_ent), n_total - n_without)  # adjust

    idx_with = rng.sample(range(len(with_ent)), n_with)
    idx_without = rng.sample(range(len(without_ent)), n_without)

    sampled = pd.concat([with_ent.iloc[idx_with], without_ent.iloc[idx_without]])
    return sampled.drop(columns=["_has_entity"]).reset_index(drop=True)

# ────────────────────────────────────────────
# Annotation sheet generation
# ────────────────────────────────────────────

def build_annotation_sheet(df_sample: pd.DataFrame) -> pd.DataFrame:
    """Build annotation sheet with regex results + blank human columns."""
    records = []
    for _, row in df_sample.iterrows():
        input_text = str(row.get("input_text", ""))
        gen_text = str(row.get("gen_text", ""))
        gt_text = str(row.get("gt_text", ""))

        in_ents = extract_hard_entities(input_text)
        gen_ents = extract_hard_entities(gen_text)
        in_sets = hard_entity_sets(input_text)
        gen_sets = hard_entity_sets(gen_text)

        # Retrieval text is not directly stored; we reconstruct attribution
        # using the stored HardRet_* counts as reference
        ret_sets = {et: set() for et in ENTITY_TYPES}  # approximation

        attr = attribute_entities(in_sets, ret_sets, gen_sets)

        rec = {
            "row_id": row.get("row_id"),
            "model": row.get("model"),
            "adapter_type": row.get("adapter_type"),
            "condition_id": row.get("ConditionID"),
            "input_text": input_text[:500],
            "gen_text": gen_text[:1000],
        }

        for et in ENTITY_TYPES:
            matched = gen_ents[et]
            rec[f"regex_{et}_count"] = len(matched)
            rec[f"regex_{et}_items"] = "; ".join(matched[:20])
            rec[f"regex_{et}_unattr"] = "; ".join(sorted(attr[et]["unattributed"]))[:200]

            # --- Human annotation columns (to be filled) ---
            rec[f"human_{et}_TP"] = ""   # True positives (regex correct)
            rec[f"human_{et}_FP"] = ""   # False positives (regex wrong)
            rec[f"human_{et}_FN"] = ""   # False negatives (regex missed)

        rec["human_notes"] = ""
        records.append(rec)

    return pd.DataFrame(records)

# ────────────────────────────────────────────
# Full extraction statistics
# ────────────────────────────────────────────

def compute_extraction_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per entity-type statistics across all scored rows."""
    stats = []
    for et in ENTITY_TYPES:
        col_out = f"HardOut_{et}"
        col_ua = f"HardOut_Unattributed_{et}"
        col_in = f"HardIn_{et}"
        col_ret = f"HardRet_{et}"

        total_out = df[col_out].sum() if col_out in df.columns else 0
        total_ua = df[col_ua].sum() if col_ua in df.columns else 0
        total_in = df[col_in].sum() if col_in in df.columns else 0
        total_ret = df[col_ret].sum() if col_ret in df.columns else 0
        rows_with = int((df[col_out] > 0).sum()) if col_out in df.columns else 0

        stats.append(dict(
            entity_type=et,
            total_in_input=int(total_in),
            total_in_retrieval=int(total_ret),
            total_in_output=int(total_out),
            total_unattributed=int(total_ua),
            rows_with_entity=rows_with,
            rows_total=len(df),
            pct_rows_with_entity=round(100 * rows_with / max(1, len(df)), 1),
        ))
    return pd.DataFrame(stats)

# ────────────────────────────────────────────
# Compute P/R/F1 from filled annotation sheet
# ────────────────────────────────────────────

def compute_prf_from_annotations(annot_path: Path) -> Optional[pd.DataFrame]:
    """If human annotations exist, compute precision/recall/F1."""
    if not annot_path.exists():
        return None
    df = pd.read_csv(annot_path)

    # Check if human columns are filled
    has_human = any(f"human_{et}_TP" in df.columns and df[f"human_{et}_TP"].notna().any()
                    for et in ENTITY_TYPES)
    if not has_human:
        return None

    results = []
    for et in ENTITY_TYPES:
        tp_col = f"human_{et}_TP"
        fp_col = f"human_{et}_FP"
        fn_col = f"human_{et}_FN"
        if tp_col not in df.columns:
            continue

        tp = pd.to_numeric(df[tp_col], errors="coerce").fillna(0).sum()
        fp = pd.to_numeric(df[fp_col], errors="coerce").fillna(0).sum()
        fn = pd.to_numeric(df[fn_col], errors="coerce").fillna(0).sum()

        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)

        results.append(dict(entity_type=et, TP=int(tp), FP=int(fp), FN=int(fn),
                            precision=round(prec, 4), recall=round(rec, 4),
                            F1=round(f1, 4)))
    return pd.DataFrame(results) if results else None

# ────────────────────────────────────────────
# Main
# ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="필수2: hard-entity annotation sheet")
    parser.add_argument("--scored-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--n-sample", type=int, default=100,
                        help="Number of rows for annotation sheet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--human-annot", type=str, default=None,
                        help="Path to filled annotation CSV (for P/R/F1 computation)")
    args = parser.parse_args()

    scored_dir = Path(args.scored_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Loading scored data...")
    df = load_scored_dir(scored_dir)
    print(f"  {len(df)} rows loaded")

    # Full extraction statistics
    print("[2/3] Computing entity extraction statistics...")
    stats_df = compute_extraction_stats(df)
    stats_path = out_dir / "entity_extraction_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"  → {stats_path}")
    print(stats_df.to_string(index=False))

    # Stratified annotation sample
    print(f"[3/3] Building annotation sheet (n={args.n_sample})...")
    sample_df = stratified_sample(df, args.n_sample, args.seed)
    sheet_df = build_annotation_sheet(sample_df)
    sheet_path = out_dir / "annotation_sheet.csv"
    sheet_df.to_csv(sheet_path, index=False)
    print(f"  → {sheet_path}  ({len(sheet_df)} rows)")

    # Optional: compute P/R/F1 from existing human annotations
    if args.human_annot:
        prf_df = compute_prf_from_annotations(Path(args.human_annot))
        if prf_df is not None:
            prf_path = out_dir / "auto_precision_recall.csv"
            prf_df.to_csv(prf_path, index=False)
            print(f"  → {prf_path}")
            print(prf_df.to_string(index=False))
        else:
            print("  [WARN] Human annotation columns not filled yet.")

    print("Done.")


if __name__ == "__main__":
    main()
