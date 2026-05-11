#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s2b_entity_level_sheet.py — Entity-level long-format annotation sheet

Outputs:
  1) entity_level_sheet.csv  — one row per extracted entity, with retrieval context
  2) entity_level_annotA.csv — same + Annotator A (LLM) pre-labels

Columns:
  row_id, condition_id, model, adapter_type, entity_type, entity_surface,
  source_text (input / gen / retrieval), in_input, in_retrieval, in_gen,
  regex_attribution (IAHE / RAHE / UAHE),
  input_text_excerpt, gen_text_excerpt, retrieval_text_excerpt,
  annotA_extraction_correct (TP/FP), annotA_attribution_correct (Y/N/NA),
  annotB_extraction_correct, annotB_attribution_correct  [blank for human]
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

# ────────────────────────────────────────────
# Hard-entity regex (replicated)
# ────────────────────────────────────────────

_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_HASH = re.compile(r"\b[a-f0-9]{7,40}\b", re.IGNORECASE)
_RE_FILENAME = re.compile(
    r"\b[\w\-.]+\.(?:gif|png|jpg|jpeg|webp|mp4|mov|avi|log|txt|csv|json|xml)\b",
    re.IGNORECASE,
)
_RE_SETTING = re.compile(
    r"\b[a-zA-Z_]+\.[a-zA-Z0-9_.]+\b(?::(true|false))?", re.IGNORECASE,
)

ENTITY_TYPES = ["URL", "HASH", "FILE", "FLAG"]
_RE_MAP = {"URL": _RE_URL, "HASH": _RE_HASH, "FILE": _RE_FILENAME, "FLAG": _RE_SETTING}


def extract_entities_with_spans(text: str) -> List[Dict[str, Any]]:
    """Extract entities with type, surface form, and character span."""
    results = []
    file_surfaces = set()

    # First pass: files (to exclude from flags)
    for m in _RE_FILENAME.finditer(text):
        file_surfaces.add(m.group(0).lower())
        results.append({"type": "FILE", "surface": m.group(0),
                        "start": m.start(), "end": m.end()})

    for m in _RE_URL.finditer(text):
        results.append({"type": "URL", "surface": m.group(0),
                        "start": m.start(), "end": m.end()})

    for m in _RE_HASH.finditer(text):
        results.append({"type": "HASH", "surface": m.group(0),
                        "start": m.start(), "end": m.end()})

    for m in _RE_SETTING.finditer(text):
        if m.group(0).lower() not in file_surfaces:
            results.append({"type": "FLAG", "surface": m.group(0),
                            "start": m.start(), "end": m.end()})

    return results


def entity_set(text: str) -> Dict[str, Set[str]]:
    ents = extract_entities_with_spans(text)
    out: Dict[str, Set[str]] = {t: set() for t in ENTITY_TYPES}
    for e in ents:
        out[e["type"]].add(e["surface"])
    return out


def get_context_window(text: str, start: int, end: int, window: int = 80) -> str:
    """Extract surrounding context for an entity."""
    ctx_start = max(0, start - window)
    ctx_end = min(len(text), end + window)
    prefix = "..." if ctx_start > 0 else ""
    suffix = "..." if ctx_end < len(text) else ""
    return prefix + text[ctx_start:ctx_end] + suffix


# ────────────────────────────────────────────
# LLM pre-labeling heuristics (Annotator A)
# ────────────────────────────────────────────

# Common FP patterns
_HEX_COLOR = re.compile(r"^[a-f0-9]{6}$", re.IGNORECASE)
_SHORT_HEX = re.compile(r"^[a-f0-9]{7,8}$", re.IGNORECASE)
_COMMON_DOTTED = {
    "e.g", "i.e", "etc.", "vs.", "fig.", "eq.", "ref.", "no.",
    "st.", "nd.", "rd.", "th.",
}


def annotA_extraction_label(etype: str, surface: str, context: str) -> str:
    """Heuristic pre-label: TP or FP with reason."""
    s = surface.strip()
    ctx_lower = context.lower()

    if etype == "URL":
        # URLs are almost always TP
        return "TP"

    elif etype == "HASH":
        # Very short hex that looks like a number or color code
        if _HEX_COLOR.match(s) and any(w in ctx_lower for w in ["color", "colour", "#", "rgb", "hex"]):
            return "FP:color_code"
        if len(s) <= 8 and s.isdigit():
            return "FP:decimal_number"
        # Bug IDs are typically 6-7 digit numbers — but our regex requires hex chars
        # If it contains only digits 0-9 (no a-f), might be a bug ID not a hash
        if all(c in "0123456789" for c in s):
            if len(s) <= 8:
                return "FP:likely_number"
            return "TP:likely_bugid_or_revision"
        # Standard git-like hash
        if len(s) >= 12:
            return "TP:likely_commit_hash"
        return "TP:likely_short_hash"

    elif etype == "FILE":
        # Filenames are usually TP if they have a recognizable extension
        return "TP"

    elif etype == "FLAG":
        s_lower = s.lower()
        # Known FP patterns
        if any(s_lower.startswith(p) for p in _COMMON_DOTTED):
            return "FP:abbreviation"
        # Version-like: e.g., "106.0b6" — probably part of a version string
        if re.match(r"^\d+\.\d+", s):
            return "FP:version_number"
        # Very short dotted name
        parts = s.split(".")
        if len(parts) == 2 and all(len(p) <= 2 for p in parts):
            return "FP:too_short"
        # Likely a real setting/property
        if any(w in s_lower for w in ["config", "setting", "enable", "disable",
                                       "preference", "pref", "flag", "option"]):
            return "TP:likely_setting"
        # Package/namespace like "mozilla.components" or "browser.tabs"
        if len(parts) >= 2 and all(p.isalpha() or p.replace("_", "").isalpha() for p in parts):
            return "TP:likely_property"
        return "TP:dotted_name"

    return "TP"


def annotA_attribution_label(
    etype: str, surface: str,
    in_input: bool, in_retrieval: bool, in_gen: bool,
    regex_attr: str,
) -> str:
    """Check if regex attribution is reasonable."""
    if not in_gen:
        return "NA:not_in_gen"
    if regex_attr == "IAHE" and in_input:
        return "Y:correct_input_attr"
    if regex_attr == "RAHE" and not in_input and in_retrieval:
        return "Y:correct_retrieval_attr"
    if regex_attr == "UAHE" and not in_input and not in_retrieval:
        return "Y:correct_unattributed"
    # Mismatch
    return f"N:mismatch_expected_{'IAHE' if in_input else 'RAHE' if in_retrieval else 'UAHE'}"


# ────────────────────────────────────────────
# Main
# ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entity-level annotation sheet")
    parser.add_argument("--scored-dir", type=str, required=True)
    parser.add_argument("--train-csv", type=str, required=True,
                        help="Path to train_seed42.csv for retrieval text recovery")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--n-sample", type=int, default=100,
                        help="Number of scored rows to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # Load train split for retrieval text recovery
    print("[1/4] Loading train split for retrieval recovery...")
    train_df = pd.read_csv(args.train_csv)
    train_input = train_df["NEW_llama_output"].tolist()
    train_gt = train_df["text"].tolist()
    print(f"  Train: {len(train_df)} rows")

    # Load scored data (sample across conditions with k>0 priority)
    print("[2/4] Loading scored data and sampling...")
    scored_dir = Path(args.scored_dir)
    all_rows = []
    for p in sorted(scored_dir.glob("scored_*.jsonl.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_rows.append(json.loads(line))
    print(f"  Total scored rows: {len(all_rows)}")

    # Stratified: 60% with retrieval (k>0), 40% without
    with_ret = [r for r in all_rows if int(r.get("k", 0)) > 0]
    no_ret = [r for r in all_rows if int(r.get("k", 0)) == 0]
    n_with = min(len(with_ret), int(args.n_sample * 0.6))
    n_without = min(len(no_ret), args.n_sample - n_with)
    n_with = min(len(with_ret), args.n_sample - n_without)

    sampled = rng.sample(with_ret, n_with) + rng.sample(no_ret, n_without)
    rng.shuffle(sampled)
    print(f"  Sampled: {len(sampled)} rows ({n_with} with retrieval, {n_without} without)")

    # Build entity-level records
    print("[3/4] Building entity-level long-format sheet...")
    records = []

    for row in sampled:
        row_id = row.get("row_id")
        cond_id = row.get("ConditionID", "")
        model = row.get("model", "")
        adapter = row.get("adapter_type", "")
        input_text = str(row.get("input_text", ""))
        gen_text = str(row.get("gen_text", ""))
        k = int(row.get("k", 0))

        # Recover retrieval text
        retrieval_text = ""
        retrieved_pos = row.get("RetrievedPos", [])
        if retrieved_pos and k > 0:
            ret_parts = []
            for pos in retrieved_pos:
                pos = int(pos)
                if 0 <= pos < len(train_input):
                    ret_parts.append(str(train_input[pos]))
                    ret_parts.append(str(train_gt[pos]))
            retrieval_text = "\n".join(ret_parts)

        # Extract entity sets
        in_sets = entity_set(input_text)
        ret_sets = entity_set(retrieval_text)
        gen_ents = extract_entities_with_spans(gen_text)
        gen_sets = entity_set(gen_text)

        # Also extract from input/retrieval for completeness
        in_ents = extract_entities_with_spans(input_text)
        ret_ents = extract_entities_with_spans(retrieval_text)

        # Process gen entities (primary focus)
        for ent in gen_ents:
            etype = ent["type"]
            surface = ent["surface"]
            in_input = surface in in_sets.get(etype, set())
            in_retrieval = surface in ret_sets.get(etype, set())

            # Determine regex attribution
            if in_input:
                regex_attr = "IAHE"
            elif in_retrieval:
                regex_attr = "RAHE"
            else:
                regex_attr = "UAHE"

            context = get_context_window(gen_text, ent["start"], ent["end"])

            # Annotator A labels
            a_extract = annotA_extraction_label(etype, surface, context)
            a_attrib = annotA_attribution_label(etype, surface, in_input,
                                                 in_retrieval, True, regex_attr)

            records.append({
                "row_id": row_id,
                "condition_id": cond_id,
                "model": model,
                "adapter_type": adapter,
                "k": k,
                "entity_type": etype,
                "entity_surface": surface,
                "source": "gen",
                "in_input": int(in_input),
                "in_retrieval": int(in_retrieval),
                "in_gen": 1,
                "regex_attribution": regex_attr,
                "gen_context": context,
                "input_text_300": input_text[:300],
                "retrieval_text_300": retrieval_text[:300] if retrieval_text else "",
                # Annotator A
                "annotA_extraction": a_extract,
                "annotA_attribution": a_attrib,
                # Annotator B (blank)
                "annotB_extraction": "",
                "annotB_attribution": "",
                # Notes
                "notes": "",
            })

        # Also add entities found ONLY in input (not in gen) — FN candidates
        for ent in in_ents:
            etype = ent["type"]
            surface = ent["surface"]
            if surface not in gen_sets.get(etype, set()):
                # This entity was in input but NOT reproduced in gen — not FN for extraction
                # (regex correctly didn't find it in gen). Skip.
                pass

    result_df = pd.DataFrame(records)
    print(f"  Total entity rows: {len(result_df)}")
    if not result_df.empty:
        print(f"  By type: {dict(result_df.entity_type.value_counts())}")
        print(f"  By attribution: {dict(result_df.regex_attribution.value_counts())}")
        print(f"  By annotA extraction: "
              f"TP={result_df.annotA_extraction.str.startswith('TP').sum()}, "
              f"FP={result_df.annotA_extraction.str.startswith('FP').sum()}")

    # Save
    print("[4/4] Saving...")
    # Full sheet (annotator A pre-labeled)
    out_path = out_dir / "entity_level_annotA.csv"
    result_df.to_csv(out_path, index=False)
    print(f"  → {out_path}")

    # Blank sheet for annotator B
    blank_df = result_df.copy()
    blank_df["annotA_extraction"] = ""
    blank_df["annotA_attribution"] = ""
    blank_path = out_dir / "entity_level_blank.csv"
    blank_df.to_csv(blank_path, index=False)
    print(f"  → {blank_path}")

    print("Done.")


if __name__ == "__main__":
    main()
