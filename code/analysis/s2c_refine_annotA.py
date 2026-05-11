#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s2c_refine_annotA.py — Refine Annotator A labels with context-aware heuristics

Key refinement: HASH regex catches attachment IDs, bug IDs, build IDs, user-agent
Gecko dates — these are NOT git hashes but ARE hard entities that need attribution.
The paper defines "hard entity" as any specific identifier that should be traceable.

Revised taxonomy:
  TP:commit_hash       — hex string with a-f chars, ≥12 chars (git SHA)
  TP:build_id          — 14-digit timestamp-like decimal (e.g., 20210513214800)
  TP:attachment_id      — 7-digit decimal in "attachment NNNNNNN" context
  TP:bug_id            — 5-8 digit decimal in "bug NNNNN" context
  TP:revision_id       — other numeric identifiers in regression/version context
  FP:gecko_epoch       — "20100101" in Gecko user-agent (constant, not a real entity)
  FP:ambiguous_number  — short decimal with no identifier context
"""

from __future__ import annotations

import re
import pandas as pd
from pathlib import Path


def refine_hash_label(surface: str, context: str) -> str:
    """Context-aware HASH extraction label."""
    s = surface.strip()
    ctx_lower = context.lower()

    has_hex = bool(re.search(r"[a-f]", s, re.IGNORECASE))

    # Actual hex hash (contains a-f)
    if has_hex:
        if len(s) >= 12:
            return "TP:commit_hash"
        elif len(s) >= 7:
            return "TP:short_hash"
        else:
            return "FP:too_short_hex"

    # Pure decimal from here on
    # Gecko epoch constant "20100101" in user-agent string
    if s == "20100101" and "gecko" in ctx_lower:
        return "FP:gecko_epoch"

    # Attachment ID: "attachment NNNNNNN" pattern
    if re.search(r"attachment\s+" + re.escape(s), ctx_lower):
        return "TP:attachment_id"

    # Bug ID: "bug NNNNN" pattern
    if re.search(r"bug\s+" + re.escape(s), ctx_lower):
        return "TP:bug_id"

    # Build ID: 14-digit timestamp (YYYYMMDDHHmmss)
    if len(s) == 14 and s[:4].isdigit():
        try:
            year = int(s[:4])
            month = int(s[4:6])
            if 2000 <= year <= 2030 and 1 <= month <= 12:
                return "TP:build_id"
        except ValueError:
            pass

    # Regressed-by, regression range, fix for context → revision/bug reference
    if any(w in ctx_lower for w in ["regress", "fix for", "introduced", "commit", "changeset"]):
        return "TP:revision_id"

    # Version-like in build context
    if any(w in ctx_lower for w in ["build id", "build:", "buildid"]):
        return "TP:build_id"

    # 5-8 digit number near "bug" or "issue" → likely bug ID even without exact pattern
    if 5 <= len(s) <= 8:
        if any(w in ctx_lower for w in ["bug", "issue", "ticket", "report"]):
            return "TP:bug_id"

    # 7-digit number with "attachment" nearby
    if 7 <= len(s) <= 8:
        if "attachment" in ctx_lower or "screenshot" in ctx_lower:
            return "TP:attachment_id"

    # Short ambiguous number (< 7 digits, no context)
    if len(s) < 7:
        return "FP:ambiguous_number"

    # Longer number with no clear context — cautious TP
    return "TP:likely_identifier"


def refine_flag_label(surface: str, context: str) -> str:
    """Refined FLAG extraction label."""
    s = surface.strip()
    s_lower = s.lower()
    ctx_lower = context.lower()

    # Known abbreviations
    common_abbrev = {"e.g", "i.e", "etc.", "vs.", "fig.", "eq.", "ref.", "no.",
                     "st.", "nd.", "rd.", "th."}
    if s_lower.rstrip(".") in {a.rstrip(".") for a in common_abbrev}:
        return "FP:abbreviation"

    # Pure version number like "106.0b6", "10.15", "109.0"
    if re.match(r"^\d+\.\d+", s):
        return "FP:version_number"

    # Very short (2 parts, each ≤2 chars)
    parts = s.split(".")
    if len(parts) == 2 and all(len(p) <= 2 for p in parts):
        return "FP:too_short"

    # Mozilla preference paths — clearly TP
    if any(w in s_lower for w in ["browser.", "dom.", "network.", "layout.",
                                   "security.", "privacy.", "toolkit.",
                                   "devtools.", "media.", "gfx.", "ui.",
                                   "general.", "services.", "extensions."]):
        return "TP:mozilla_pref"

    # General setting/config pattern
    if any(w in s_lower for w in ["config", "setting", "enable", "disable",
                                   "preference", "pref", "flag", "option"]):
        return "TP:setting"

    # about:config context
    if "about:config" in ctx_lower or "preference" in ctx_lower:
        return "TP:likely_pref"

    # Package/namespace
    if len(parts) >= 2 and all(p.replace("_", "").replace("-", "").isalpha() for p in parts if p):
        return "TP:namespace"

    return "TP:dotted_name"


def main():
    out_dir = Path("analysis/output")
    df = pd.read_csv(out_dir / "entity_level_annotA.csv")
    print(f"Loaded {len(df)} entity rows")

    # Refine HASH labels
    hash_mask = df["entity_type"] == "HASH"
    for idx in df[hash_mask].index:
        surface = str(df.at[idx, "entity_surface"])
        context = str(df.at[idx, "gen_context"])
        df.at[idx, "annotA_extraction"] = refine_hash_label(surface, context)

    # Refine FLAG labels
    flag_mask = df["entity_type"] == "FLAG"
    for idx in df[flag_mask].index:
        surface = str(df.at[idx, "entity_surface"])
        context = str(df.at[idx, "gen_context"])
        df.at[idx, "annotA_extraction"] = refine_flag_label(surface, context)

    # Summary
    print("\n=== Refined Annotator A labels ===")
    for et in ["URL", "HASH", "FILE", "FLAG"]:
        sub = df[df.entity_type == et]
        tp = sub.annotA_extraction.str.startswith("TP").sum()
        fp = sub.annotA_extraction.str.startswith("FP").sum()
        total = len(sub)
        print(f"  {et:5s}: {total:4d} entities, TP={tp:3d} FP={fp:3d} "
              f"(precision={tp/max(1,total)*100:.1f}%)")

    print("\n--- Detailed breakdown ---")
    print(df.groupby("entity_type")["annotA_extraction"].value_counts().to_string())

    # Impact on attribution: reclassified TP entities may change UAHE counts
    print("\n--- Attribution after reclassification ---")
    tp_ents = df[df.annotA_extraction.str.startswith("TP")]
    fp_ents = df[df.annotA_extraction.str.startswith("FP")]
    print(f"  TP entities: {len(tp_ents)}")
    print(f"    IAHE={tp_ents.regex_attribution.eq('IAHE').sum()}, "
          f"RAHE={tp_ents.regex_attribution.eq('RAHE').sum()}, "
          f"UAHE={tp_ents.regex_attribution.eq('UAHE').sum()}")
    print(f"  FP entities (should be excluded from attribution): {len(fp_ents)}")
    print(f"    IAHE={fp_ents.regex_attribution.eq('IAHE').sum()}, "
          f"RAHE={fp_ents.regex_attribution.eq('RAHE').sum()}, "
          f"UAHE={fp_ents.regex_attribution.eq('UAHE').sum()}")

    # Save
    df.to_csv(out_dir / "entity_level_annotA.csv", index=False)
    print(f"\n→ Saved to {out_dir / 'entity_level_annotA.csv'}")

    # Also update blank sheet (keep same entity rows, blank labels)
    blank = df.copy()
    blank["annotA_extraction"] = ""
    blank["annotA_attribution"] = ""
    blank.to_csv(out_dir / "entity_level_blank.csv", index=False)

    # Compute micro P/R/F1 estimate (treating TP as correct extraction, FP as wrong)
    print("\n=== Estimated micro Precision (extraction only) ===")
    for et in ["URL", "HASH", "FILE", "FLAG"]:
        sub = df[df.entity_type == et]
        tp = sub.annotA_extraction.str.startswith("TP").sum()
        total = len(sub)
        print(f"  {et:5s}: precision={tp/max(1,total):.3f} ({tp}/{total})")
    tp_all = df.annotA_extraction.str.startswith("TP").sum()
    print(f"  ALL  : precision={tp_all/max(1,len(df)):.3f} ({tp_all}/{len(df)})")


if __name__ == "__main__":
    main()
