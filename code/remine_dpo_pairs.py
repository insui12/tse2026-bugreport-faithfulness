#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
remine_dpo_pairs.py

Re-mine (re-select) DPO preference pairs from a *frozen* candidate pool saved as
`*_candidates_full.jsonl.gz`, WITHOUT re-running LLM generation.

Why this exists (EMSE/TSE rationale):
- Separates stochastic candidate generation (GPU) from deterministic pair selection (CPU).
- Enables sensitivity / robustness analysis by varying selection policy knobs
  (quality-floor, margin, UAHE-cap, conditional-completeness floor, sec-completeness floor, UAHE ordering, pairs-per-prompt)
  while holding the candidate pool constant.

Expected input format (one JSON object per line, gzip-compressed):
Each line is a candidate record like those produced by your existing mining pipeline.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ------------------------------
# Small utilities (kept local)
# ------------------------------

_KNOWN_DATA_SUFFIXES = (".jsonl.gz", ".jsonl", ".gz", ".csv.gz", ".csv")


def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _strip_known_suffixes(filename: str) -> str:
    name = str(filename)
    for suf in _KNOWN_DATA_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def _sanitize_tag(tag: Optional[str]) -> str:
    if tag is None:
        return "none"
    t = str(tag).strip()
    if not t:
        return "none"
    t = re.sub(r"[^a-zA-Z0-9._-]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t or "none"


def _join_nonempty(parts: Sequence[Optional[str]], sep: str = "_") -> str:
    out: List[str] = []
    for p in parts:
        if p is None:
            continue
        s = str(p).strip()
        if not s:
            continue
        s = re.sub(r"[\\/]+", "_", s)
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        if s:
            out.append(s)
    return sep.join(out)


def _config_fingerprint(config: Dict[str, Any], length: int = 12) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[: int(length)]


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def touch(path: Path, content: str = "") -> None:
    ensure_dir(path.parent)
    path.write_text(str(content), encoding="utf-8")


def done_marker_path(data_path: Path) -> Path:
    return data_path.with_name(_strip_known_suffixes(data_path.name) + ".done")


def meta_path_for(data_path: Path) -> Path:
    return data_path.with_name(_strip_known_suffixes(data_path.name) + ".meta.json")


def lock_path_for(data_path: Path) -> Path:
    return data_path.with_name(_strip_known_suffixes(data_path.name) + ".lock")


def tmp_path_for(data_path: Path) -> Path:
    return data_path.with_name(data_path.name + ".tmp")


def _remove_if_exists(p: Path) -> None:
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


def clear_artifacts(data_path: Path) -> None:
    """Remove output artifact + its sidecars + tmp (best effort)."""
    for p in [
        data_path,
        tmp_path_for(data_path),
        meta_path_for(data_path),
        done_marker_path(data_path),
        lock_path_for(data_path),
    ]:
        _remove_if_exists(p)


def _try_git_commit(project_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def collect_env_info(project_root: Path) -> Dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "git_commit": _try_git_commit(project_root),
    }


@contextmanager
def exclusive_lock(lock_path: Path, stale_hours: float = 24.0):
    ensure_dir(lock_path.parent)
    if lock_path.exists():
        try:
            age_h = (time.time() - lock_path.stat().st_mtime) / 3600.0
        except Exception:
            age_h = 0.0
        if stale_hours is not None and float(age_h) > float(stale_hours):
            print(f"[WARN] Removing stale lock (age={age_h:.1f}h): {lock_path}")
            try:
                lock_path.unlink()
            except Exception:
                pass
        else:
            raise FileExistsError(f"Lock exists: {lock_path}")

    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, f"pid={os.getpid()} time={datetime.now().isoformat()}\n".encode("utf-8"))
    finally:
        os.close(fd)

    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass


# ------------------------------
# Filename inference (best-effort)
# ------------------------------

_RE_DPO_CAND_BASE = re.compile(
    r"^dpo_pairs_(?P<model>.+?)_seed(?P<seed>\d+)_(?P<pref>balanced|hard|structure)_(?P<align>retrieval_free|retrieval_aware)(?:_|$)"
)


def infer_context_from_filename(path: Path) -> Dict[str, Any]:
    base = _strip_known_suffixes(path.name)
    base = base.replace("_candidates_full", "").replace("_candidates", "")
    m = _RE_DPO_CAND_BASE.match(base)
    if not m:
        return {"base_name": base}
    gd = m.groupdict()
    return {
        "model_key": gd.get("model"),
        "seed": int(gd.get("seed")),
        "pref_profile": gd.get("pref"),
        "align_context": gd.get("align"),
        "base_name": base,
    }


def peek_first_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                return json.loads(line)
    except Exception:
        return None
    return None


# ------------------------------
# Pair selection logic
# ------------------------------

def _get_float(d: Any, key: str, default: float = 0.0) -> float:
    if not isinstance(d, dict):
        return float(default)
    v = d.get(key, default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _sort_candidates(group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(r: Dict[str, Any]) -> Tuple[int, float, int]:
        cr = r.get("candidate_rank", None)
        try:
            cr_i = int(cr) if cr is not None else 10**9
        except Exception:
            cr_i = 10**9

        ps = r.get("pref_score", 0.0)
        try:
            ps_f = float(ps)
        except Exception:
            ps_f = 0.0

        ci = r.get("candidate_index", None)
        try:
            ci_i = int(ci) if ci is not None else 10**9
        except Exception:
            ci_i = 10**9

        return (cr_i, -ps_f, ci_i)

    return sorted(group, key=key)


def select_pairs_from_group(
    group: List[Dict[str, Any]],
    *,
    quality_floor: float,
    margin: float,
    uahe_cap: float,
    min_cond_filled_rate: float,
    min_sec_filled_rate: float,
    enforce_uahe_order: bool,
    pairs_per_prompt: int,
    save_pref_components: bool,
) -> Tuple[List[Dict[str, Any]], str]:
    if not group:
        return [], "no_candidates"

    g_sorted = _sort_candidates(group)

    chosen = g_sorted[0]
    chosen_text = str(chosen.get("candidate_text", "")).strip()
    prompt = str(chosen.get("input_text", "")).strip()
    row_id = int(chosen.get("row_id", -1))

    chosen_score = float(chosen.get("pref_score", 0.0))
    chosen_comps = chosen.get("components", {})

    if chosen_score < float(quality_floor):
        return [], "below_quality_floor"

    # [수정됨] 환각 캡이 0.0이어도 정확히 잡히도록 >= 0.0 으로 변경
    chosen_uahe = _get_float(chosen_comps, "UAHE_per_1kTok", 0.0)
    if float(uahe_cap) >= 0.0 and chosen_uahe > float(uahe_cap):
        return [], "uahe_cap"

    chosen_cond = _get_float(chosen_comps, "CondFilledRate", _get_float(chosen_comps, "SecFilled", 0.0))
    if float(min_cond_filled_rate) > 0.0 and chosen_cond < float(min_cond_filled_rate):
        return [], "min_cond"

    # [추가됨] Universal Gate 완성을 위한 SecFilled 절대 기준 추가
    chosen_sec = _get_float(chosen_comps, "SecFilled", 0.0)
    if float(min_sec_filled_rate) > 0.0 and chosen_sec < float(min_sec_filled_rate):
        return [], "min_sec"

    need = max(1, int(pairs_per_prompt))
    rejects: List[Dict[str, Any]] = []
    seen_texts = {chosen_text}

    for rj in reversed(g_sorted):
        if rj is chosen:
            continue
        rj_text = str(rj.get("candidate_text", "")).strip()
        if not rj_text:
            continue
        if rj_text in seen_texts:
            continue

        rj_score = float(rj.get("pref_score", 0.0))
        diff = (chosen_score - rj_score)
        if diff + 1e-12 < float(margin):
            continue

        if bool(enforce_uahe_order):
            rj_comps = rj.get("components", {})
            rj_uahe = _get_float(rj_comps, "UAHE_per_1kTok", 0.0)
            if chosen_uahe > rj_uahe:
                continue

        rejects.append(rj)
        seen_texts.add(rj_text)
        if len(rejects) >= need:
            break

    if not rejects:
        return [], "no_reject"

    out: List[Dict[str, Any]] = []
    for rj in rejects:
        rj_text = str(rj.get("candidate_text", "")).strip()
        rj_score = float(rj.get("pref_score", 0.0))
        rj_comps = rj.get("components", {})

        pair = {
            "row_id": int(row_id),
            "prompt": prompt,
            "chosen": chosen_text,
            "rejected": rj_text,
            "chosen_score": float(chosen_score),
            "rejected_score": float(rj_score),
        }
        if save_pref_components:
            pair["chosen_components"] = chosen_comps
            pair["rejected_components"] = rj_comps

        out.append(pair)

    return out, "ok"


# ------------------------------
# Streaming reader (grouped by row_id)
# ------------------------------

def iter_candidate_groups(
    cand_full_path: Path,
    *,
    allow_noncontiguous: bool = False,
) -> Tuple[Iterable[Tuple[int, List[Dict[str, Any]]]], Dict[str, Any]]:
    info_holder: Dict[str, Any] = {}

    def _gen():
        nonlocal info_holder
        try:
            sha1 = hashlib.sha1()
            n_lines = 0
            versions = set()
            seen_rids = set()
            cur_rid: Optional[int] = None
            cur_group: List[Dict[str, Any]] = []

            def flush_current():
                nonlocal cur_rid, cur_group
                if cur_rid is None:
                    return None
                rid0 = int(cur_rid)
                grp0 = cur_group
                seen_rids.add(rid0)
                cur_rid = None
                cur_group = []
                return rid0, grp0

            with gzip.open(cand_full_path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    n_lines += 1
                    sha1.update((line + "\n").encode("utf-8"))
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue

                    rid = int(r.get("row_id", -1))
                    ver = r.get("pref_score_version", None)
                    if ver is not None:
                        versions.add(str(ver))

                    if cur_rid is None:
                        if rid in seen_rids:
                            raise ValueError(f"Non-contiguous row_id encountered: {rid} (already seen).")
                        cur_rid = rid

                    if rid != cur_rid:
                        flushed = flush_current()
                        if flushed is not None:
                            yield flushed
                        if rid in seen_rids:
                            raise ValueError(f"Non-contiguous row_id encountered: {rid} (already seen).")
                        cur_rid = rid
                        cur_group = []

                    cur_group.append(r)

            flushed = flush_current()
            if flushed is not None:
                yield flushed

            info_holder.clear()
            info_holder.update({
                "source_lines": int(n_lines),
                "source_sha1": sha1.hexdigest(),
                "pref_score_versions": sorted(list(versions)),
                "grouping_mode": "stream_contiguous",
            })

        except ValueError as e:
            if not allow_noncontiguous:
                raise
            print(f"[WARN] {e} -> falling back to in-memory grouping (--allow-noncontiguous).")
            sha1 = hashlib.sha1()
            n_lines = 0
            versions = set()
            by_rid: Dict[int, List[Dict[str, Any]]] = {}
            with gzip.open(cand_full_path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    n_lines += 1
                    sha1.update((line + "\n").encode("utf-8"))
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    rid = int(r.get("row_id", -1))
                    ver = r.get("pref_score_version", None)
                    if ver is not None:
                        versions.add(str(ver))
                    by_rid.setdefault(rid, []).append(r)

            info_holder.clear()
            info_holder.update({
                "source_lines": int(n_lines),
                "source_sha1": sha1.hexdigest(),
                "pref_score_versions": sorted(list(versions)),
                "grouping_mode": "buffer_all",
                "n_unique_row_id": int(len(by_rid)),
            })

            for rid in sorted(by_rid.keys()):
                yield int(rid), by_rid[rid]

    return _gen(), info_holder


# ------------------------------
# Main remine procedure
# ------------------------------

def remine_one_file(
    *,
    project_root: Path,
    cand_full_path: Path,
    out_root: Path,
    run_tag: str,
    seed_override: Optional[int],
    policy: Dict[str, Any],
    overwrite: bool,
    dry_run: bool,
    lock_stale_hours: float,
    allow_noncontiguous: bool,
    max_prompts: int,
    write_csv: bool,
) -> Optional[Path]:
    cand_full_path = cand_full_path.resolve()
    if not cand_full_path.exists():
        raise FileNotFoundError(f"candidates_full not found: {cand_full_path}")

    inferred = infer_context_from_filename(cand_full_path)
    first = peek_first_json(cand_full_path) or {}

    model_key = inferred.get("model_key") or first.get("model") or "unknown"
    pref_profile = inferred.get("pref_profile") or first.get("pref_profile") or "balanced"
    align_context = inferred.get("align_context") or first.get("align_context") or "retrieval_free"

    seed = inferred.get("seed", None)
    if seed is None:
        if seed_override is not None and int(seed_override) > 0:
            seed = int(seed_override)
        else:
            raise ValueError(
                f"Cannot infer seed from filename: {cand_full_path.name}. "
                "Either keep standard naming or pass --seed-override."
            )

    out_dir = ensure_dir(Path(out_root) / f"seed{int(seed)}" / str(pref_profile))

    cfg_for_hash = {
        "command": "remine_dpo_pairs",
        "source_candidates_full": str(cand_full_path),
        "model_key": str(model_key),
        "seed": int(seed),
        "pref_profile": str(pref_profile),
        "align_context": str(align_context),
        "policy": policy,
    }
    cfg_hash = _config_fingerprint(cfg_for_hash)
    run_id = now_id()

    out_base = _join_nonempty(
        [
            "dpo_pairs",
            str(model_key),
            f"seed{int(seed)}",
            str(pref_profile),
            str(align_context),
            "remine",
            run_tag,
            cfg_hash,
            run_id,
        ]
    )

    out_jsonl = out_dir / (out_base + ".jsonl")
    out_csv = out_dir / (out_base + ".csv")
    out_meta = meta_path_for(out_jsonl)
    out_done = done_marker_path(out_jsonl)
    out_lock = lock_path_for(out_jsonl)
    out_tmp = tmp_path_for(out_jsonl)
    out_csv_tmp = tmp_path_for(out_csv) if write_csv else None

    if out_done.exists() and out_jsonl.exists() and not overwrite:
        print(f"[REMINE] Skip existing DONE: {out_jsonl}")
        return out_jsonl

    if overwrite:
        clear_artifacts(out_jsonl)
        try:
            if out_csv.exists():
                out_csv.unlink()
        except Exception:
            pass
    else:
        if (out_jsonl.exists() or out_tmp.exists() or out_lock.exists()) and not out_done.exists():
            print(
                f"[WARN] Existing incomplete output for {out_jsonl.name}. "
                "Use --overwrite to regenerate. Skipping."
            )
            return None

    stats = {
        "prompts_total": 0,
        "prompts_kept": 0,
        "pairs_total": 0,
        "reject_reason_counts": Counter(),
        "pref_score_versions": [],
        "grouping_mode": None,
        "source_sha1": None,
        "source_lines": None,
    }

    with exclusive_lock(out_lock, stale_hours=float(lock_stale_hours)):
        gen, info_holder = iter_candidate_groups(cand_full_path, allow_noncontiguous=allow_noncontiguous)

        f_jsonl = None
        csv_writer = None
        f_csv = None

        try:
            if not dry_run:
                ensure_dir(out_jsonl.parent)
                f_jsonl = out_tmp.open("w", encoding="utf-8")

                if write_csv:
                    f_csv = out_csv_tmp.open("w", encoding="utf-8", newline="")  # type: ignore[union-attr]
                    fieldnames = ["row_id", "prompt", "chosen", "rejected", "chosen_score", "rejected_score"]
                    if bool(policy.get("save_pref_components", False)):
                        fieldnames += ["chosen_components", "rejected_components"]
                    csv_writer = csv.DictWriter(f_csv, fieldnames=fieldnames, extrasaction="ignore")
                    csv_writer.writeheader()

            for rid, group in gen:
                stats["prompts_total"] += 1
                if max_prompts > 0 and stats["prompts_total"] > int(max_prompts):
                    break

                pairs, reason = select_pairs_from_group(
                    group,
                    quality_floor=float(policy["quality_floor"]),
                    margin=float(policy["margin"]),
                    uahe_cap=float(policy["uahe_cap"]),
                    min_cond_filled_rate=float(policy["min_cond_filled_rate"]),
                    min_sec_filled_rate=float(policy.get("min_sec_filled_rate", -1.0)),
                    enforce_uahe_order=bool(policy["enforce_uahe_order"]),
                    pairs_per_prompt=int(policy["pairs_per_prompt"]),
                    save_pref_components=bool(policy["save_pref_components"]),
                )

                stats["reject_reason_counts"][reason] += 1
                if reason == "ok":
                    stats["prompts_kept"] += 1
                    stats["pairs_total"] += len(pairs)

                    if not dry_run and f_jsonl is not None:
                        for pr in pairs:
                            f_jsonl.write(json.dumps(pr, ensure_ascii=False) + "\n")
                            if csv_writer is not None:
                                row = dict(pr)
                                if "chosen_components" in row and isinstance(row["chosen_components"], dict):
                                    row["chosen_components"] = json.dumps(row["chosen_components"], ensure_ascii=False)
                                if "rejected_components" in row and isinstance(row["rejected_components"], dict):
                                    row["rejected_components"] = json.dumps(row["rejected_components"], ensure_ascii=False)
                                csv_writer.writerow(row)

            if isinstance(info_holder, dict) and info_holder:
                stats["pref_score_versions"] = info_holder.get("pref_score_versions", [])
                stats["grouping_mode"] = info_holder.get("grouping_mode", None)
                stats["source_sha1"] = info_holder.get("source_sha1", None)
                stats["source_lines"] = info_holder.get("source_lines", None)

            if stats["source_sha1"] is None or stats["source_lines"] is None:
                sha1 = hashlib.sha1()
                n_lines = 0
                with gzip.open(cand_full_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        n_lines += 1
                        sha1.update((line + "\n").encode("utf-8"))
                stats["source_sha1"] = sha1.hexdigest()
                stats["source_lines"] = int(n_lines)
                stats["grouping_mode"] = stats["grouping_mode"] or ("buffer_all" if allow_noncontiguous else "stream_contiguous")

            if dry_run:
                print(
                    f"[REMINE] DRY-RUN {cand_full_path.name}: prompts={stats['prompts_total']} "
                    f"kept={stats['prompts_kept']} pairs={stats['pairs_total']} "
                    f"reasons={dict(stats['reject_reason_counts'])}"
                )
                return None

            assert f_jsonl is not None
            f_jsonl.flush()
            os.fsync(f_jsonl.fileno())
            f_jsonl.close()
            f_jsonl = None
            os.replace(out_tmp, out_jsonl)

            if write_csv and f_csv is not None and out_csv_tmp is not None:
                f_csv.flush()
                os.fsync(f_csv.fileno())
                f_csv.close()
                f_csv = None
                os.replace(out_csv_tmp, out_csv)

            meta = {
                "command": "remine_dpo_pairs",
                "timestamp": run_id,
                "run_tag": run_tag,
                "project_root": str(project_root),
                "env": collect_env_info(project_root),
                "config_hash": cfg_hash,
                "config_for_hash": cfg_for_hash,
                "source_candidates_full": str(cand_full_path),
                "source_candidates_sha1": stats["source_sha1"],
                "source_candidates_lines": int(stats["source_lines"]) if stats["source_lines"] is not None else None,
                "model_key": str(model_key),
                "seed": int(seed),
                "pref_profile": str(pref_profile),
                "align_context": str(align_context),
                "policy": policy,
                "prompts_total": int(stats["prompts_total"]),
                "prompts_kept": int(stats["prompts_kept"]),
                "pairs": int(stats["pairs_total"]),
                "reject_reason_counts": {k: int(v) for k, v in stats["reject_reason_counts"].items()},
                "pref_score_versions": stats["pref_score_versions"],
                "grouping_mode": stats["grouping_mode"],
                "output_jsonl": str(out_jsonl),
                "output_csv": (str(out_csv) if write_csv else None),
            }
            atomic_write_json(out_meta, meta)
            touch(out_done, now_id())

            print(
                f"[REMINE] Saved: {out_jsonl} pairs={stats['pairs_total']} "
                f"(prompts_kept={stats['prompts_kept']}/{stats['prompts_total']})"
            )
            if write_csv:
                print(f"[REMINE] CSV:   {out_csv}")
            print(f"[REMINE] Meta:  {out_meta}")

            return out_jsonl

        finally:
            try:
                if f_jsonl is not None:
                    f_jsonl.close()
            except Exception:
                pass
            try:
                if f_csv is not None:
                    f_csv.close()
            except Exception:
                pass
            if not dry_run:
                _remove_if_exists(out_tmp)
                if out_csv_tmp is not None:
                    _remove_if_exists(out_csv_tmp)


# ------------------------------
# CLI
# ------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="remine_dpo_pairs.py",
        description="Re-mine DPO preference pairs from *_candidates_full.jsonl.gz by changing selection policy knobs.",
    )
    ap.add_argument("--project-root", type=str, default=".", help="Project root (for meta env + default out-root).")
    ap.add_argument(
        "--candidates-full",
        nargs="+",
        required=True,
        help="Path(s) or glob(s) to *_candidates_full.jsonl.gz",
    )

    ap.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="Output root directory. Default: <project-root>/data/dpo_pairs",
    )
    ap.add_argument("--run-tag", type=str, default=None, help="Tag to include in output filenames.")
    ap.add_argument(
        "--seed-override",
        type=int,
        default=-1,
        help="If seed cannot be inferred from filename, use this seed (otherwise ignored).",
    )

    ap.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they exist.")
    ap.add_argument("--dry-run", action="store_true", help="Only compute expected yields; do not write outputs.")
    ap.add_argument("--lock-stale-hours", type=float, default=24.0, help="Remove lock if older than this (hours).")
    ap.add_argument(
        "--allow-noncontiguous",
        action="store_true",
        help="Allow non-contiguous row_id blocks by buffering all candidates in memory (slower, higher RAM).",
    )
    ap.add_argument(
        "--max-prompts",
        type=int,
        default=0,
        help="Debug: process only first N prompt groups (0=all).",
    )
    ap.add_argument("--no-csv", action="store_true", help="Do not write CSV output (JSONL only).")

    # --- Policy knobs ---
    ap.add_argument("--quality-floor", type=float, default=0.0) # 기본값을 0.0으로 수정
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--pairs-per-prompt", type=int, default=2)
    ap.add_argument("--uahe-cap", type=float, default=-1.0, help="<0 disables.") # 기본값을 -1.0으로 수정
    ap.add_argument("--min-cond-filled-rate", type=float, default=0.6, help="<=0 disables.")
    ap.add_argument("--min-sec-filled-rate", type=float, default=-1.0, help="<=0 disables. Validates basic structure.") # 새로 추가됨
    ap.add_argument("--enforce-uahe-order", action="store_true")
    ap.add_argument("--save-pref-components", action="store_true")

    return ap


def expand_globs(patterns: List[str]) -> List[Path]:
    import glob

    out: List[Path] = []
    for pat in patterns:
        m = glob.glob(pat)
        if m:
            out.extend([Path(x) for x in m])
        else:
            if any(ch in pat for ch in ["*", "?", "["]):
                raise FileNotFoundError(f"No files matched glob: {pat}")
            out.append(Path(pat))
    seen = set()
    uniq: List[Path] = []
    for p in out:
        rp = str(Path(p).resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(Path(rp))
    return uniq


def main() -> None:
    args = build_parser().parse_args()

    project_root = Path(args.project_root).resolve()
    cand_paths = expand_globs(list(args.candidates_full))
    if not cand_paths:
        raise ValueError("No candidates_full files resolved.")

    out_root = Path(args.out_root).resolve() if args.out_root else (project_root / "data" / "dpo_pairs")

    run_tag = _sanitize_tag(args.run_tag)
    seed_override = int(args.seed_override) if int(args.seed_override) > 0 else None

    policy = {
        "quality_floor": float(args.quality_floor),
        "margin": float(args.margin),
        "uahe_cap": float(args.uahe_cap),
        "min_cond_filled_rate": float(args.min_cond_filled_rate),
        "min_sec_filled_rate": float(args.min_sec_filled_rate), # 새로 추가됨
        "enforce_uahe_order": bool(args.enforce_uahe_order),
        "pairs_per_prompt": int(args.pairs_per_prompt),
        "save_pref_components": bool(args.save_pref_components),
        "reject_strategy": "bottom_first",
    }

    print(f"[REMINE] project_root={project_root}")
    print(f"[REMINE] out_root={out_root}")
    print(f"[REMINE] files={len(cand_paths)}")
    print(f"[REMINE] policy={policy}")

    for p in cand_paths:
        remine_one_file(
            project_root=project_root,
            cand_full_path=p,
            out_root=out_root,
            run_tag=run_tag,
            seed_override=seed_override,
            policy=policy,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
            lock_stale_hours=float(args.lock_stale_hours),
            allow_noncontiguous=bool(args.allow_noncontiguous),
            max_prompts=int(args.max_prompts),
            write_csv=(not bool(args.no_csv)),
        )


if __name__ == "__main__":
    main()