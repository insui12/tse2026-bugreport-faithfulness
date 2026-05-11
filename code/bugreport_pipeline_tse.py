#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bugreport_pipeline_tse.py

Refactored single-file pipeline for TSE/EMSE-style experiments on structured bug report generation
under summary-grounded inputs and optional retrieval-augmented prompting (RAG-as-exemplar).

Design goals (TSE/EMSE-ready):
- Deterministic, metadata-first experiments (no filename parsing).
- Two-stage execution:
    GEN  : LLM generation + lightweight regex metrics + retrieval provenance
    SCORE: Offline heavy metrics (ROUGE / CTQRS / SBERT) with CPU/GPU optimized batching
- Retrieval reproducibility:
    - retriever-build (dense + lexical index artifacts)
    - retrieval-cache (top-k results for split, optional leave-one-out for train)
- Construct validity improvements:
    - Hard entities are measured via attribution (input / retrieval / unattributed), NOT "hallucination" by default.
    - ContextSupportRate / ContextUnattributedRate included for EMSE reporting (hard-entity grounding).
    - Length-normalized risk metrics included.
    - [EMSE UPDATE] Group-based split & retrieval exclusion to prevent GT-sharing data leakage.
- Hardware utilization:
    - Single GPU (98GB VRAM): one 7B model at a time is the recommended strategy.
    - Tokenizer threading + regex multiprocessing separated for maximal throughput.
    - Micro-batch autotuning for stable high GPU utilization without OOM loops.
- GEN-SUITE usability:
    - Optional --preload-model/--keep-model-loaded for multi-condition suites to reduce load overhead.

Author: (Seojin Choi)
Version: 1.6 (Feb 2026 - EMSE Patch)
"""

from __future__ import annotations

try:
    import setproctitle
    user_id = "2022810073"
    project_name = "TSE_Worker"
    setproctitle.setproctitle(f"[{user_id}] {project_name}")
except ImportError:
    pass

import argparse
import concurrent.futures
import dataclasses
import gzip
import hashlib
import json
import gc
import os
import pickle
import random
import re
import sys
import time
import platform
import socket
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from collections import Counter, defaultdict
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ==============================
# Configuration (edit if needed)
# ==============================

DEFAULT_INPUT_COL = "NEW_llama_output"
DEFAULT_GT_COL = "text"

# Local model registry (override via env vars)
MODEL_ID: Dict[str, str] = {
    "qwen2.5-7b": os.environ.get(
        "QWEN_MODEL_ID",
        "/home/selab/2026/LLM_Models/Qwen/2.5/7B-Instruct",
    ),
    "mistral-7b-v0.3": os.environ.get(
        "MISTRAL_MODEL_ID",
        "/home/selab/2026/LLM_Models/Mistral/v0.3/7B-Instruct",
    ),
    # keep others if you want
    "llama-3.2-3b": os.environ.get(
        "LLAMA_MODEL_ID",
        "/home/selab/2026/LLM_Models/LLaMa/3.2/3B-Instruct",
    ),
}

DEFAULT_DENSE_EMB_MODEL = os.environ.get("RAG_EMB_MODEL", "BAAI/bge-large-en-v1.5")
DEFAULT_SBERT_MODEL = os.environ.get("SBERT_MODEL", "sentence-transformers/all-mpnet-base-v2")

DEFAULT_RAW_XLSX_REL = Path("data/Plus14_filtered_bug_report_scores_Summary.xlsx")
SPLIT_DIR_REL = Path("data/splits")
RESULTS_DIR_REL = Path("results")
RUNS_DIR_REL = Path("results/runs")
CACHE_DIR_REL = Path("results/cache")
ADAPTER_DIR_REL = Path("adapters")
CANDIDATES_DIR_REL = Path("data/candidates")
DPO_DIR_REL = Path("data/dpo_pairs")

# Prompt budgeting defaults
DEFAULT_MAX_PROMPT_TOKENS = 4096
DEFAULT_MAX_INPUT_TOKENS = 2048
DEFAULT_MAX_EX_IN_TOKENS = 256
DEFAULT_MAX_EX_OUT_TOKENS = 512
DEFAULT_INPUT_TRUNCATION: Literal["head", "tail", "head_tail"] = "head_tail"

# Canonical required sections for completeness
REQUIRED_SECTIONS = ("steps", "expected", "actual", "environment")

# ==============================
# System prompts (TSE-aligned)
# ==============================

SYSTEM_PROMPT_TEMPLATE_ON = """You are an expert QA engineer.

Task:
Given a short issue summary (often incomplete), write a structured bug report.

Rules:
- Use ONLY information supported by the given input text and (if provided) the retrieved historical examples.
- Do NOT invent unverifiable details (URLs, commit hashes, revision IDs, exact filenames) unless explicitly present.
- If information is missing, write "not provided".
- Use the section headers below exactly once each, in markdown bold.

**affected versions**
- not provided

**affected platforms**
- not provided

**steps to reproduce**
1. not provided

**expected result**
- not provided

**actual result**
- not provided

**regression range**
- not provided

**notes**
- not provided
"""

SYSTEM_PROMPT_TEMPLATE_OFF = """You are an expert QA engineer.

Given a short issue summary (often incomplete), write a clear bug report.
Ground the report strictly in the input (and retrieved examples if provided); do not invent unverifiable details.
"""

FS_EXAMPLE_TEMPLATE = """### Example Input:
{input_text}

### Example Bug Report:
{output_text}
"""

# ==============================
# Utilities
# ==============================


def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # lazy

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def cleanup_cuda() -> None:
    """Best-effort CUDA memory cleanup."""
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _sanitize_run_tag(tag: Optional[str]) -> str:
    if tag is None:
        return "none"
    t = str(tag).strip()
    if not t:
        return "none"
    t = re.sub(r"[^a-zA-Z0-9._-]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t or "none"

# 모듈 상단에 이 워커 함수를 추가해야 다른 프로세스들이 이 함수를 들고가서 일할 수 있습니다.
def _dpo_score_worker(args_tuple):
    inp, rtxt, g, tcnt, prof, ver = args_tuple
    in_sets = hard_entity_sets(inp)
    ret_sets = hard_entity_sets(rtxt)
    sc, comps = preference_score(
        input_text=inp,
        retrieval_text=rtxt,
        output_text=g,
        resp_tokens=int(tcnt),
        pref_profile=prof,
        score_version=ver,
        input_sets=in_sets,
        retrieval_sets=ret_sets
    )
    return sc, comps, g, tcnt

def _join_nonempty(parts: Sequence[Optional[str]], sep: str = "_") -> str:
    out: List[str] = []
    for p in parts:
        if p is None:
            continue
        s = str(p).strip()
        if not s:
            continue
        s = re.sub(r"[\\/]+", "_", s)         # avoid path separators
        s = re.sub(r"\s+", "_", s)            # spaces -> _
        s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        if s:
            out.append(s)
    return sep.join(out)


def load_tokenizer_robust(
    model_id: str,
    *,
    use_fast: bool = True,
    trust_remote_code: bool = True,
    padding_side: str = "left",
    truncation_side: str = "left",
):
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=bool(use_fast),
            trust_remote_code=bool(trust_remote_code),
        )
    except Exception as e:
        if use_fast:
            print(
                f"[WARN] AutoTokenizer(use_fast=True) failed for {model_id}: {e}. "
                "Retrying with use_fast=False."
            )
            tok = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=False,
                trust_remote_code=bool(trust_remote_code),
            )
        else:
            raise

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    tok.padding_side = str(padding_side)
    tok.truncation_side = str(truncation_side)
    return tok


def _config_fingerprint(config: Dict[str, Any], length: int = 12) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[: int(length)]


def sha1_text(s: str, length: int = 12) -> str:
    return hashlib.sha1(str(s).encode("utf-8")).hexdigest()[: int(length)]

# --- EMSE Group Split & Leakage Prevention Utilities ---
_RE_WS = re.compile(r"\s+")

def _normalize_ws(s: str) -> str:
    # EMSE 관점: 동일 bug report가 개행/공백만 다른 경우까지 동일 그룹으로 묶어 누수를 원천 차단
    return _RE_WS.sub(" ", str(s).strip())

def compute_group_id_from_gt(gt_text: str) -> str:
    # 충돌 방지 위해 full sha1 사용 (40 hex chars)
    return hashlib.sha1(_normalize_ws(gt_text).encode("utf-8")).hexdigest()


def _clean_cell_to_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def jsonl_gz_write(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def jsonl_gz_iter(path: Path) -> Iterable[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def jsonl_gz_to_csv_gz(
    in_path: Path,
    out_path: Path,
    *,
    prefer_cols: Optional[List[str]] = None,
) -> None:
    prefer_cols = prefer_cols or []

    keys: set = set()
    for r in jsonl_gz_iter(in_path):
        keys.update(r.keys())

    pref_set = set(prefer_cols)
    fieldnames = list(prefer_cols) + sorted([k for k in keys if k not in pref_set])

    ensure_dir(out_path.parent)
    with gzip.open(out_path, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in jsonl_gz_iter(in_path):
            row: Dict[str, Any] = {}
            for k in fieldnames:
                v = r.get(k, "")
                if isinstance(v, (dict, list)):
                    row[k] = json.dumps(v, ensure_ascii=False)
                else:
                    row[k] = v
            w.writerow(row)


def save_meta(path: Path, meta: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_meta(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


# ==============================
# Data: load/split/cache
# ==============================

@dataclass
class DataSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


# ==============================
# Run directories + atomic artifacts (GEN/SCORE concurrency safe)
# ==============================

_KNOWN_DATA_SUFFIXES = (
    ".jsonl.gz",
    ".jsonl",
    ".gz",
    ".parquet",
    ".xlsx",
)

def _strip_known_suffixes(filename: str) -> str:
    name = str(filename)
    for suf in _KNOWN_DATA_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def sidecar_path(data_path: Path, side_suffix: str) -> Path:
    base = _strip_known_suffixes(data_path.name)
    return data_path.with_name(base + side_suffix)


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
    return sidecar_path(data_path, ".done")


def lock_path_for(data_path: Path) -> Path:
    return sidecar_path(data_path, ".lock")


def meta_path_for(data_path: Path) -> Path:
    return sidecar_path(data_path, ".meta.json")


def tmp_path_for(data_path: Path) -> Path:
    return data_path.with_name(data_path.name + ".tmp")


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


def last_run_file(project_root: Path) -> Path:
    return ensure_dir(project_root / RUNS_DIR_REL) / "LAST_RUN.txt"


def write_last_run(project_root: Path, run_dir: Path) -> None:
    atomic_write_text(last_run_file(project_root), str(run_dir), encoding="utf-8")


def read_last_run(project_root: Path) -> Optional[Path]:
    p = last_run_file(project_root)
    if not p.exists():
        return None
    s = p.read_text(encoding="utf-8").strip()
    return Path(s) if s else None


def default_run_dir(project_root: Path, run_tag: str, seed: int) -> Path:
    tag = _sanitize_run_tag(run_tag)
    return project_root / RUNS_DIR_REL / f"{now_id()}_{tag}_seed{int(seed)}"


def resolve_run_dir(
    *,
    project_root: Path,
    run_dir_arg: Optional[str],
    run_tag: str,
    seed: int,
    create: bool = True,
) -> Tuple[Path, bool]:
    arg = str(run_dir_arg).strip() if run_dir_arg is not None else "auto"
    arg_l = arg.lower()

    created_new = False
    if arg_l in {"", "auto", "new"}:
        run_dir = default_run_dir(project_root, run_tag, seed)
        created_new = True
    elif arg_l == "last":
        last = read_last_run(project_root)
        if last is None:
            raise FileNotFoundError(
                f"LAST_RUN.txt not found under {project_root / RUNS_DIR_REL}. "
                "Run GEN once (it writes LAST_RUN.txt) or pass --run-dir explicitly."
            )
        run_dir = last
        if not run_dir.is_absolute():
            run_dir = (project_root / run_dir).resolve()
    else:
        run_dir = Path(arg)
        if not run_dir.is_absolute():
            run_dir = (project_root / run_dir).resolve()

    run_dir = run_dir.resolve()

    if create:
        ensure_dir(run_dir)
        for sub in ("gen", "scored", "summary", "logs"):
            ensure_dir(run_dir / sub)

        meta_p = run_dir / "run.meta.json"
        if not meta_p.exists():
            meta = {
                "run_dir": str(run_dir),
                "created_at": now_id(),
                "project_root": str(project_root),
                "seed": int(seed),
                "run_tag": _sanitize_run_tag(run_tag),
                "env": collect_env_info(project_root),
            }
            atomic_write_json(meta_p, meta)

    if created_new:
        write_last_run(project_root, run_dir)

    return run_dir, created_new


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


def _remove_if_exists(p: Path) -> None:
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


def clear_artifacts(data_path: Path) -> None:
    for p in [
        data_path,
        tmp_path_for(data_path),
        meta_path_for(data_path),
        done_marker_path(data_path),
        lock_path_for(data_path),
    ]:
        _remove_if_exists(p)


def read_raw_dataset(
    raw_path: Path,
    input_col: str,
    gt_col: str,
    keep_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    df = pd.read_excel(raw_path)
    keep_cols = keep_cols or []
    cols = [input_col, gt_col] + [c for c in keep_cols if c in df.columns]

    # Avoid duplicate columns
    cols_unique: List[str] = []
    seen: set = set()
    for c in cols:
        if c not in seen:
            cols_unique.append(c)
            seen.add(c)
    cols = cols_unique

    for c in (input_col, gt_col):
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in {raw_path}. Columns={list(df.columns)[:30]}...")

    df = df[cols].copy()

    df[input_col] = df[input_col].map(_clean_cell_to_str)
    df[gt_col] = df[gt_col].map(_clean_cell_to_str)

    # filter empty
    df = df[(df[input_col].str.len() > 0) & (df[gt_col].str.len() > 0)].reset_index(drop=False)

    # Stable row id for reproducibility: original Excel row index after filtering
    df = df.rename(columns={"index": "row_id"})
    if "row_id" not in df.columns:
        df["row_id"] = np.arange(len(df), dtype=int)
    df["row_id"] = df["row_id"].astype(int)

    # EMSE 핵심: group_id (same-GT cluster key)
    df["group_id"] = df[gt_col].map(compute_group_id_from_gt).astype(str)

    return df


def assert_no_group_overlap(splits: DataSplits, group_col: str = "group_id") -> None:
    if group_col not in splits.train.columns or group_col not in splits.val.columns or group_col not in splits.test.columns:
        raise KeyError(f"Missing group_col='{group_col}' in one of the splits.")

    g_tr = set(splits.train[group_col].astype(str).tolist())
    g_va = set(splits.val[group_col].astype(str).tolist())
    g_te = set(splits.test[group_col].astype(str).tolist())

    inter = (g_tr & g_va) | (g_tr & g_te) | (g_va & g_te)
    if inter:
        # EMSE 투고 관점: 이건 "실험 설계 오류"에 해당하므로 경고가 아니라 실패 처리
        raise RuntimeError(
            f"[SPLIT-LEAK] group overlap detected across splits: {len(inter)} shared groups. "
            "Use group split and regenerate splits."
        )


def split_dataset(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    *,
    split_mode: Literal["group", "row"] = "group",
    group_col: str = "group_id",
) -> DataSplits:
    """
    EMSE 권장: group split (bug report/GT 단위)로 Train/Val/Test 분리하여 데이터 누수 차단.
    - split_mode="group": 같은 group_id가 여러 split에 섞이지 않음 (내적 타당성 보장)
    - split_mode="row"  : 레거시/디버그용 (누수 가능)
    """
    from sklearn.model_selection import train_test_split, GroupShuffleSplit  # type: ignore

    # 🌟 [EMSE Patch] 완벽한 시드 고정 및 결정론적 재현성을 위한 데이터프레임 초기 정렬
    df = df.sort_values(by=[group_col, "row_id"]).reset_index(drop=True)

    if split_mode == "row":
        train_df, temp_df = train_test_split(
            df, test_size=(1.0 - float(train_ratio)), random_state=int(seed), shuffle=True
        )
        rel_val = float(val_ratio) / (1.0 - float(train_ratio))
        val_df, test_df = train_test_split(temp_df, test_size=(1.0 - rel_val), random_state=int(seed), shuffle=True)
        return DataSplits(
            train=train_df.reset_index(drop=True),
            val=val_df.reset_index(drop=True),
            test=test_df.reset_index(drop=True),
        )

    # --- group split (EMSE default) ---
    if group_col not in df.columns:
        raise KeyError(f"split_mode='group' requires column '{group_col}' in dataframe.")

    groups = df[group_col].astype(str)
    gss1 = GroupShuffleSplit(n_splits=1, train_size=float(train_ratio), random_state=int(seed))
    tr_idx, tmp_idx = next(gss1.split(df, groups=groups))
    train_df = df.iloc[tr_idx].copy()
    temp_df = df.iloc[tmp_idx].copy()

    rel_val = float(val_ratio) / max(1e-9, (1.0 - float(train_ratio)))
    rel_val = min(max(rel_val, 0.0), 1.0)
    groups_tmp = temp_df[group_col].astype(str)

    gss2 = GroupShuffleSplit(n_splits=1, train_size=float(rel_val), random_state=int(seed))
    va_rel_idx, te_rel_idx = next(gss2.split(temp_df, groups=groups_tmp))
    val_df = temp_df.iloc[va_rel_idx].copy()
    test_df = temp_df.iloc[te_rel_idx].copy()

    splits = DataSplits(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )
    assert_no_group_overlap(splits, group_col=group_col)
    return splits


def cache_splits(project_root: Path, splits: DataSplits, seed: int) -> None:
    split_dir = ensure_dir(project_root / SPLIT_DIR_REL)
    splits.train.to_csv(split_dir / f"train_seed{seed}.csv", index=False)
    splits.val.to_csv(split_dir / f"val_seed{seed}.csv", index=False)
    splits.test.to_csv(split_dir / f"test_seed{seed}.csv", index=False)


def load_cached_splits(project_root: Path, seed: int) -> Optional[DataSplits]:
    split_dir = project_root / SPLIT_DIR_REL
    train_p = split_dir / f"train_seed{seed}.csv"
    val_p = split_dir / f"val_seed{seed}.csv"
    test_p = split_dir / f"test_seed{seed}.csv"
    if train_p.exists() and val_p.exists() and test_p.exists():
        train_df = pd.read_csv(train_p)
        val_df = pd.read_csv(val_p)
        test_df = pd.read_csv(test_p)
        # enforce row_id type if present
        for d in (train_df, val_df, test_df):
            if "row_id" in d.columns:
                d["row_id"] = d["row_id"].astype(int)
        return DataSplits(train=train_df, val=val_df, test=test_df)
    return None


# ==============================
# Retrieval: dense + lexical
# ==============================


@dataclass
class DenseIndexConfig:
    emb_model: str
    normalize: bool = True


@dataclass
class LexicalIndexConfig:
    backend: Literal["tfidf", "bm25"] = "tfidf"
    max_features: int = 200_000
    ngram: int = 2
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


_RE_BM25_TOKEN = re.compile(r"(?u)\b\w+\b")


def _bm25_tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _RE_BM25_TOKEN.finditer(str(text))]


def _dense_index_dir(project_root: Path, seed: int) -> Path:
    return ensure_dir(project_root / CACHE_DIR_REL / "retrieval_index" / str(seed))


def _dense_index_path(project_root: Path, seed: int, cfg: DenseIndexConfig, input_col: str) -> Path:
    key = sha1_text(json.dumps({"cfg": dataclasses.asdict(cfg), "input_col": str(input_col)}, sort_keys=True))
    return _dense_index_dir(project_root, seed) / f"dense_{key}.npz"


def _lex_index_path(project_root: Path, seed: int, cfg: LexicalIndexConfig, input_col: str) -> Path:
    key = sha1_text(json.dumps({"cfg": dataclasses.asdict(cfg), "input_col": str(input_col)}, sort_keys=True))
    return _dense_index_dir(project_root, seed) / f"lexical_{cfg.backend}_{key}.pkl"


def build_dense_index(
    *,
    project_root: Path,
    seed: int,
    train_texts: List[str],
    cfg: DenseIndexConfig,
    input_col: str = DEFAULT_INPUT_COL,
    device: Literal["cpu", "cuda", "auto"] = "cuda",
    batch_size: int = 256,
    force: bool = False,
) -> Path:
    out_path = _dense_index_path(project_root, seed, cfg, input_col)
    if out_path.exists() and not force:
        return out_path

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise RuntimeError(f"sentence-transformers required for dense retrieval: {e}")

    dev = device
    if device == "auto":
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            dev = "cpu"

    model = SentenceTransformer(cfg.emb_model, device=dev)
    try:
        embs = model.encode(
            train_texts,
            batch_size=int(batch_size),
            convert_to_numpy=True,
            normalize_embeddings=bool(cfg.normalize),
            show_progress_bar=True,
        ).astype(np.float32)
    finally:
        try:
            del model
        except Exception:
            pass
        cleanup_cuda()

    ensure_dir(out_path.parent)
    np.savez_compressed(out_path, embs=embs, emb_model=cfg.emb_model, normalize=cfg.normalize)
    return out_path


def build_lexical_index(
    *,
    project_root: Path,
    seed: int,
    train_texts: List[str],
    cfg: LexicalIndexConfig,
    input_col: str = DEFAULT_INPUT_COL,
    force: bool = False,
) -> Path:
    out_path = _lex_index_path(project_root, seed, cfg, input_col)
    if out_path.exists() and not force:
        return out_path

    if cfg.backend == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

        vec = TfidfVectorizer(
            max_features=int(cfg.max_features),
            ngram_range=(1, int(cfg.ngram)),
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        X = vec.fit_transform(train_texts)

        ensure_dir(out_path.parent)
        with out_path.open("wb") as f:
            pickle.dump({"backend": "tfidf", "vectorizer": vec, "matrix": X}, f, protocol=pickle.HIGHEST_PROTOCOL)
        return out_path

    # --- BM25 ---
    toks = [_bm25_tokenize(t) for t in train_texts]
    N = len(toks)
    doc_len = np.array([len(x) for x in toks], dtype=np.int32)
    avgdl = float(doc_len.mean()) if N > 0 else 0.0

    df: Dict[str, int] = defaultdict(int)
    postings_tmp: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    for i, tt in enumerate(toks):
        cnt = Counter(tt)
        for term, tf in cnt.items():
            df[term] += 1
            postings_tmp[term].append((int(i), int(tf)))

    idf: Dict[str, float] = {}
    for term, dfi in df.items():
        idf[term] = float(math.log(1.0 + (N - float(dfi) + 0.5) / (float(dfi) + 0.5)))

    postings: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for term, pairs in postings_tmp.items():
        docs = np.fromiter((p[0] for p in pairs), dtype=np.int32)
        tfs = np.fromiter((p[1] for p in pairs), dtype=np.float32)
        postings[term] = (docs, tfs)

    ensure_dir(out_path.parent)
    with out_path.open("wb") as f:
        pickle.dump(
            {
                "backend": "bm25",
                "k1": float(cfg.bm25_k1),
                "b": float(cfg.bm25_b),
                "avgdl": float(avgdl),
                "doc_len": doc_len,
                "n_docs": N,
                "idf": idf,
                "postings": postings,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return out_path


def _load_dense_embs(path: Path) -> np.ndarray:
    z = np.load(path)
    return z["embs"].astype(np.float32)


def _load_lexical(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "backend" not in obj and "vectorizer" in obj and "matrix" in obj:
        obj["backend"] = "tfidf"
    return obj


def _as_exclusion_list(ex: Any, n: int) -> List[int]:
    if ex is None:
        return []
    if isinstance(ex, (list, tuple, set, np.ndarray)):
        idxs = [int(x) for x in ex]
    else:
        idxs = [int(ex)]
    # filter + unique
    out = sorted({i for i in idxs if 0 <= i < int(n)})
    return out


def _topk_from_scores(row: np.ndarray, k_eff: int) -> np.ndarray:
    n = int(row.shape[0])
    k_eff = int(min(max(0, k_eff), n))
    if k_eff <= 0:
        return np.array([], dtype=int)
    if k_eff >= n:
        top = np.argsort(-row)[:k_eff]
        return top.astype(int)
    top = np.argpartition(-row, k_eff - 1)[:k_eff]
    top = top[np.argsort(-row[top])]
    return top.astype(int)


def retrieve_topk_dense(
    *,
    dense_embs: np.ndarray,
    query_texts: List[str],
    emb_model: str,
    device: Literal["cpu", "cuda", "auto"] = "cuda",
    batch_size: int = 256,
    k: int = 1,
    normalize: bool = True,
    exclude_pos: Optional[List[Any]] = None,
) -> Tuple[List[List[int]], List[List[float]]]:
    if dense_embs.ndim != 2 or dense_embs.shape[0] <= 0:
        raise ValueError(f"dense_embs must be [N,D] with N>0 (got {dense_embs.shape})")

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise RuntimeError(f"sentence-transformers required for dense retrieval: {e}")

    dev = device
    if device == "auto":
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            dev = "cpu"

    model = SentenceTransformer(emb_model, device=dev)
    try:
        q_embs = model.encode(
            query_texts,
            batch_size=int(batch_size),
            convert_to_numpy=True,
            normalize_embeddings=bool(normalize),
            show_progress_bar=False,
        ).astype(np.float32)
    finally:
        try:
            del model
        except Exception:
            pass
        cleanup_cuda()

    sims = np.dot(q_embs, dense_embs.T)  # [B, N]
    n = dense_embs.shape[0]
    k_eff = min(int(k), int(n))
    idxs_all: List[List[int]] = []
    scores_all: List[List[float]] = []

    exclude_pos = exclude_pos or [None] * len(query_texts)
    for i in range(len(query_texts)):
        row = sims[i]
        ex = exclude_pos[i]
        ex_list = _as_exclusion_list(ex, n)
        if ex_list:
            row = row.copy()
            row[ex_list] = -1e9

        top = _topk_from_scores(row, k_eff)
        idxs = top.tolist()
        scs = [float(row[j]) for j in idxs]
        idxs_all.append(idxs)
        scores_all.append(scs)

    return idxs_all, scores_all


def retrieve_topk_lexical(
    *,
    lex_obj: Dict[str, Any],
    query_texts: List[str],
    k: int = 1,
    exclude_pos: Optional[List[Any]] = None,
) -> Tuple[List[List[int]], List[List[float]]]:
    backend = str(lex_obj.get("backend", "tfidf"))
    if backend == "bm25":
        return retrieve_topk_bm25(bm25_obj=lex_obj, query_texts=query_texts, k=k, exclude_pos=exclude_pos)
    return retrieve_topk_tfidf(tfidf_obj=lex_obj, query_texts=query_texts, k=k, exclude_pos=exclude_pos)


def retrieve_topk_tfidf(
    *,
    tfidf_obj: Dict[str, Any],
    query_texts: List[str],
    k: int = 1,
    exclude_pos: Optional[List[Any]] = None,
) -> Tuple[List[List[int]], List[List[float]]]:
    from sklearn.preprocessing import normalize  # type: ignore

    vectorizer = tfidf_obj["vectorizer"]
    matrix = tfidf_obj["matrix"]

    Q = vectorizer.transform(query_texts)
    Q = normalize(Q, norm="l2", copy=False)

    M = matrix
    try:
        M_norm = normalize(M, norm="l2", copy=False)
    except Exception:
        M_norm = M

    sims = Q @ M_norm.T  # sparse [B, N]
    n = M_norm.shape[0]
    k_eff = min(int(k), int(n))
    idxs_all: List[List[int]] = []
    scores_all: List[List[float]] = []

    exclude_pos = exclude_pos or [None] * len(query_texts)

    for i in range(Q.shape[0]):
        row = sims.getrow(i).toarray().ravel()  # dense 1D
        ex = exclude_pos[i]
        ex_list = _as_exclusion_list(ex, n)
        if ex_list:
            row[ex_list] = -1e9

        top = _topk_from_scores(row, k_eff)
        idxs = top.tolist()
        scs = [float(row[j]) for j in idxs]
        idxs_all.append(idxs)
        scores_all.append(scs)

    return idxs_all, scores_all


def retrieve_topk_bm25(
    *,
    bm25_obj: Dict[str, Any],
    query_texts: List[str],
    k: int = 1,
    exclude_pos: Optional[List[Any]] = None,
) -> Tuple[List[List[int]], List[List[float]]]:
    postings: Dict[str, Tuple[np.ndarray, np.ndarray]] = bm25_obj["postings"]
    idf: Dict[str, float] = bm25_obj["idf"]
    doc_len: np.ndarray = np.asarray(bm25_obj["doc_len"], dtype=np.float32)
    avgdl = float(bm25_obj["avgdl"])
    k1 = float(bm25_obj.get("k1", 1.5))
    b = float(bm25_obj.get("b", 0.75))

    n_docs = int(bm25_obj.get("n_docs", len(doc_len)))
    k_eff = min(int(k), int(n_docs))
    idxs_all: List[List[int]] = []
    scores_all: List[List[float]] = []
    exclude_pos = exclude_pos or [None] * len(query_texts)

    for qi, qtext in enumerate(query_texts):
        q_tokens = _bm25_tokenize(qtext)
        q_counter = Counter(q_tokens)
        scores = np.zeros(n_docs, dtype=np.float32)
        for term, qtf in q_counter.items():
            post = postings.get(term)
            if post is None:
                continue
            docs, tfs = post
            denom = tfs + k1 * (1.0 - b + b * (doc_len[docs] / max(1e-9, avgdl)))
            scores[docs] += (idf.get(term, 0.0) * (tfs * (k1 + 1.0) / denom) * float(qtf)).astype(np.float32)

        ex = exclude_pos[qi]
        ex_list = _as_exclusion_list(ex, n_docs)
        if ex_list:
            scores[ex_list] = -1e9

        top = _topk_from_scores(scores, k_eff)
        idxs = top.tolist()
        scs = [float(scores[j]) for j in idxs]
        idxs_all.append(idxs)
        scores_all.append(scs)

    return idxs_all, scores_all


def retrieval_cache_path(
    project_root: Path,
    seed: int,
    split: Literal["train", "val", "test"],
    retriever_type: Literal["dense", "lexical"],
    retriever_mode: Literal["similar", "random"],
    k: int,
    cache_tag: str,
) -> Path:
    return ensure_dir(project_root / CACHE_DIR_REL / "retrieval_cache" / str(seed) / split) / (
        f"cache_{retriever_type}_{retriever_mode}_k{k}_{cache_tag}.jsonl.gz"
    )


def build_retrieval_cache(
    *,
    project_root: Path,
    seed: int,
    input_col: str = DEFAULT_INPUT_COL,
    split: Literal["train", "val", "test"],
    retriever_type: Literal["dense", "lexical"],
    retriever_mode: Literal["similar", "random"],
    k: int,
    dense_cfg: DenseIndexConfig,
    lex_cfg: LexicalIndexConfig,
    dense_device: Literal["cpu", "cuda", "auto"] = "cuda",
    dense_batch_size: int = 256,
    random_seed_offset: int = 7,
    leave_one_out: bool = False,
    exclude_same_group: bool = True,
    group_col: str = "group_id",
    force: bool = False,
) -> Path:
    """
    EMSE 권장:
    - leave_one_out=True   : 자기 자신(row_id) 제외
    - exclude_same_group=True: 동일 group_id(동일 GT) 전체 제외  <-- 핵심 (자매행 누수 차단)
    """
    splits = load_cached_splits(project_root, seed)
    if splits is None:
        raise RuntimeError("Cached splits not found. Run data-prepare first.")
    train_df = splits.train.reset_index(drop=True)
    split_df = getattr(splits, split).reset_index(drop=True)

    train_texts = train_df[input_col].astype(str).tolist()
    query_texts = split_df[input_col].astype(str).tolist()
    if len(train_texts) <= 0:
        raise RuntimeError("Train split is empty; cannot build retrieval cache.")

    # map row_id -> pos in train
    train_row_ids = train_df["row_id"].astype(int).tolist()
    pos_by_row_id = {int(rid): int(i) for i, rid in enumerate(train_row_ids)}

    # map group_id -> positions in train (for group exclusion)
    pos_by_group: Dict[str, List[int]] = {}
    if exclude_same_group:
        if group_col not in train_df.columns or group_col not in split_df.columns:
            raise KeyError(f"exclude_same_group=True requires '{group_col}' in both train and {split} splits.")
        pos_by_group = defaultdict(list)
        for i, g in enumerate(train_df[group_col].astype(str).tolist()):
            pos_by_group[str(g)].append(int(i))

    # build per-query exclusion list (may be None/int/list[int])
    exclude_pos: List[Optional[Any]] = [None] * len(split_df)
    if leave_one_out or exclude_same_group:
        query_row_ids = split_df["row_id"].astype(int).tolist()
        query_groups = split_df[group_col].astype(str).tolist() if exclude_same_group else [""] * len(split_df)

        n_train = len(train_texts)
        for i, rid in enumerate(query_row_ids):
            banned: List[int] = []
            if leave_one_out:
                p = pos_by_row_id.get(int(rid), None)
                if p is not None:
                    banned.append(int(p))
            if exclude_same_group:
                banned.extend(pos_by_group.get(str(query_groups[i]), []))
            # unique + range filter
            banned = sorted({x for x in banned if 0 <= int(x) < n_train})
            exclude_pos[i] = banned if banned else None

    cache_tag = sha1_text(
        json.dumps(
            {
                "dense": dataclasses.asdict(dense_cfg),
                "lex": dataclasses.asdict(lex_cfg),
                "leave_one_out": bool(leave_one_out),
                "exclude_same_group": bool(exclude_same_group),
                "group_col": str(group_col),
                "random_seed_offset": int(random_seed_offset),
                "input_col": str(input_col),
            },
            sort_keys=True,
        )
    )

    out_path = retrieval_cache_path(project_root, seed, split, retriever_type, retriever_mode, k, cache_tag)
    if out_path.exists() and not force:
        return out_path

    rng = random.Random(int(seed) + int(random_seed_offset))

    if retriever_mode == "random":
        n = len(train_texts)
        rows = []
        all_pos = list(range(n))
        for i, rid in enumerate(split_df["row_id"].astype(int).tolist()):
            ex = exclude_pos[i]
            banned = set(_as_exclusion_list(ex, n))
            pool = [p for p in all_pos if p not in banned]
            kk = min(int(k), len(pool))
            idxs = rng.sample(pool, k=kk) if kk > 0 else []
            rows.append({"query_row_id": int(rid), "retrieved_pos": [int(x) for x in idxs], "retrieved_scores": []})

        jsonl_gz_write(out_path, rows)
        save_meta(
            meta_path_for(out_path),
            {
                "command": "retrieval-cache",
                "timestamp": now_id(),
                "seed": int(seed),
                "split": split,
                "retriever_type": retriever_type,
                "retriever_mode": retriever_mode,
                "k": int(k),
                "cache_tag": cache_tag,
                "leave_one_out": bool(leave_one_out),
                "exclude_same_group": bool(exclude_same_group),
                "group_col": str(group_col),
                "input_col": str(input_col),
            },
        )
        return out_path

    # similar mode
    if retriever_type == "dense":
        idx_path = build_dense_index(
            project_root=project_root,
            seed=seed,
            train_texts=train_texts,
            cfg=dense_cfg,
            input_col=input_col,
            device=dense_device,
            batch_size=dense_batch_size,
            force=force,
        )
        dense_embs = _load_dense_embs(idx_path)
        idxs_all, scores_all = retrieve_topk_dense(
            dense_embs=dense_embs,
            query_texts=query_texts,
            emb_model=dense_cfg.emb_model,
            device=dense_device,
            batch_size=dense_batch_size,
            k=int(k),
            normalize=bool(dense_cfg.normalize),
            exclude_pos=exclude_pos,
        )
    else:
        lex_path = build_lexical_index(
            project_root=project_root,
            seed=seed,
            train_texts=train_texts,
            cfg=lex_cfg,
            input_col=input_col,
            force=force,
        )
        lex_obj = _load_lexical(lex_path)
        idxs_all, scores_all = retrieve_topk_lexical(
            lex_obj=lex_obj,
            query_texts=query_texts,
            k=int(k),
            exclude_pos=exclude_pos,
        )

    rows = []
    for rid, idxs, scs in zip(split_df["row_id"].astype(int).tolist(), idxs_all, scores_all):
        rows.append(
            {
                "query_row_id": int(rid),
                "retrieved_pos": [int(x) for x in idxs],
                "retrieved_scores": [float(x) for x in scs],
            }
        )

    jsonl_gz_write(out_path, rows)
    save_meta(
        meta_path_for(out_path),
        {
            "command": "retrieval-cache",
            "timestamp": now_id(),
            "seed": int(seed),
            "split": split,
            "retriever_type": retriever_type,
            "retriever_mode": retriever_mode,
            "k": int(k),
            "cache_tag": cache_tag,
            "leave_one_out": bool(leave_one_out),
            "exclude_same_group": bool(exclude_same_group),
            "group_col": str(group_col),
            "input_col": str(input_col),
            "dense_cfg": dataclasses.asdict(dense_cfg),
            "lex_cfg": dataclasses.asdict(lex_cfg),
        },
    )
    return out_path


def load_retrieval_cache(cache_path: Path) -> Dict[int, Dict[str, Any]]:
    m: Dict[int, Dict[str, Any]] = {}
    for r in jsonl_gz_iter(cache_path):
        m[int(r["query_row_id"])] = r
    return m


# ==============================
# Prompt building with budgeting
# ==============================

def build_prompt(input_text: str, system_prompt: str, examples: List[Tuple[str, str]]) -> str:
    blocks = [system_prompt.strip()]
    for ex_in, ex_out in examples:
        blocks.append(FS_EXAMPLE_TEMPLATE.format(input_text=ex_in.strip(), output_text=ex_out.strip()))
    blocks.append("### Input:\n" + input_text.strip() + "\n\n### Response:\n")
    return "\n\n".join(blocks)


def _truncate_text_to_tokens(
    tokenizer,
    text: str,
    max_tokens: int,
    keep: Literal["head", "tail", "head_tail"] = "head_tail",
) -> Tuple[str, bool]:
    if max_tokens <= 0:
        return "", (len(str(text).strip()) > 0)
    ids = tokenizer.encode(str(text), add_special_tokens=False)
    if len(ids) <= max_tokens:
        return str(text), False
    if keep == "head":
        ids2 = ids[:max_tokens]
    elif keep == "tail":
        ids2 = ids[-max_tokens:]
    else:
        head = max_tokens // 2
        tail = max_tokens - head
        ids2 = ids[:head] + ids[-tail:]
    s2 = tokenizer.decode(ids2, skip_special_tokens=True)
    return s2, True


_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_HASH = re.compile(r"\b[a-f0-9]{7,40}\b", re.IGNORECASE)
_RE_FILENAME = re.compile(r"\b[\w\-.]+\.(?:gif|png|jpg|jpeg|webp|mp4|mov|avi|log|txt|csv|json|xml)\b", re.IGNORECASE)
_RE_SETTING = re.compile(r"\b[a-zA-Z_]+\.[a-zA-Z0-9_.]+\b(?::(true|false))?", re.IGNORECASE)


def mask_hard_entities(text: str) -> str:
    t = str(text)
    t = _RE_URL.sub("<URL>", t)
    t = _RE_HASH.sub("<HASH>", t)
    t = _RE_FILENAME.sub("<FILE>", t)
    t = _RE_SETTING.sub("<FLAG>", t)
    return t


def build_prompt_budgeted(
    *,
    tokenizer,
    input_text: str,
    system_prompt: str,
    examples: List[Tuple[str, str]],
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    max_ex_in_tokens: int = DEFAULT_MAX_EX_IN_TOKENS,
    max_ex_out_tokens: int = DEFAULT_MAX_EX_OUT_TOKENS,
    input_truncation: Literal["head", "tail", "head_tail"] = DEFAULT_INPUT_TRUNCATION,
) -> Tuple[str, Dict[str, Any], List[Tuple[str, str]]]:
    original_input = str(input_text)
    inp_tr, inp_truncated = _truncate_text_to_tokens(tokenizer, original_input, int(max_input_tokens), keep=input_truncation)

    ex_tr: List[Tuple[str, str]] = []
    for ex_in, ex_out in examples:
        ex_in_tr, _ = _truncate_text_to_tokens(tokenizer, str(ex_in), int(max_ex_in_tokens), keep="head_tail")
        ex_out_tr, _ = _truncate_text_to_tokens(tokenizer, str(ex_out), int(max_ex_out_tokens), keep="head_tail")
        ex_tr.append((ex_in_tr, ex_out_tr))

    def _tok_len(txt: str) -> int:
        return len(tokenizer.encode(txt, add_special_tokens=False))

    base_prompt = build_prompt(inp_tr, system_prompt, [])
    base_tokens = _tok_len(base_prompt)

    if base_tokens > int(max_prompt_tokens):
        overhead_prompt = build_prompt("", system_prompt, [])
        overhead_tokens = _tok_len(overhead_prompt)
        budget_for_input = max(16, int(max_prompt_tokens) - overhead_tokens)
        inp_tr2, did2 = _truncate_text_to_tokens(tokenizer, original_input, budget_for_input, keep=input_truncation)
        inp_tr = inp_tr2
        inp_truncated = inp_truncated or did2
        base_prompt = build_prompt(inp_tr, system_prompt, [])
        base_tokens = _tok_len(base_prompt)

    prompt = build_prompt(inp_tr, system_prompt, ex_tr)
    prompt_tokens = _tok_len(prompt)

    examples_dropped = 0
    while ex_tr and prompt_tokens > int(max_prompt_tokens):
        ex_tr.pop()
        examples_dropped += 1
        prompt = build_prompt(inp_tr, system_prompt, ex_tr)
        prompt_tokens = _tok_len(prompt)

    examples_tokens = max(0, prompt_tokens - base_tokens)
    input_tokens = _tok_len(str(inp_tr))

    info = {
        "PromptTokens": int(prompt_tokens),
        "InputTokens": int(input_tokens),
        "ExamplesTokens": int(examples_tokens),
        "RequestedK": int(len(examples)),
        "EffectiveK": int(len(ex_tr)),
        "InputTruncated": bool(inp_truncated),
        "ExamplesDropped": int(examples_dropped),
        "MaxPromptTokens": int(max_prompt_tokens),
        "MaxInputTokens": int(max_input_tokens),
        "MaxExInTokens": int(max_ex_in_tokens),
        "MaxExOutTokens": int(max_ex_out_tokens),
        "InputTruncation": str(input_truncation),
    }
    return prompt, info, ex_tr


# ==============================
# Section parsing + metrics
# ==============================

_RE_MD_BOLD_HEADER = re.compile(r"^\*\*(.+?)\*\*\s*:?\s*(.*)$")
_RE_BRACKET_HEADER = re.compile(r"^\[(.+?)\]\s*:?\s*(.*)$")

def _canon_header(h: str) -> str:
    x = re.sub(r"[^a-z0-9 ]+", " ", h.lower()).strip()
    x = re.sub(r"\s+", " ", x)
    if "step" in x and ("reproduce" in x or "repro" in x):
        return "steps"
    if "expected" in x:
        return "expected"
    if "actual" in x:
        return "actual"
    if "affected version" in x or x in {"versions", "version"}:
        return "env_versions"
    if "affected platform" in x or "platform" in x or "tested platform" in x or x in {"os"}:
        return "env_platforms"
    if "environment" in x:
        return "environment"
    if "regression" in x:
        return "regression"
    if "note" in x or "additional" in x:
        return "notes"
    if "evidence" in x or "attachment" in x or "log" in x:
        return "evidence"
    if "summary" in x:
        return "summary"
    return x

PLACEHOLDER_PATTERNS = [
    re.compile(r"<[^>]{1,100}>"),
    re.compile(r"\b(tbd|todo)\b", re.IGNORECASE),
    re.compile(r"\b(n/a|na)\b", re.IGNORECASE),
]
UNKNOWN_PATTERNS = [
    re.compile(r"\b(not provided|unknown|unspecified|not available)\b", re.IGNORECASE),
]

def parse_sections(text: str) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    lines = str(text).splitlines()
    headers: List[Tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        s = line.strip()
        m1 = _RE_MD_BOLD_HEADER.match(s)
        if m1:
            headers.append((i, _canon_header(m1.group(1)), (m1.group(2) or "").strip()))
            continue
        m2 = _RE_BRACKET_HEADER.match(s)
        if m2:
            headers.append((i, _canon_header(m2.group(1)), (m2.group(2) or "").strip()))
            continue

    if not headers:
        return {}, {}

    headers_sorted = sorted(headers, key=lambda x: x[0])
    headers_sorted.append((len(lines), "__END__", ""))

    sections: Dict[str, List[str]] = {}
    counts: Dict[str, int] = {}

    for idx in range(len(headers_sorted) - 1):
        start_i, h, inline_rest = headers_sorted[idx]
        end_i, _, _ = headers_sorted[idx + 1]
        body_lines: List[str] = []
        if inline_rest:
            body_lines.append(inline_rest)
        body_lines.extend(lines[start_i + 1 : end_i])
        body = "\n".join(body_lines).strip()
        sections.setdefault(h, []).append(body)
        counts[h] = counts.get(h, 0) + 1
    return sections, counts

def _is_placeholder(s: str) -> bool:
    x = str(s).strip()
    if not x:
        return True
    for p in PLACEHOLDER_PATTERNS:
        if p.search(x):
            return True
    if re.fullmatch(r"[.\s]+", x):
        return True
    return False

def _is_unknown_fill(s: str) -> bool:
    x = str(s).strip()
    if not x:
        return True
    for p in UNKNOWN_PATTERNS:
        if p.search(x):
            return True
    return False

def count_steps_in_text(s: str) -> int:
    if not str(s).strip():
        return 0
    lines = [ln.strip() for ln in str(s).splitlines() if ln.strip()]
    cnt = 0
    for ln in lines:
        if re.match(r"^\d+\.\s+", ln) or re.match(r"^\d+\)\s+", ln) or re.match(r"^-\s+", ln) or re.match(r"^\*\s+", ln):
            cnt += 1
    return cnt

def _agg_env_bodies(sections: Dict[str, List[str]]) -> List[str]:
    bodies = []
    for k in ("environment", "env_versions", "env_platforms"):
        if k in sections and sections[k]:
            bodies.append(str(sections[k][0]).strip())
    return bodies

def _env_presence(sections: Dict[str, List[str]]) -> bool:
    return any(k in sections for k in ("environment", "env_versions", "env_platforms"))

def _env_unknown(bodies: List[str]) -> bool:
    if not bodies:
        return True
    return all(_is_unknown_fill(b) for b in bodies)

def _env_placeholder(bodies: List[str]) -> bool:
    if not bodies:
        return True
    return all(_is_placeholder(b) for b in bodies)

def _env_filled(bodies: List[str]) -> bool:
    if not bodies:
        return False
    for b in bodies:
        if _is_placeholder(b) or _is_unknown_fill(b):
            continue
        words = b.split()
        if len(words) >= 2:
            return True
        if re.search(r"\d", b):
            return True
    return False

def section_metrics(text: str) -> Dict[str, float]:
    sections, counts = parse_sections(text)

    present_steps = "steps" in sections
    present_expected = "expected" in sections
    present_actual = "actual" in sections
    present_env = _env_presence(sections)

    sec_presence = (present_steps + present_expected + present_actual + present_env) / float(len(REQUIRED_SECTIONS))

    filled_flags: List[bool] = []
    placeholder_flags: List[bool] = []
    unknown_flags: List[bool] = []

    # steps
    if not present_steps:
        filled_flags.append(False)
        placeholder_flags.append(True)
        unknown_flags.append(True)
    else:
        body = str(sections["steps"][0]).strip()
        ph = _is_placeholder(body)
        unk = _is_unknown_fill(body)
        placeholder_flags.append(ph)
        unknown_flags.append(unk)
        filled_flags.append(count_steps_in_text(body) >= 2)

    # expected
    if not present_expected:
        filled_flags.append(False)
        placeholder_flags.append(True)
        unknown_flags.append(True)
    else:
        body = str(sections["expected"][0]).strip()
        ph = _is_placeholder(body)
        unk = _is_unknown_fill(body)
        placeholder_flags.append(ph)
        unknown_flags.append(unk)
        filled_flags.append((not ph) and (len(body.split()) >= 5))

    # actual
    if not present_actual:
        filled_flags.append(False)
        placeholder_flags.append(True)
        unknown_flags.append(True)
    else:
        body = str(sections["actual"][0]).strip()
        ph = _is_placeholder(body)
        unk = _is_unknown_fill(body)
        placeholder_flags.append(ph)
        unknown_flags.append(unk)
        filled_flags.append((not ph) and (len(body.split()) >= 5))

    # env
    env_bodies = _agg_env_bodies(sections)
    ph_env = _env_placeholder(env_bodies)
    unk_env = _env_unknown(env_bodies)
    placeholder_flags.append(ph_env)
    unknown_flags.append(unk_env)
    filled_flags.append(_env_filled(env_bodies))

    sec_filled = float(sum(1 for x in filled_flags if x)) / float(len(REQUIRED_SECTIONS))
    placeholder_rate = float(sum(1 for x in placeholder_flags if x)) / float(len(REQUIRED_SECTIONS))
    unknown_rate = float(sum(1 for x in unknown_flags if x)) / float(len(REQUIRED_SECTIONS))

    dup_count = 0
    for k, c in counts.items():
        if k in ("steps", "expected", "actual", "environment", "env_versions", "env_platforms", "regression", "notes", "evidence"):
            dup_count += max(0, int(c) - 1)

    word_count = len(str(text).split())
    steps_count = count_steps_in_text(sections["steps"][0]) if "steps" in sections else 0

    return {
        "SecPresence": float(sec_presence),
        "SecFilled": float(sec_filled),
        "PlaceholderRate": float(placeholder_rate),
        "UnknownRate": float(unknown_rate),
        "DupSectionCount": float(dup_count),
        "WordCount": float(word_count),
        "StepsCount": float(steps_count),
    }

# ==============================
# Input-signal-aware section completeness (EMSE construct-valid)
# ==============================

_RE_SIG_STEPS = re.compile(r"\b(step|repro|reproduce|steps)\b|(\d+\.\s)", re.IGNORECASE)
_RE_SIG_EXPECTED = re.compile(r"\bexpected\b|should\b|supposed to\b", re.IGNORECASE)
_RE_SIG_ACTUAL = re.compile(r"\bactual\b|instead\b|but\b|error\b|exception\b|crash\b|fails?\b", re.IGNORECASE)
_RE_SIG_ENV = re.compile(
    r"\b(os|windows|linux|ubuntu|macos|android|ios|version|v\d+(\.\d+)+|build|device|browser|chrome|firefox|safari)\b",
    re.IGNORECASE,
)

def infer_input_signals(input_text: str) -> Dict[str, int]:
    t = str(input_text)
    return {
        "steps": int(bool(_RE_SIG_STEPS.search(t))),
        "expected": int(bool(_RE_SIG_EXPECTED.search(t))),
        "actual": int(bool(_RE_SIG_ACTUAL.search(t))),
        "environment": int(bool(_RE_SIG_ENV.search(t))),
    }

def section_alignment_metrics2(input_text: str, output_text: str) -> Dict[str, float]:
    sig = infer_input_signals(input_text)
    sec = section_metrics(output_text)
    sections, _counts = parse_sections(output_text)

    def _is_filled_body_for(section_key: str) -> bool:
        if section_key == "environment":
            env_bodies = _agg_env_bodies(sections)
            return _env_filled(env_bodies)
        if section_key not in sections or not sections[section_key]:
            return False
        body = str(sections[section_key][0]).strip()
        if _is_placeholder(body) or _is_unknown_fill(body):
            return False
        if section_key == "steps":
            return count_steps_in_text(body) >= 2
        return len(body.split()) >= 5

    def _is_unknown_ok_for(section_key: str) -> bool:
        if section_key == "environment":
            env_bodies = _agg_env_bodies(sections)
            return _env_unknown(env_bodies) or _env_placeholder(env_bodies)
        if section_key not in sections or not sections[section_key]:
            return True
        body = str(sections[section_key][0]).strip()
        return _is_unknown_fill(body) or _is_placeholder(body)

    req = ["steps", "expected", "actual", "environment"]
    signal_secs   = [k for k in req if int(sig.get(k, 0)) == 1]
    nosignal_secs = [k for k in req if int(sig.get(k, 0)) == 0]

    sig_filled = sum(1 for k in signal_secs if _is_filled_body_for(k))
    nosig_ok   = sum(1 for k in nosignal_secs if _is_unknown_ok_for(k))

    n_sig = len(signal_secs)
    n_nosig = len(nosignal_secs)

    sig_rate = 1.0 if n_sig == 0 else (sig_filled / float(n_sig))
    nosig_rate = 1.0 if n_nosig == 0 else (nosig_ok / float(n_nosig))

    w_sig = 0.7 if n_sig > 0 else 0.0
    w_nosig = 0.3 if n_nosig > 0 else 0.0
    w_sum = w_sig + w_nosig
    cond_rate = (w_sig * sig_rate + w_nosig * nosig_rate) / (w_sum if w_sum > 0 else 1.0)

    return {
        "InputSignal_Steps": float(sig.get("steps", 0)),
        "InputSignal_Expected": float(sig.get("expected", 0)),
        "InputSignal_Actual": float(sig.get("actual", 0)),
        "InputSignal_Env": float(sig.get("environment", 0)),
        "SignalCount": float(n_sig),
        "NoSignalCount": float(n_nosig),
        "SignalFilledRate": float(sig_rate),
        "NoSignalFilledRate": float(nosig_rate),
        "CondFilledRate": float(cond_rate),
        "SecFilled": float(sec.get("SecFilled", 0.0)),
        "SecPresence": float(sec.get("SecPresence", 0.0)),
    }

# ==============================
# Hard entity attribution (TSE/EMSE construct-valid)
# ==============================

def extract_hard_entities(text: str) -> Dict[str, List[str]]:
    t = str(text)
    files = _RE_FILENAME.findall(t)
    file_set = set(files)
    flags = [m.group(0) for m in _RE_SETTING.finditer(t) if m.group(0) not in file_set]
    return {
        "url": _RE_URL.findall(t),
        "hash": _RE_HASH.findall(t),
        "file": files,
        "flag": flags,
    }

def hard_entity_sets(text: str) -> Dict[str, set]:
    d = extract_hard_entities(text)
    return {k: set(v) for k, v in d.items()}

def _hard_attribution_from_sets(
    in_sets: Dict[str, set],
    ret_sets: Dict[str, set],
    out_sets: Dict[str, set],
    *,
    resp_tokens: int,
) -> Dict[str, float]:
    kinds = ["url", "hash", "file", "flag"]
    out_total = 0
    in_attr_total = 0
    ret_attr_total = 0
    unat_total = 0
    feats: Dict[str, float] = {}

    for k in kinds:
        H_in = in_sets.get(k, set())
        H_ret = ret_sets.get(k, set())
        H_out = out_sets.get(k, set())

        out_cnt = len(H_out)
        in_attr = len(H_out & H_in)
        ret_attr = len((H_out - H_in) & H_ret)
        unat = len(H_out - (H_in | H_ret))

        out_total += out_cnt
        in_attr_total += in_attr
        ret_attr_total += ret_attr
        unat_total += unat

        feats[f"HardIn_{k.upper()}"] = float(len(H_in))
        feats[f"HardRet_{k.upper()}"] = float(len(H_ret))
        feats[f"HardOut_{k.upper()}"] = float(out_cnt)
        feats[f"HardOut_InputAttributed_{k.upper()}"] = float(in_attr)
        feats[f"HardOut_RetrievalAttributed_{k.upper()}"] = float(ret_attr)
        feats[f"HardOut_Unattributed_{k.upper()}"] = float(unat)

    IUHE = float(ret_attr_total + unat_total)
    feats["HardOut_Total"] = float(out_total)
    feats["IAHE_Total"] = float(in_attr_total)
    feats["RAHE_Total"] = float(ret_attr_total)
    feats["UAHE_Total"] = float(unat_total)
    feats["IUHE_Total"] = IUHE

    feats["TransferRate"] = float(ret_attr_total) / max(1.0, IUHE)
    feats["UnattributedRate"] = float(unat_total) / max(1.0, float(out_total))

    feats["ContextSupportRate"] = float(in_attr_total + ret_attr_total) / max(1.0, float(out_total))
    feats["ContextUnattributedRate"] = float(unat_total) / max(1.0, float(out_total))

    tok_denom = max(1.0, float(resp_tokens))
    scale = 1000.0 / tok_denom
    feats["RespTokens"] = float(resp_tokens)
    feats["UAHE_per_1kTok"] = float(unat_total) * scale
    feats["IUHE_per_1kTok"] = float(IUHE) * scale
    feats["RAHE_per_1kTok"] = float(ret_attr_total) * scale
    return feats


def hard_attribution_metrics(
    input_text: str,
    retrieval_text: str,
    output_text: str,
    resp_tokens: int,
) -> Dict[str, float]:
    return _hard_attribution_from_sets(
        hard_entity_sets(input_text),
        hard_entity_sets(retrieval_text),
        hard_entity_sets(output_text),
        resp_tokens=int(resp_tokens),
    )

def hard_attribution_metrics_precomputed(
    *,
    input_sets: Dict[str, set],
    retrieval_sets: Dict[str, set],
    output_text: str,
    resp_tokens: int,
) -> Dict[str, float]:
    return _hard_attribution_from_sets(
        input_sets,
        retrieval_sets,
        hard_entity_sets(output_text),
        resp_tokens=int(resp_tokens),
    )

# ==============================
# Similarity normalization for scoring
# ==============================

def normalize_for_similarity(text: str) -> str:
    t = str(text)
    t = _RE_URL.sub("<URL>", t)
    t = _RE_HASH.sub("<HASH>", t)
    t = _RE_FILENAME.sub("<FILE>", t)
    return t

# ==============================
# LLM loading / generation (GPU)
# ==============================

@dataclass
class LLMConfig:
    model_key: str
    model_id: str
    adapter_path: Optional[Path]
    load_in_4bit: bool
    torch_dtype: Any  # torch.dtype
    device_map: str = "auto"


def _primary_device(model) -> str:
    dev = getattr(model, "device", None)
    if dev is not None:
        try:
            return str(dev)
        except Exception:
            pass
    try:
        import torch  # noqa
        p = next(model.parameters())
        return str(p.device)
    except Exception:
        return "cpu"


def load_llm_and_tokenizer(cfg: LLMConfig):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError(f"Missing torch/transformers: {e}")

    use_4bit = bool(cfg.load_in_4bit)
    quant_config = None
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except Exception:
            print("[WARN] BitsAndBytesConfig not available; disabling 4bit quantization.")
            use_4bit = False
        else:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=cfg.torch_dtype,
            )

    base_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if cfg.device_map:
        base_kwargs["device_map"] = cfg.device_map

    def _load_with_kwargs(kwargs: Dict[str, Any]):
        return AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)

    def _load_with_fallback(kwargs: Dict[str, Any]):
        try:
            return _load_with_kwargs(kwargs)
        except Exception as e:
            msg = str(e).lower()
            if ("accelerate" in msg) or ("device_map" in msg):
                kwargs2 = dict(kwargs)
                kwargs2.pop("device_map", None)
                model2 = _load_with_kwargs(kwargs2)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                try:
                    model2 = model2.to(device)
                except Exception:
                    pass
                return model2
            raise

    load_kwargs = dict(base_kwargs)
    if use_4bit and quant_config is not None:
        load_kwargs["quantization_config"] = quant_config
    else:
        load_kwargs["torch_dtype"] = cfg.torch_dtype

    try:
        model = _load_with_fallback(load_kwargs)
    except Exception as e:
        if use_4bit:
            print(f"[WARN] 4bit load failed ({e}); retrying without quantization.")
            load_kwargs2 = dict(base_kwargs)
            load_kwargs2["torch_dtype"] = cfg.torch_dtype
            model = _load_with_fallback(load_kwargs2)
        else:
            raise

    tok = load_tokenizer_robust(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    tok.truncation_side = "left"

    if cfg.adapter_path is not None:
        try:
            from peft import PeftModel  # type: ignore
        except Exception as e:
            raise RuntimeError(f"PEFT not installed but adapter_path given: {e}")
        model = PeftModel.from_pretrained(model, str(cfg.adapter_path))

    model.eval()
    return model, tok


def generate_texts_with_token_counts(
    model,
    tokenizer,
    prompts: List[str],
    *,
    max_new_tokens: int,
    decode: Literal["greedy", "sample"] = "greedy",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_prompt_tokens: Optional[int] = None,
) -> Tuple[List[str], List[int]]:
    import torch

    tok = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=(max_prompt_tokens is not None),
        max_length=(int(max_prompt_tokens) if max_prompt_tokens is not None else None),
    )

    dev = _primary_device(model)
    tok = {k: v.to(dev) for k, v in tok.items()}
    input_len = tok["input_ids"].shape[1]

    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if decode == "greedy":
        gen_kwargs.update(dict(do_sample=False, temperature=0.0))
    else:
        gen_kwargs.update(dict(do_sample=True, temperature=float(temperature), top_p=float(top_p)))

    with torch.inference_mode():
        out = model.generate(**tok, **gen_kwargs)

    gen_ids = out[:, input_len:]
    gen_len = int(gen_ids.shape[1])

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        counts = [gen_len] * int(gen_ids.shape[0])
    else:
        idx = torch.arange(gen_len, device=gen_ids.device).unsqueeze(0).expand(gen_ids.size(0), gen_len)
        idx2 = torch.where(gen_ids.eq(int(eos_id)), idx, torch.full_like(idx, gen_len))
        first_eos = idx2.min(dim=1).values
        counts = [int(x) for x in first_eos.tolist()]

    decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    cleaned = [d.strip() for d in decoded]
    return cleaned, counts


def generate_microbatched(
    model,
    tokenizer,
    prompts: List[str],
    *,
    max_new_tokens: int,
    decode: Literal["greedy", "sample"],
    temperature: float,
    top_p: float,
    max_prompt_tokens: Optional[int],
    micro_batch: int,
) -> Tuple[List[str], List[int]]:
    import torch

    outputs: List[str] = []
    counts: List[int] = []
    i = 0
    mb = max(1, int(micro_batch))
    while i < len(prompts):
        cur = min(mb, len(prompts) - i)
        try:
            out, cnt = generate_texts_with_token_counts(
                model,
                tokenizer,
                prompts[i : i + cur],
                max_new_tokens=max_new_tokens,
                decode=decode,
                temperature=temperature,
                top_p=top_p,
                max_prompt_tokens=max_prompt_tokens,
            )
            outputs.extend(out)
            counts.extend(cnt)
            i += cur
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if cur <= 1:
                raise
            mb = max(1, cur // 2)
            print(f"[WARN] OOM during generate(); reducing micro_batch -> {mb} and retrying...")
    return outputs, counts


def generate_texts_with_token_counts_multi(
    model,
    tokenizer,
    prompts: List[str],
    *,
    max_new_tokens: int,
    num_return_sequences: int,
    decode: Literal["sample"] = "sample",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_prompt_tokens: Optional[int] = None,
) -> Tuple[List[str], List[int]]:
    import torch
    nrs = max(1, int(num_return_sequences))
    if decode != "sample":
        raise ValueError("generate_texts_with_token_counts_multi only supports decode='sample'.")

    tok = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=(max_prompt_tokens is not None),
        max_length=(int(max_prompt_tokens) if max_prompt_tokens is not None else None),
    )

    dev = _primary_device(model)
    tok = {k: v.to(dev) for k, v in tok.items()}
    input_len = tok["input_ids"].shape[1]

    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        num_return_sequences=nrs,
    )

    with torch.inference_mode():
        out = model.generate(**tok, **gen_kwargs)

    gen_ids = out[:, input_len:]
    gen_len = int(gen_ids.shape[1])

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        counts = [gen_len] * int(gen_ids.shape[0])
    else:
        idx = torch.arange(gen_len, device=gen_ids.device).unsqueeze(0).expand(gen_ids.size(0), gen_len)
        idx2 = torch.where(gen_ids.eq(int(eos_id)), idx, torch.full_like(idx, gen_len))
        first_eos = idx2.min(dim=1).values
        counts = [int(x) for x in first_eos.tolist()]

    decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    cleaned = [d.strip() for d in decoded]
    return cleaned, counts


def generate_microbatched_multi(
    model,
    tokenizer,
    prompts: List[str],
    *,
    max_new_tokens: int,
    num_return_sequences: int,
    temperature: float,
    top_p: float,
    max_prompt_tokens: Optional[int],
    micro_batch_seqs: int,
) -> Tuple[List[str], List[int]]:
    """
    EMSE-safe multi-return micro-batching.

    Key guarantee:
    - Always returns exactly len(prompts) * num_return_sequences outputs.
    - Never silently changes num_return_sequences (candidate count consistency for DPO).
    - On OOM:
        (1) reduce prompts_per_call,
        (2) if still OOM for a single prompt, chunk the sequences but keep total count.
    """
    import torch

    outputs: List[str] = []
    counts: List[int] = []

    nrs_total = max(1, int(num_return_sequences))
    seq_mb = int(micro_batch_seqs) if int(micro_batch_seqs) > 0 else nrs_total
    seq_mb = max(nrs_total, seq_mb)

    def _gen_one_prompt_in_chunks(prompt: str, total_nrs: int) -> Tuple[List[str], List[int]]:
        remain = int(total_nrs)
        out_all: List[str] = []
        cnt_all: List[int] = []

        chunk = int(total_nrs)
        while remain > 0:
            chunk = min(chunk, remain)
            try:
                out, cnt = generate_texts_with_token_counts_multi(
                    model,
                    tokenizer,
                    [prompt],
                    max_new_tokens=int(max_new_tokens),
                    num_return_sequences=int(chunk),
                    decode="sample",
                    temperature=float(temperature),
                    top_p=float(top_p),
                    max_prompt_tokens=max_prompt_tokens,
                )
                if len(out) != chunk or len(cnt) != chunk:
                    raise RuntimeError(
                        f"[GEN-CHUNK] unexpected chunk length: got {len(out)} outputs for chunk={chunk}"
                    )
                out_all.extend(out)
                cnt_all.extend(cnt)
                remain -= chunk
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if chunk <= 1:
                    # even 1 sample OOM -> cannot proceed safely
                    raise
                chunk = max(1, chunk // 2)
                print(f"[WARN] OOM for single prompt; reducing chunk num_return_sequences -> {chunk} and retrying...")

        if len(out_all) != total_nrs or len(cnt_all) != total_nrs:
            raise RuntimeError(
                f"[GEN-CHUNK] final length mismatch: expected {total_nrs}, got {len(out_all)}"
            )
        return out_all, cnt_all

    prompts_per_call = max(1, seq_mb // nrs_total)

    i = 0
    while i < len(prompts):
        cur = min(int(prompts_per_call), len(prompts) - i)
        batch_prompts = prompts[i : i + cur]
        try:
            out, cnt = generate_texts_with_token_counts_multi(
                model,
                tokenizer,
                batch_prompts,
                max_new_tokens=int(max_new_tokens),
                num_return_sequences=int(nrs_total),
                decode="sample",
                temperature=float(temperature),
                top_p=float(top_p),
                max_prompt_tokens=max_prompt_tokens,
            )
            expected = cur * nrs_total
            if len(out) != expected or len(cnt) != expected:
                raise RuntimeError(
                    f"[GEN-MULTI] length mismatch: expected {expected} outputs "
                    f"(cur={cur} * nrs={nrs_total}), got {len(out)}"
                )

            outputs.extend(out)
            counts.extend(cnt)
            i += cur

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

            if cur > 1:
                prompts_per_call = max(1, cur // 2)
                print(f"[WARN] OOM during multi-generate(); reducing prompts_per_call -> {prompts_per_call} and retrying...")
                continue

            # cur == 1: chunk sequences, but KEEP total_nrs
            print("[WARN] OOM with a single prompt batch; chunking sequences while preserving total num_return_sequences...")
            out1, cnt1 = _gen_one_prompt_in_chunks(batch_prompts[0], nrs_total)
            outputs.extend(out1)
            counts.extend(cnt1)
            i += 1

    # Final hard guarantee (prevents silent corruption in DPO indexing)
    final_expected = len(prompts) * nrs_total
    if len(outputs) != final_expected or len(counts) != final_expected:
        raise RuntimeError(
            f"[GEN-MULTI] FINAL length mismatch: expected {final_expected}, got {len(outputs)}"
        )

    return outputs, counts


def autotune_seq_micro_batch_multi(
    model,
    tokenizer,
    sample_prompts: List[str],
    *,
    max_new_tokens: int,
    num_return_sequences: int,
    temperature: float,
    top_p: float,
    max_prompt_tokens: Optional[int],
    start_seq_mb: int,
    max_seq_mb: int,
) -> int:
    import torch

    nrs = max(1, int(num_return_sequences))
    seq_mb = max(nrs, int(start_seq_mb))
    max_seq_mb = max(seq_mb, int(max_seq_mb))

    if not sample_prompts:
        return seq_mb

    best = seq_mb
    n_src = len(sample_prompts)

    while seq_mb <= int(max_seq_mb):
        prompts_per_call = max(1, int(seq_mb) // nrs)
        rep = (sample_prompts * ((prompts_per_call + n_src - 1) // n_src))[:prompts_per_call]
        try:
            _ = generate_texts_with_token_counts_multi(
                model,
                tokenizer,
                rep,
                max_new_tokens=max_new_tokens,
                num_return_sequences=nrs,
                temperature=temperature,
                top_p=top_p,
                max_prompt_tokens=max_prompt_tokens,
            )
            best = seq_mb
            seq_mb = seq_mb * 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
        except Exception:
            break
        finally:
            cleanup_cuda()

    return best


def autotune_micro_batch(
    model,
    tokenizer,
    sample_prompts: List[str],
    *,
    max_new_tokens: int,
    decode: Literal["greedy", "sample"],
    temperature: float,
    top_p: float,
    max_prompt_tokens: Optional[int],
    start_mb: int,
    max_mb: int,
) -> int:
    import torch

    mb = max(1, int(start_mb))
    if not sample_prompts:
        return mb

    best = mb
    n_src = len(sample_prompts)

    while mb <= int(max_mb):
        rep = (sample_prompts * ((mb + n_src - 1) // n_src))[:mb]
        try:
            _ = generate_texts_with_token_counts(
                model,
                tokenizer,
                rep,
                max_new_tokens=max_new_tokens,
                decode=decode,
                temperature=temperature,
                top_p=top_p,
                max_prompt_tokens=max_prompt_tokens,
            )
            best = mb
            mb = mb * 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
        except Exception:
            break
        finally:
            cleanup_cuda()
    return best


# ==============================
# GEN stage: generation + lightweight metrics
# ==============================


def _infer_adapter_path(
    project_root: Path,
    model_key: str,
    adapter_type: str,
    pref_profile: str,
    align_context: str,
) -> Optional[Path]:
    adapter_type = str(adapter_type)
    if adapter_type == "base":
        return None
    if adapter_type == "sft":
        root = project_root / ADAPTER_DIR_REL / "sft" / model_key
    elif adapter_type == "dpo":
        root1 = project_root / ADAPTER_DIR_REL / "dpo" / align_context / pref_profile / model_key
        root2 = project_root / ADAPTER_DIR_REL / "dpo" / pref_profile / model_key
        root = root1 if root1.exists() else root2
    else:
        raise ValueError(f"Unknown adapter_type: {adapter_type}")

    latest_txt = root / "LATEST_ADAPTER.txt"
    if latest_txt.exists():
        p = Path(latest_txt.read_text(encoding="utf-8").strip())
        if p.exists():
            return p

    runs = sorted([d for d in root.glob("RUN_*") if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _resolve_precision(precision: str):
    try:
        import torch
    except Exception as e:
        raise RuntimeError(f"torch is required for generation: {e}")

    has_cuda = torch.cuda.is_available()
    bf16_supported = False
    try:
        bf16_supported = bool(has_cuda and torch.cuda.is_bf16_supported())
    except Exception:
        bf16_supported = False

    prec = str(precision).lower().strip()
    if prec not in {"auto", "4bit", "fp16", "bf16"}:
        raise ValueError(f"Unknown precision: {precision}")

    device_map = "auto"

    if prec == "4bit":
        if not has_cuda:
            print("[WARN] --precision 4bit requested but CUDA not available; falling back to fp32 on CPU.")
            return False, torch.float32, device_map
        dtype = torch.bfloat16 if bf16_supported else torch.float16
        return True, dtype, device_map

    if prec == "fp16":
        dtype = torch.float16 if has_cuda else torch.float32
        return False, dtype, device_map

    if prec == "bf16":
        dtype = torch.bfloat16 if bf16_supported else (torch.float16 if has_cuda else torch.float32)
        return False, dtype, device_map

    if has_cuda:
        dtype = torch.bfloat16 if bf16_supported else torch.float16
        return True, dtype, device_map
    return False, torch.float32, device_map


def gen_run(
    *,
    project_root: Path,
    run_dir: Path,
    seed: int,
    run_tag: str,
    models: List[str],
    input_col: str,
    gt_col: str,
    adapter_type: Literal["base", "sft", "dpo"],
    pref_profile: Literal["none", "balanced", "hard", "structure"],
    align_context: Literal["none", "retrieval_free", "retrieval_aware"],
    template: Literal["on", "off"],
    retriever_type: Literal["none", "dense", "lexical"],
    retriever_mode: Literal["none", "similar", "random"],
    k: int,
    retrieval_mask: Literal["none", "hardmask"],
    retrieval_cache: str,
    dense_emb_model: str,
    lex_backend: Literal["tfidf", "bm25"],
    tfidf_max_features: int,
    tfidf_ngram: int,
    bm25_k1: float,
    bm25_b: float,
    decode: Literal["greedy", "sample"],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    precision: Literal["auto", "4bit", "fp16", "bf16"],
    macro_batch: int,
    micro_batch: int,
    autotune_mb: bool,
    tokenizer_threads: int,
    regex_processes: int,
    pipeline_prefetch: int,
    max_prompt_tokens: int,
    max_input_tokens: int,
    max_ex_in_tokens: int,
    max_ex_out_tokens: int,
    input_truncation: Literal["head", "tail", "head_tail"],
    limit: int,
    export_csv: bool,
    overwrite: bool = False,
    lock_stale_hours: float = 24.0,
) -> List[Path]:
    run_tag = _sanitize_run_tag(run_tag)
    set_seed(seed)

    if adapter_type == "dpo":
        if pref_profile == "none":
            raise ValueError("--pref-profile must be one of {balanced,hard,structure} when --adapter dpo")
        if align_context not in {"retrieval_free", "retrieval_aware"}:
            raise ValueError("--align-context must be retrieval_free or retrieval_aware when --adapter dpo")

    run_dir = Path(run_dir).resolve()
    gen_dir = ensure_dir(run_dir / "gen")
    ensure_dir(run_dir / "logs")

    dense_cfg = DenseIndexConfig(emb_model=str(dense_emb_model), normalize=True)
    lex_cfg = LexicalIndexConfig(
        backend=str(lex_backend),
        max_features=int(tfidf_max_features),
        ngram=int(tfidf_ngram),
        bm25_k1=float(bm25_k1),
        bm25_b=float(bm25_b),
    )

    if int(k) <= 0 or retriever_type == "none":
        retriever_type = "none"
        retriever_mode = "none"
        k = 0
    else:
        if retriever_mode == "none":
            raise ValueError("--retriever-mode must be 'similar' or 'random' when --retriever-type != none and k>0")

    splits = load_cached_splits(project_root, seed)
    if splits is None:
        raise RuntimeError("Cached splits not found. Run data-prepare first.")
    test_df = splits.test.reset_index(drop=True)
    train_df = splits.train.reset_index(drop=True)
    if limit and limit > 0:
        test_df = test_df.iloc[: int(limit)].copy()
    expected_rows = int(len(test_df))

    cache_map: Optional[Dict[int, Dict[str, Any]]] = None
    cache_path_used: Optional[str] = None
    if retriever_type != "none" and int(k) > 0:
        if retrieval_cache != "auto":
            cache_path = Path(retrieval_cache)
        else:
            cand_dir = project_root / CACHE_DIR_REL / "retrieval_cache" / str(seed) / "test"
            if not cand_dir.exists():
                raise FileNotFoundError(
                    "retrieval-cache not found. Run retrieval-cache first or pass --retrieval-cache explicitly."
                )
            patt = f"cache_{retriever_type}_{retriever_mode}_k{int(k)}_*.jsonl.gz"
            cands_all = sorted(cand_dir.glob(patt), key=lambda p: p.stat().st_mtime, reverse=True)

            cands: List[Path] = []
            for cp in cands_all:
                meta_candidates = [cp.with_suffix(".meta.json"), meta_path_for(cp)]
                meta = None
                for mpath in meta_candidates:
                    if mpath.exists():
                        meta = load_meta(mpath)
                        if meta:
                            break

                if meta and meta.get("input_col") and str(meta.get("input_col")) != str(input_col):
                    continue

                if meta and retriever_type == "dense":
                    mdcfg = meta.get("dense_cfg", {})
                    if isinstance(mdcfg, dict) and mdcfg.get("emb_model") and str(mdcfg.get("emb_model")) != str(dense_cfg.emb_model):
                        continue
                if meta and retriever_type == "lexical":
                    mlcfg = meta.get("lex_cfg", {})
                    if isinstance(mlcfg, dict):
                        if mlcfg.get("backend") and str(mlcfg.get("backend")) != str(lex_cfg.backend):
                            continue
                        if str(lex_cfg.backend) == "bm25":
                            if mlcfg.get("bm25_k1") is not None and float(mlcfg.get("bm25_k1")) != float(lex_cfg.bm25_k1):
                                continue
                            if mlcfg.get("bm25_b") is not None and float(mlcfg.get("bm25_b")) != float(lex_cfg.bm25_b):
                                continue
                cands.append(cp)

            if not cands:
                raise FileNotFoundError(
                    f"No retrieval cache matching pattern {patt} (input_col={input_col}) under {cand_dir}"
                )
            cache_path = cands[0]

        cache_map = load_retrieval_cache(cache_path)
        cache_path_used = str(cache_path)

    train_inputs = train_df[input_col].astype(str).tolist()
    train_outputs = train_df[gt_col].astype(str).tolist()

    outputs: List[Path] = []

    for model_key in models:
        if model_key not in MODEL_ID:
            raise KeyError(f"Unknown model key: {model_key}. Available: {list(MODEL_ID.keys())}")

        model_id = MODEL_ID[model_key]
        use_4bit, torch_dtype, device_map = _resolve_precision(precision)
        if precision in {"bf16", "fp16"}:
            use_4bit = False

        adapter_path = _infer_adapter_path(project_root, model_key, adapter_type, pref_profile, align_context)
        if adapter_type != "base" and (adapter_path is None or not adapter_path.exists()):
            raise RuntimeError(
                f"Adapter not found for model={model_key}, adapter={adapter_type}, pref={pref_profile}, align={align_context}"
            )

        condition_cfg = {
            "command": "gen",
            "InputRole": "summary",
            "InputField": str(input_col),
            "seed": int(seed),
            "model_key": model_key,
            "model_id": str(model_id),
            "adapter_type": adapter_type,
            "adapter_path": str(adapter_path) if adapter_path else None,
            "pref_profile": pref_profile,
            "align_context": align_context,
            "template": template,
            "retriever_type": retriever_type,
            "retriever_mode": retriever_mode,
            "k": int(k),
            "retrieval_mask": retrieval_mask,
            "decode": decode,
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature) if decode == "sample" else 0.0,
            "top_p": float(top_p) if decode == "sample" else 1.0,
            "precision": precision,
            "prompt_budgeting": {
                "max_prompt_tokens": int(max_prompt_tokens),
                "max_input_tokens": int(max_input_tokens),
                "max_ex_in_tokens": int(max_ex_in_tokens),
                "max_ex_out_tokens": int(max_ex_out_tokens),
                "input_truncation": str(input_truncation),
            },
            "retrieval_cache": cache_path_used,
            "dense_cfg": dataclasses.asdict(dense_cfg),
            "lex_cfg": dataclasses.asdict(lex_cfg),
        }
        condition_id = _config_fingerprint(condition_cfg, length=12)

        runtime_cfg = {
            "macro_batch": int(macro_batch),
            "micro_batch_arg": int(micro_batch),
            "autotune_micro_batch": bool(autotune_mb),
            "tokenizer_threads": int(tokenizer_threads),
            "regex_processes": int(regex_processes),
            "pipeline_prefetch": int(pipeline_prefetch),
        }

        out_path = gen_dir / f"gen_{model_key}_{condition_id}.jsonl.gz"
        tmp_path = tmp_path_for(out_path)
        meta_path = meta_path_for(out_path)
        done_path = done_marker_path(out_path)
        lock_path = lock_path_for(out_path)

        if done_path.exists() and out_path.exists() and not overwrite:
            print(f"[GEN] Skip existing DONE: {out_path.name}")
            outputs.append(out_path)
            continue

        if overwrite:
            clear_artifacts(out_path)
        else:
            if (out_path.exists() or tmp_path.exists() or lock_path.exists()) and not done_path.exists():
                print(
                    f"[WARN] Existing incomplete GEN artifact for {out_path.name}. "
                    "Use --overwrite to regenerate. Skipping."
                )
                continue

        cfg = LLMConfig(
            model_key=model_key,
            model_id=model_id,
            adapter_path=adapter_path,
            load_in_4bit=use_4bit,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        print(
            f"\n[GEN] Loading model={model_key} adapter={adapter_type} pref={pref_profile} align={align_context} "
            f"4bit={use_4bit} dtype={torch_dtype}"
        )
        model, tok = load_llm_and_tokenizer(cfg)

        from transformers import AutoTokenizer

        prep_tok = load_tokenizer_robust(model_id)
        if prep_tok.pad_token is None:
            prep_tok.pad_token = prep_tok.eos_token
        if prep_tok.pad_token_id is None:
            prep_tok.pad_token_id = prep_tok.eos_token_id
        prep_tok.padding_side = "left"
        prep_tok.truncation_side = "left"

        system_prompt = SYSTEM_PROMPT_TEMPLATE_ON if template == "on" else SYSTEM_PROMPT_TEMPLATE_OFF

        try:
            with exclusive_lock(lock_path, stale_hours=float(lock_stale_hours)):
                tok_threads = max(1, int(tokenizer_threads))
                rx_procs = max(0, int(regex_processes))

                thread_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
                if tok_threads >= 2:
                    thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=tok_threads)

                process_pool: Optional[concurrent.futures.ProcessPoolExecutor] = None
                if rx_procs >= 2:
                    import multiprocessing as mp
                    mp_ctx = mp.get_context("spawn")
                    process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=rx_procs, mp_context=mp_ctx)

                def _examples_for_row(row_id: int) -> Tuple[List[Tuple[str, str]], List[int], List[float]]:
                    if retriever_type == "none" or int(k) <= 0 or cache_map is None:
                        return [], [], []
                    r = cache_map.get(int(row_id))
                    if not r:
                        return [], [], []
                    idxs = [int(x) for x in r.get("retrieved_pos", [])][: int(k)]
                    scs_raw = [float(x) for x in r.get("retrieved_scores", [])]
                    scs = scs_raw[: len(idxs)] if scs_raw else []
                    examples: List[Tuple[str, str]] = []
                    for pos in idxs:
                        if 0 <= pos < len(train_inputs):
                            ex_in = train_inputs[pos]
                            ex_out = train_outputs[pos]
                            if retrieval_mask == "hardmask":
                                ex_in = mask_hard_entities(ex_in)
                                ex_out = mask_hard_entities(ex_out)
                            examples.append((ex_in, ex_out))
                    return examples, idxs, scs

                mb = int(micro_batch) if int(micro_batch) > 0 else 8
                if autotune_mb:
                    sample_rows = test_df.iloc[: min(8, len(test_df))]
                    sample_prompts: List[str] = []
                    for rid, inp in zip(
                        sample_rows["row_id"].astype(int).tolist(),
                        sample_rows[input_col].astype(str).tolist(),
                    ):
                        ex, _, _ = _examples_for_row(rid)
                        ptxt, _, _ = build_prompt_budgeted(
                            tokenizer=prep_tok,
                            input_text=inp,
                            system_prompt=system_prompt,
                            examples=ex,
                            max_prompt_tokens=max_prompt_tokens,
                            max_input_tokens=max_input_tokens,
                            max_ex_in_tokens=max_ex_in_tokens,
                            max_ex_out_tokens=max_ex_out_tokens,
                            input_truncation=input_truncation,
                        )
                        sample_prompts.append(ptxt)
                    try:
                        tuned = autotune_micro_batch(
                            model,
                            tok,
                            sample_prompts,
                            max_new_tokens=max_new_tokens,
                            decode=decode,
                            temperature=temperature,
                            top_p=top_p,
                            max_prompt_tokens=max_prompt_tokens,
                            start_mb=mb,
                            max_mb=max(16, mb * 8),
                        )
                        mb = max(1, int(tuned))
                        print(f"[GEN] autotuned micro_batch={mb}")
                    except Exception as e:
                        print(f"[WARN] micro-batch autotune failed ({e}); using micro_batch={mb}")

                import queue
                import threading

                ensure_dir(out_path.parent)
                t0 = time.time()
                total = len(test_df)

                prefetch = max(1, int(pipeline_prefetch))
                prep_q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=prefetch)
                gen_q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=prefetch)
                err_holder: Dict[str, Any] = {"exc": None}

                batch_starts = list(range(0, total, int(macro_batch)))
                stats = {"row_count": 0, "sha1": hashlib.sha1()}

                def _prep_thread_fn() -> None:
                    try:
                        for start_i in batch_starts:
                            if err_holder["exc"] is not None:
                                break
                            batch = test_df.iloc[start_i : start_i + int(macro_batch)]
                            row_ids = batch["row_id"].astype(int).tolist()
                            inputs = batch[input_col].astype(str).tolist()
                            gts = batch[gt_col].astype(str).tolist()

                            def _build_one(args_tuple):
                                rid, inp = args_tuple
                                ex, ex_pos, ex_scores = _examples_for_row(int(rid))
                                ptxt, pinfo, eff_ex = build_prompt_budgeted(
                                    tokenizer=prep_tok,
                                    input_text=inp,
                                    system_prompt=system_prompt,
                                    examples=ex,
                                    max_prompt_tokens=max_prompt_tokens,
                                    max_input_tokens=max_input_tokens,
                                    max_ex_in_tokens=max_ex_in_tokens,
                                    max_ex_out_tokens=max_ex_out_tokens,
                                    input_truncation=input_truncation,
                                )
                                eff_k = int(pinfo.get("EffectiveK", 0))
                                used_pos = ex_pos[:eff_k]
                                used_scores = ex_scores[:eff_k] if ex_scores else []
                                ret_text = "\n".join([a + "\n" + b for a, b in eff_ex])
                                ret_hash = sha1_text(ret_text)
                                return ptxt, pinfo, used_pos, used_scores, ret_text, ret_hash

                            if thread_pool is not None:
                                built = list(thread_pool.map(_build_one, zip(row_ids, inputs)))
                            else:
                                built = [_build_one(x) for x in zip(row_ids, inputs)]

                            pack = {
                                "start": int(start_i),
                                "row_ids": row_ids,
                                "inputs": inputs,
                                "gts": gts,
                                "prompts": [b[0] for b in built],
                                "pinfos": [b[1] for b in built],
                                "used_pos_list": [b[2] for b in built],
                                "used_scores_list": [b[3] for b in built],
                                "ret_text_list": [b[4] for b in built],
                                "ret_hash_list": [b[5] for b in built],
                            }
                            prep_q.put(pack)
                        prep_q.put(None)
                    except Exception as e:
                        err_holder["exc"] = e
                        try:
                            prep_q.put(None)
                        except Exception:
                            pass

                def _eval_thread_fn(f_out) -> None:
                    processed = 0
                    try:
                        while True:
                            pack = gen_q.get()
                            if pack is None:
                                break
                            if err_holder["exc"] is not None:
                                break

                            row_ids = pack["row_ids"]
                            inputs = pack["inputs"]
                            gts = pack["gts"]
                            gens = pack["gens"]
                            gen_tok_counts = pack["gen_tok_counts"]
                            pinfos = pack["pinfos"]
                            used_pos_list = pack["used_pos_list"]
                            used_scores_list = pack["used_scores_list"]
                            ret_hash_list = pack["ret_hash_list"]
                            ret_text_list = pack["ret_text_list"]

                            eval_args = list(zip(inputs, ret_text_list, gens, gen_tok_counts))
                            if process_pool is not None:
                                metrics_list = list(process_pool.map(global_eval_gen_row, eval_args))
                            else:
                                metrics_list = [global_eval_gen_row(x) for x in eval_args]

                            for rid, inp, gt, gen, tok_cnt, pinfo, used_pos, used_scores, ret_hash, (sec, align, hattr) in zip(
                                row_ids,
                                inputs,
                                gts,
                                gens,
                                gen_tok_counts,
                                pinfos,
                                used_pos_list,
                                used_scores_list,
                                ret_hash_list,
                                metrics_list,
                            ):
                                row = {
                                    "ConditionID": condition_id,
                                    "RunTag": run_tag,
                                    "seed": int(seed),
                                    "InputRole": "summary",
                                    "InputField": input_col,
                                    "model": model_key,
                                    "model_id": model_id,
                                    "adapter_type": adapter_type,
                                    "adapter_path": str(adapter_path) if adapter_path else None,
                                    "pref_profile": pref_profile,
                                    "align_context": align_context,
                                    "template": template,
                                    "retriever_type": retriever_type,
                                    "retriever_mode": retriever_mode,
                                    "k": int(k),
                                    "retrieval_mask": retrieval_mask,
                                    "decode": decode,
                                    "precision": precision,
                                    "max_new_tokens": int(max_new_tokens),
                                    "temperature": float(temperature) if decode == "sample" else 0.0,
                                    "top_p": float(top_p) if decode == "sample" else 1.0,
                                    "row_id": int(rid),
                                    "input_text": inp,
                                    "gt_text": gt,
                                    "gen_text": gen,
                                    "gen_tok_count": int(tok_cnt),
                                    "RetrievedPos": used_pos,
                                    "RetrievedScores": used_scores,
                                    "RetrievalTextHash": ret_hash,
                                    **pinfo,
                                    **sec,
                                    **align,
                                    **hattr,
                                }
                                line = json.dumps(row, ensure_ascii=False) + "\n"
                                f_out.write(line)
                                stats["row_count"] += 1
                                stats["sha1"].update(line.encode("utf-8"))

                            processed += len(row_ids)
                            if processed % max(1, int(macro_batch) * 10) < len(row_ids):
                                elapsed = time.time() - t0
                                print(f"[GEN] {model_key} {processed}/{total} elapsed={elapsed:.1f}s")
                    except Exception as e:
                        err_holder["exc"] = e

                try:
                    with gzip.open(tmp_path, "wt", encoding="utf-8") as f_out:
                        t_prep = threading.Thread(target=_prep_thread_fn, daemon=True)
                        t_eval = threading.Thread(target=_eval_thread_fn, args=(f_out,), daemon=True)
                        t_prep.start()
                        t_eval.start()

                        while True:
                            pack = prep_q.get()
                            if pack is None:
                                break
                            if err_holder["exc"] is not None:
                                break

                            prompts = pack["prompts"]
                            gens, gen_tok_counts = generate_microbatched(
                                model,
                                tok,
                                prompts,
                                max_new_tokens=max_new_tokens,
                                decode=decode,
                                temperature=temperature,
                                top_p=top_p,
                                max_prompt_tokens=max_prompt_tokens,
                                micro_batch=mb,
                            )
                            pack["gens"] = gens
                            pack["gen_tok_counts"] = gen_tok_counts
                            gen_q.put(pack)

                        gen_q.put(None)
                        t_prep.join()
                        t_eval.join()

                        if err_holder["exc"] is not None:
                            raise err_holder["exc"]

                    os.replace(tmp_path, out_path)

                finally:
                    if thread_pool is not None:
                        thread_pool.shutdown(wait=True)
                    if process_pool is not None:
                        process_pool.shutdown(wait=True)

                meta_obj = {
                    "command": "gen",
                    "timestamp": now_id(),
                    "run_dir": str(run_dir),
                    "project_root": str(project_root),
                    "condition_id": condition_id,
                    "condition_config": condition_cfg,
                    "runtime_config": runtime_cfg,
                    "adapter_path": str(adapter_path) if adapter_path else None,
                    "output_path": str(out_path),
                    "expected_rows": int(expected_rows),
                    "row_count": int(stats["row_count"]),
                    "content_sha1": stats["sha1"].hexdigest(),
                    "limit": int(limit),
                }
                atomic_write_json(meta_path, meta_obj)
                touch(done_path, now_id())

                print(f"[GEN] Saved: {out_path}")
                outputs.append(out_path)

        except FileExistsError as e:
            print(f"[WARN] GEN lock exists; skipping {out_path.name}: {e}")
        except Exception:
            _remove_if_exists(tmp_path)
            raise
        finally:
            try:
                import torch
                del model
                del tok
                del prep_tok
                torch.cuda.empty_cache()
            except Exception:
                pass

        if export_csv and out_path.exists():
            try:
                base = _strip_known_suffixes(out_path.name)
                csv_path = out_path.with_name(base + ".csv.gz")
                prefer_cols = [
                    "ConditionID",
                    "RunTag",
                    "seed",
                    "InputRole",
                    "InputField",
                    "model",
                    "adapter_type",
                    "pref_profile",
                    "align_context",
                    "template",
                    "retriever_type",
                    "retriever_mode",
                    "k",
                    "retrieval_mask",
                    "decode",
                    "precision",
                    "row_id",
                    "input_text",
                    "gt_text",
                    "gen_text",
                ]
                jsonl_gz_to_csv_gz(out_path, csv_path, prefer_cols=prefer_cols)
                print(f"[GEN] CSV saved: {csv_path}")
                outputs.append(csv_path)
            except Exception as e:
                print(f"[WARN] CSV export failed: {e}")

    return outputs


def global_eval_gen_row(args: Tuple[str, str, str, int]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    inp, ret, gen, tok_cnt = args
    sec = section_metrics(gen)
    align = section_alignment_metrics2(inp, gen)
    hattr = hard_attribution_metrics(inp, ret, gen, resp_tokens=int(tok_cnt))
    return sec, align, hattr


# ==============================
# SCORE stage: ROUGE / CTQRS / SBERT
# ==============================

def rouge_metrics(gen_text: str, ref_text: str) -> Dict[str, float]:
    try:
        from rouge_score import rouge_scorer  # type: ignore
    except Exception:
        return {"ROUGE1_R": float("nan"), "ROUGE1_F1": float("nan")}
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    scores = scorer.score(ref_text, gen_text)["rouge1"]
    return {"ROUGE1_R": float(scores.recall), "ROUGE1_F1": float(scores.fmeasure)}


class SBERTScorer:
    def __init__(self, model_name: str, device: Literal["cpu", "cuda", "auto"] = "cuda") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _resolve_device(self) -> str:
        if self.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return self.device

    def load(self) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        self._model = SentenceTransformer(self.model_name, device=self._resolve_device())

    def score_pairs(self, a_list: List[str], b_list: List[str], batch_size: int = 512) -> List[float]:
        n = min(len(a_list), len(b_list))
        if n <= 0:
            return []
        if self._model is None:
            self.load()
        assert self._model is not None
        emb_a = self._model.encode(
            a_list[:n],
            batch_size=int(batch_size),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        emb_b = self._model.encode(
            b_list[:n],
            batch_size=int(batch_size),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sims = np.sum(emb_a * emb_b, axis=1)
        return [float(x) for x in sims.tolist()]


def load_ctqrs_fn(ctqrs_path: Optional[str]) -> Optional[Any]:
    if not ctqrs_path or ctqrs_path == "SKIP":
        return None
    p = Path(ctqrs_path)
    if not p.exists():
        for cand in [Path("evaluation/perfect_ctqrs.py"), Path("perfect_ctqrs.py")]:
            if cand.exists():
                p = cand
                break
    if not p.exists():
        return None
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("ctqrs_module", str(p))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        if hasattr(mod, "evaluate_bug_report"):
            return getattr(mod, "evaluate_bug_report")
    except Exception:
        return None
    return None


def ctqrs_score(evaluate_bug_report_fn: Optional[Any], text: str) -> float:
    if evaluate_bug_report_fn is None:
        return float("nan")
    try:
        r = evaluate_bug_report_fn(text)
        total = r.get("total_score", None)
        maxp = r.get("max_possible", None)
        if total is None or maxp in (None, 0):
            return float("nan")
        return float(total) / float(maxp)
    except Exception:
        return float("nan")


def score_run(
    *,
    gen_paths: List[Path],
    out_dir: Path,
    run_tag: str,
    metrics: List[str],
    do_normalize: bool,
    ctqrs_path: Optional[str],
    rouge_processes: int,
    ctqrs_processes: int,
    sbert_model: str,
    sbert_device: Literal["cpu", "cuda", "auto"],
    sbert_batch_size: int,
    export_csv: bool,
    overwrite: bool = False,
    lock_stale_hours: float = 24.0,
    only_done_gen: bool = True,
) -> List[Path]:
    run_tag = _sanitize_run_tag(run_tag)
    ensure_dir(out_dir)

    do_rouge = "rouge" in metrics
    do_ctqrs = "ctqrs" in metrics
    do_sbert = "sbert" in metrics

    _ = load_ctqrs_fn(ctqrs_path) if do_ctqrs else None
    sbert = SBERTScorer(sbert_model, device=sbert_device) if do_sbert else None

    out_paths: List[Path] = []

    for gen_path in gen_paths:
        gen_path = Path(gen_path)
        if only_done_gen and not done_marker_path(gen_path).exists():
            print(f"[SCORE] Skip (GEN not done): {gen_path.name}")
            continue

        gen_meta = load_meta(meta_path_for(gen_path)) or {}
        condition_id = gen_meta.get("condition_id", None)
        condition_cfg = gen_meta.get("condition_config", {}) if isinstance(gen_meta.get("condition_config", {}), dict) else {}
        model_key = condition_cfg.get("model_key", None)

        if not condition_id:
            m = re.search(r"gen_[^_]+_([0-9a-f]{8,})", gen_path.name)
            condition_id = m.group(1) if m else sha1_text(gen_path.name)
        if not model_key:
            try:
                it = jsonl_gz_iter(gen_path)
                first = next(iter(it))
                model_key = first.get("model", "unknown")
            except Exception:
                model_key = "unknown"

        out_path = Path(out_dir) / f"scored_{model_key}_{condition_id}.jsonl.gz"
        tmp_path = tmp_path_for(out_path)
        meta_path = meta_path_for(out_path)
        done_path = done_marker_path(out_path)
        lock_path = lock_path_for(out_path)

        if done_path.exists() and out_path.exists() and not overwrite:
            print(f"[SCORE] Skip existing DONE: {out_path.name}")
            out_paths.append(out_path)
            continue

        if overwrite:
            clear_artifacts(out_path)
        else:
            if (out_path.exists() or tmp_path.exists() or lock_path.exists()) and not done_path.exists():
                print(
                    f"[WARN] Existing incomplete SCORE artifact for {out_path.name}. "
                    "Use --overwrite to regenerate. Skipping."
                )
                continue

        try:
            with exclusive_lock(lock_path, stale_hours=float(lock_stale_hours)):
                rows = list(jsonl_gz_iter(gen_path))
                if not rows:
                    print(f"[SCORE] empty gen file: {gen_path}")
                    continue

                gens = [r.get("gen_text", "") for r in rows]
                gts = [r.get("gt_text", "") for r in rows]

                gens_n = [normalize_for_similarity(x) for x in gens] if do_normalize else gens
                gts_n = [normalize_for_similarity(x) for x in gts] if do_normalize else gts

                rouge_list: List[Dict[str, float]] = [
                    {"ROUGE1_R": float("nan"), "ROUGE1_F1": float("nan")} for _ in rows
                ]
                rouge_n_list: List[Dict[str, float]] = [
                    {"ROUGE1_R_NORM": float("nan"), "ROUGE1_F1_NORM": float("nan")} for _ in rows
                ]
                ctqrs_list: List[float] = [float("nan")] * len(rows)
                sbert_list: List[float] = [float("nan")] * len(rows)
                sbert_n_list: List[float] = [float("nan")] * len(rows)

                import multiprocessing as mp
                mp_ctx = mp.get_context("spawn")
                if do_rouge:
                    rp = max(1, int(rouge_processes))
                    with concurrent.futures.ProcessPoolExecutor(max_workers=rp, mp_context=mp_ctx) as pool:
                        rouge_list = list(pool.map(_rouge_pair, zip(gens, gts)))
                        if do_normalize:
                            rouge_n = list(pool.map(_rouge_pair, zip(gens_n, gts_n)))
                            rouge_n_list = [
                                {"ROUGE1_R_NORM": d["ROUGE1_R"], "ROUGE1_F1_NORM": d["ROUGE1_F1"]} for d in rouge_n
                            ]

                if do_ctqrs:
                    cp = max(1, int(ctqrs_processes))
                    with concurrent.futures.ProcessPoolExecutor(max_workers=cp, mp_context=mp_ctx) as pool:
                        ctqrs_list = list(pool.map(_ctqrs_text, [(ctqrs_path, x) for x in gens]))

                if do_sbert and sbert is not None:
                    sbert_list = sbert.score_pairs(gens, gts, batch_size=int(sbert_batch_size))
                    if do_normalize:
                        sbert_n_list = sbert.score_pairs(gens_n, gts_n, batch_size=int(sbert_batch_size))

                stats = {"row_count": 0, "sha1": hashlib.sha1()}
                ensure_dir(out_path.parent)
                with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
                    for r, rg, rgn, ctq, sb, sbn in zip(
                        rows, rouge_list, rouge_n_list, ctqrs_list, sbert_list, sbert_n_list
                    ):
                        r2 = dict(r)
                        r2.update(
                            {
                                "CTQRS": float(ctq),
                                "SBERT": float(sb),
                                "SBERT_NORM": float(sbn) if do_normalize else float("nan"),
                                **rg,
                                **rgn,
                            }
                        )
                        line = json.dumps(r2, ensure_ascii=False) + "\n"
                        f.write(line)
                        stats["row_count"] += 1
                        stats["sha1"].update(line.encode("utf-8"))

                os.replace(tmp_path, out_path)

                save_obj = {
                    "command": "score",
                    "timestamp": now_id(),
                    "run_tag": run_tag,
                    "source_gen": str(gen_path),
                    "source_gen_meta": gen_meta,
                    "metrics": metrics,
                    "do_normalize": bool(do_normalize),
                    "ctqrs_path": ctqrs_path,
                    "sbert_model": sbert_model,
                    "sbert_device": sbert_device,
                    "sbert_batch_size": int(sbert_batch_size),
                    "output_path": str(out_path),
                    "row_count": int(stats["row_count"]),
                    "content_sha1": stats["sha1"].hexdigest(),
                }
                atomic_write_json(meta_path, save_obj)
                touch(done_path, now_id())

                print(f"[SCORE] Saved: {out_path}")
                out_paths.append(out_path)

                if export_csv:
                    try:
                        csv_path = out_path.with_suffix(".csv.gz")
                        jsonl_gz_to_csv_gz(out_path, csv_path)
                        print(f"[SCORE] CSV saved: {csv_path}")
                        out_paths.append(csv_path)
                    except Exception as e:
                        print(f"[WARN] SCORE CSV export failed: {e}")

        except FileExistsError as e:
            print(f"[WARN] SCORE lock exists; skipping {out_path.name}: {e}")
            continue
        except Exception:
            _remove_if_exists(tmp_path)
            raise

    return out_paths


def _rouge_pair(args: Tuple[str, str]) -> Dict[str, float]:
    a, b = args
    return rouge_metrics(a, b)

_GLOBAL_CTQRS_FN = None

def _ctqrs_text(args: Tuple[Optional[str], str]) -> float:
    global _GLOBAL_CTQRS_FN
    path, text = args
    if _GLOBAL_CTQRS_FN is None:
        _GLOBAL_CTQRS_FN = load_ctqrs_fn(path)
    return ctqrs_score(_GLOBAL_CTQRS_FN, text)


# ==============================
# Aggregate: groupby + bootstrap CI
# ==============================


def bootstrap_ci(
    values: List[float],
    iters: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    vals = [v for v in values if v == v]  # drop NaN
    if not vals:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(int(seed))
    n = len(vals)
    means = []
    for _ in range(int(iters)):
        samp = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(samp) / n)
    means.sort()
    mean = sum(vals) / n
    alpha = (1.0 - float(ci)) / 2.0
    lo = means[int(alpha * len(means))]
    hi = means[int((1.0 - alpha) * len(means)) - 1]
    return float(mean), float(lo), float(hi)


def aggregate_run(
    *,
    scored_glob: str,
    out_csv: Path,
    groupby: List[str],
    stat: Literal["bootstrap_ci", "mean_ci", "median_iqr"],
    bootstrap_iters: int,
    ci: float,
    seed: int,
) -> None:
    import glob
    paths = [Path(p) for p in glob.glob(scored_glob)]
    if not paths:
        raise FileNotFoundError(f"No files matched: {scored_glob}")

    rows: List[Dict[str, Any]] = []
    for p in paths:
        for r in jsonl_gz_iter(p):
            rows.append(r)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No rows loaded for aggregation.")

    for g in groupby:
        if g not in df.columns:
            df[g] = "NA"

    metric_cols = [
        "CTQRS",
        "ROUGE1_R",
        "ROUGE1_F1",
        "SBERT",
        "ROUGE1_R_NORM",
        "ROUGE1_F1_NORM",
        "SBERT_NORM",
        "SecFilled",
        "SecPresence",
        "PlaceholderRate",
        "UnknownRate",
        "UAHE_Total",
        "IUHE_Total",
        "RAHE_Total",
        "UAHE_per_1kTok",
        "IUHE_per_1kTok",
        "RAHE_per_1kTok",
        "TransferRate",
        "ContextSupportRate",
        "ContextUnattributedRate",
        "CondFilledRate",
        "SignalFilledRate",
        "NoSignalFilledRate",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    out_rows = []
    grouped = df.groupby(groupby, dropna=False)
    for key, sub in grouped:
        rec: Dict[str, Any] = {}
        if isinstance(key, tuple):
            for g, v in zip(groupby, key):
                rec[g] = v
        else:
            rec[groupby[0]] = key

        for m in metric_cols:
            vals = [float(x) for x in sub[m].tolist()]
            if stat == "bootstrap_ci":
                mean, lo, hi = bootstrap_ci(vals, iters=bootstrap_iters, ci=ci, seed=seed)
                rec[f"{m}_mean"] = mean
                rec[f"{m}_ci_lo"] = lo
                rec[f"{m}_ci_hi"] = hi
            elif stat == "median_iqr":
                vv = [v for v in vals if v == v]
                if vv:
                    rec[f"{m}_median"] = float(np.median(vv))
                    rec[f"{m}_iqr"] = float(np.percentile(vv, 75) - np.percentile(vv, 25))
                else:
                    rec[f"{m}_median"] = float("nan")
                    rec[f"{m}_iqr"] = float("nan")
            else:
                vv = [v for v in vals if v == v]
                rec[f"{m}_mean"] = float(np.mean(vv)) if vv else float("nan")
                rec[f"{m}_std"] = float(np.std(vv)) if vv else float("nan")

        rec["n"] = int(len(sub))
        out_rows.append(rec)

    out_df = pd.DataFrame(out_rows)
    ensure_dir(out_csv.parent)
    out_df.to_csv(out_csv, index=False)
    print(f"[AGG] Saved: {out_csv}")


# ==============================
# Training (optional): SFT + DPO
# ==============================

def load_latest_adapter(adapter_root: Path) -> Optional[Path]:
    latest_txt = adapter_root / "LATEST_ADAPTER.txt"
    if latest_txt.exists():
        try:
            p = Path(latest_txt.read_text(encoding="utf-8").strip())
            if p.exists():
                return p
        except Exception:
            pass

    runs = sorted(
        [d for d in adapter_root.glob("RUN_*") if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return runs[0] if runs else None


class SFTDataset:
    def __init__(self, df: pd.DataFrame, tokenizer, input_col: str, gt_col: str, system_prompt: str, max_len: int):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.input_col = input_col
        self.gt_col = gt_col
        self.system_prompt = system_prompt
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.df.iloc[idx]
        inp = str(r[self.input_col]).strip()
        tgt = str(r[self.gt_col]).strip()

        prompt_text = build_prompt(inp, self.system_prompt, examples=[])
        prompt_ids = self.tok(prompt_text, add_special_tokens=False).input_ids
        target_ids = self.tok(tgt, add_special_tokens=False).input_ids

        eos = self.tok.eos_token_id
        input_ids = prompt_ids + target_ids + ([eos] if eos is not None else [])
        labels = [-100] * len(prompt_ids) + target_ids + ([eos] if eos is not None else [])

        if len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len]
            labels = labels[: self.max_len]

        attn = [1] * len(input_ids)
        return {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "labels": np.array(labels, dtype=np.int64),
            "attention_mask": np.array(attn, dtype=np.int64),
        }


def sft_collate(tokenizer):
    import torch
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.eos_token_id

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, labels, attention_mask = [], [], []
        for x in batch:
            L = len(x["input_ids"])
            pad_len = max_len - L
            input_ids.append(np.pad(x["input_ids"], (0, pad_len), constant_values=pad_id))
            attention_mask.append(np.pad(x["attention_mask"], (0, pad_len), constant_values=0))
            labels.append(np.pad(x["labels"], (0, pad_len), constant_values=-100))
        return {
            "input_ids": torch.tensor(np.stack(input_ids), dtype=torch.long),
            "attention_mask": torch.tensor(np.stack(attention_mask), dtype=torch.long),
            "labels": torch.tensor(np.stack(labels), dtype=torch.long),
        }
    return collate


def cmd_train_sft(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    set_seed(int(args.seed))

    splits = load_cached_splits(project_root, seed=int(args.seed))
    if splits is None:
        raise RuntimeError("Cached splits not found. Run data-prepare first.")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
    except Exception as e:
        raise RuntimeError(
            "SFT training requires extra deps: torch, transformers, peft, bitsandbytes. "
            f"Import failed: {e}"
        )

    system_prompt = SYSTEM_PROMPT_TEMPLATE_ON if args.template == "on" else SYSTEM_PROMPT_TEMPLATE_OFF
    run_tag = _sanitize_run_tag(getattr(args, "run_tag", None))

    adapter_root = ensure_dir(project_root / ADAPTER_DIR_REL / "sft")
    for model_key in args.models:
        if model_key not in MODEL_ID:
            raise KeyError(f"Unknown model key: {model_key}. Available: {list(MODEL_ID.keys())}")
        model_id = MODEL_ID[model_key]

        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        quant_config = None
        if args.lora_type == "qlora":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )

        load_kwargs = {"device_map": "auto", "trust_remote_code": True}
        if args.lora_type == "qlora" and quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
        else:
            load_kwargs["torch_dtype"] = compute_dtype

        base_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        tok = load_tokenizer_robust(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "left"
        tok.truncation_side = "left"

        if args.lora_type == "qlora":
            base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

        peft_cfg = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(base_model, peft_cfg)

        train_ds = SFTDataset(
            splits.train,
            tokenizer=tok,
            input_col=args.input_col,
            gt_col=args.gt_col,
            system_prompt=system_prompt,
            max_len=int(args.max_seq_len),
        )
        eval_ds = SFTDataset(
            splits.val,
            tokenizer=tok,
            input_col=args.input_col,
            gt_col=args.gt_col,
            system_prompt=system_prompt,
            max_len=int(args.max_seq_len),
        )
        collator = sft_collate(tok)

        cfg_for_hash = {
            "command": "train_sft",
            "model_key": model_key,
            "seed": int(args.seed),
            "input_col": args.input_col,
            "gt_col": args.gt_col,
            "template": args.template,
            "lora_type": args.lora_type,
            "lora_r": int(args.lora_r),
            "lora_alpha": int(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "grad_accum_steps": int(args.grad_accum_steps),
            "max_seq_len": int(args.max_seq_len),
            "save_steps": int(args.save_steps),
            "logging_steps": int(args.logging_steps),
        }
        cfg_hash = _config_fingerprint(cfg_for_hash)
        run_id = now_id()
        run_dir = ensure_dir(adapter_root / model_key / _join_nonempty(["RUN_SFT", run_id, run_tag, cfg_hash]))

        train_args_kwargs = dict(
            output_dir=str(run_dir),
            per_device_train_batch_size=int(args.batch_size),
            per_device_eval_batch_size=int(args.batch_size),
            gradient_accumulation_steps=int(args.grad_accum_steps),
            num_train_epochs=float(args.epochs),
            learning_rate=float(args.lr),
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=int(args.logging_steps),
            save_strategy="steps",
            save_steps=int(args.save_steps),
            evaluation_strategy="steps",
            eval_steps=int(args.save_steps),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
            report_to="none",
            bf16=(args.lora_type == "lora" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
            fp16=(args.lora_type == "lora" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
            gradient_checkpointing=(args.lora_type == "qlora"),
            remove_unused_columns=False,
        )
        try:
            train_args = TrainingArguments(**train_args_kwargs)
        except TypeError:
            if "evaluation_strategy" in train_args_kwargs:
                train_args_kwargs["eval_strategy"] = train_args_kwargs.pop("evaluation_strategy")
                train_args = TrainingArguments(**train_args_kwargs)
            else:
                raise

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
        )
        trainer.train()

        model.save_pretrained(str(run_dir))
        tok.save_pretrained(str(run_dir))

        atomic_write_json(
            run_dir / "train_config.json",
            {
                "command": "train_sft",
                "timestamp": run_id,
                "run_tag": run_tag,
                "config_hash": cfg_hash,
                "config_for_hash": cfg_for_hash,
                "model_key": model_key,
                "model_id": model_id,
                "seed": int(args.seed),
                "input_col": args.input_col,
                "gt_col": args.gt_col,
                "template": args.template,
                "lora_type": args.lora_type,
            },
        )

        latest_txt = ensure_dir(adapter_root / model_key) / "LATEST_ADAPTER.txt"
        latest_txt.write_text(str(run_dir), encoding="utf-8")
        print(f"[SFT] Saved adapter: {run_dir}")


def cmd_mine_dpo_pairs(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    set_seed(int(args.seed))

    splits = load_cached_splits(project_root, seed=int(args.seed))
    if splits is None:
        raise RuntimeError("Cached splits not found. Run data-prepare first.")

    train_df = splits.train.reset_index(drop=True)
    if int(getattr(args, "limit", 0) or 0) > 0:
        train_df = train_df.iloc[: int(args.limit)].copy()

    dense_cfg = DenseIndexConfig(emb_model=str(args.dense_emb_model), normalize=True)
    lex_cfg = LexicalIndexConfig(
        backend=str(args.lex_backend),
        max_features=int(args.tfidf_max_features),
        ngram=int(args.tfidf_ngram),
        bm25_k1=float(args.bm25_k1),
        bm25_b=float(args.bm25_b),
    )

    if args.input_col not in train_df.columns or args.gt_col not in train_df.columns:
        raise KeyError(f"Missing columns in train split: input_col={args.input_col}, gt_col={args.gt_col}")
    train_inputs = train_df[args.input_col].astype(str).tolist()
    train_outputs = train_df[args.gt_col].astype(str).tolist()

    cache_map: Optional[Dict[int, Dict[str, Any]]] = None
    cache_path: Optional[Path] = None
    if str(args.align_context) == "retrieval_aware" and int(args.k) > 0:
        cache_path = build_retrieval_cache(
            project_root=project_root,
            seed=int(args.seed),
            input_col=args.input_col,
            split="train",
            retriever_type=args.retriever_type,
            retriever_mode=args.retriever_mode,
            k=int(args.k),
            dense_cfg=dense_cfg,
            lex_cfg=lex_cfg,
            dense_device=args.dense_device,
            dense_batch_size=int(args.dense_batch_size),
            leave_one_out=True,
            exclude_same_group=True,  # EMSE PATCH
            group_col="group_id",
            force=bool(args.force_cache),
        )
        cache_map = load_retrieval_cache(cache_path)
        print(f"[DPO-MINE] Using train retrieval cache: {cache_path}")

    system_prompt = SYSTEM_PROMPT_TEMPLATE_ON
    run_tag = _sanitize_run_tag(getattr(args, "run_tag", None))

    out_dir = ensure_dir(project_root / DPO_DIR_REL / f"seed{int(args.seed)}" / str(args.pref_profile))
    cand_dir = ensure_dir(project_root / CACHE_DIR_REL / "dpo_candidates")

    def _raw_examples_for_row(row_id: int) -> Tuple[List[Tuple[str, str]], List[int]]:
        if cache_map is None or int(args.k) <= 0:
            return [], []
        r = cache_map.get(int(row_id))
        if not r:
            return [], []
        idxs = [int(x) for x in r.get("retrieved_pos", [])][: int(args.k)]
        ex: List[Tuple[str, str]] = []
        used_pos: List[int] = []
        for pos in idxs:
            if 0 <= pos < len(train_inputs):
                ex_in, ex_out = train_inputs[pos], train_outputs[pos]
                if args.retrieval_mask == "hardmask":
                    ex_in, ex_out = mask_hard_entities(ex_in), mask_hard_entities(ex_out)
                ex.append((ex_in, ex_out))
                used_pos.append(int(pos))
        return ex, used_pos

    for model_key in args.models:
        if model_key not in MODEL_ID:
            raise KeyError(f"Unknown model key: {model_key}. Available: {list(MODEL_ID.keys())}")
        model_id = MODEL_ID[model_key]

        nc = max(1, int(args.num_candidates))
        input_bs = max(1, int(args.input_batch_size))
        gen_mode = str(getattr(args, "candidate_gen_mode", "repeat"))

        adapter_path: Optional[Path] = None
        if str(args.generator_adapter) == "sft":
            sft_root = project_root / ADAPTER_DIR_REL / "sft" / model_key
            adapter_path = Path(args.sft_adapter).resolve() if args.sft_adapter else load_latest_adapter(sft_root)
            if adapter_path is None or not adapter_path.exists():
                raise RuntimeError(
                    f"SFT adapter not found for {model_key}. Train SFT first or pass --sft-adapter."
                )

        if int(args.micro_batch) > 0:
            mb = int(args.micro_batch)
        else:
            mb = max(nc, nc * 4) if gen_mode == "num_return_sequences" else 4

        use_4bit, torch_dtype, device_map = _resolve_precision(args.precision)
        cfg = LLMConfig(
            model_key=model_key,
            model_id=model_id,
            adapter_path=adapter_path,
            load_in_4bit=use_4bit,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        print(f"\n[DPO-MINE] Loading generator model={model_key} adapter={args.generator_adapter} 4bit={use_4bit} dtype={torch_dtype}")
        model, tok = load_llm_and_tokenizer(cfg)

        prep_tok = load_tokenizer_robust(model_id)
        if prep_tok.pad_token is None:
            prep_tok.pad_token = prep_tok.eos_token
        if prep_tok.pad_token_id is None:
            prep_tok.pad_token_id = prep_tok.eos_token_id
        prep_tok.padding_side = "left"
        prep_tok.truncation_side = "left"

        cfg_for_hash = {
            "command": "mine_dpo_pairs",
            "model_key": model_key,
            "seed": int(args.seed),
            "pref_profile": str(args.pref_profile),
            "align_context": str(args.align_context),
            "pref_score_version": str(getattr(args, "pref_score_version", "v2_signalaware")),
            "pairs_per_prompt": int(getattr(args, "pairs_per_prompt", 1)),
            "uahe_cap": float(getattr(args, "uahe_cap", -1.0)),
            "min_cond_filled_rate": float(getattr(args, "min_cond_filled_rate", 0.0)),
            "enforce_uahe_order": bool(getattr(args, "enforce_uahe_order", False)),
            "save_candidates_jsonl_gz": bool(getattr(args, "save_candidates_jsonl_gz", False)),
            "k": int(args.k),
            "retriever_type": str(getattr(args, "retriever_type", "dense")),
            "retriever_mode": str(getattr(args, "retriever_mode", "similar")),
            "retrieval_mask": str(getattr(args, "retrieval_mask", "none")),
            "num_candidates": int(args.num_candidates),
            "candidate_gen_mode": gen_mode,
            "gen_max_new_tokens": int(args.gen_max_new_tokens),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "quality_floor": float(args.quality_floor),
            "margin": float(args.margin),
            "dense_cfg": dataclasses.asdict(dense_cfg),
            "lex_cfg": dataclasses.asdict(lex_cfg),
            "generator_adapter": str(args.generator_adapter),
            "generator_adapter_path": str(adapter_path) if adapter_path else None,
            "prompt_budgeting": {
                "max_prompt_tokens": int(args.max_prompt_tokens),
                "max_input_tokens": int(args.max_input_tokens),
                "max_ex_in_tokens": int(args.max_ex_in_tokens),
                "max_ex_out_tokens": int(args.max_ex_out_tokens),
                "input_truncation": str(args.input_truncation),
            },
            "limit": int(getattr(args, "limit", 0) or 0),
        }
        cfg_hash = _config_fingerprint(cfg_for_hash)
        run_id = now_id()

        base = _join_nonempty(
            [
                "dpo_pairs",
                model_key,
                f"seed{int(args.seed)}",
                str(args.pref_profile),
                str(args.align_context),
                run_tag,
                cfg_hash,
                run_id,
            ]
        )
        out_jsonl = out_dir / (base + ".jsonl")
        out_csv = out_dir / (base + ".csv")
        cand_csv = cand_dir / (base + "_candidates.csv")
        cand_full_path = cand_dir / (base + "_candidates_full.jsonl.gz")

        cand_full_f = None
        if bool(getattr(args, "save_candidates_jsonl_gz", False)):
            ensure_dir(cand_full_path.parent)
            cand_full_f = gzip.open(cand_full_path, "wt", encoding="utf-8")

        try:
            if bool(getattr(args, "autotune_micro_batch", False)):
                sample_rows = train_df.iloc[: min(8, len(train_df))]
                sample_prompts: List[str] = []
                for rid, inp in zip(
                    sample_rows["row_id"].astype(int).tolist() if "row_id" in sample_rows.columns else list(range(len(sample_rows))),
                    sample_rows[args.input_col].astype(str).tolist(),
                ):
                    ex0, _pos0 = _raw_examples_for_row(int(rid)) if str(args.align_context) == "retrieval_aware" else ([], [])
                    ptxt, _pinfo, _eff = build_prompt_budgeted(
                        tokenizer=prep_tok,
                        input_text=inp,
                        system_prompt=system_prompt,
                        examples=ex0,
                        max_prompt_tokens=int(args.max_prompt_tokens),
                        max_input_tokens=int(args.max_input_tokens),
                        max_ex_in_tokens=int(args.max_ex_in_tokens),
                        max_ex_out_tokens=int(args.max_ex_out_tokens),
                        input_truncation=args.input_truncation,
                    )
                    sample_prompts.append(ptxt)

                try:
                    if gen_mode == "num_return_sequences":
                        tuned = autotune_seq_micro_batch_multi(
                            model,
                            tok,
                            sample_prompts,
                            max_new_tokens=int(args.gen_max_new_tokens),
                            num_return_sequences=nc,
                            temperature=float(args.temperature),
                            top_p=float(args.top_p),
                            max_prompt_tokens=int(args.max_prompt_tokens),
                            start_seq_mb=mb,
                            max_seq_mb=max(64, mb * 8),
                        )
                        mb = max(nc, int(tuned))
                    else:
                        tuned = autotune_micro_batch(
                            model,
                            tok,
                            sample_prompts,
                            max_new_tokens=int(args.gen_max_new_tokens),
                            decode="sample",
                            temperature=float(args.temperature),
                            top_p=float(args.top_p),
                            max_prompt_tokens=int(args.max_prompt_tokens),
                            start_mb=mb,
                            max_mb=max(16, mb * 8),
                        )
                        mb = max(1, int(tuned))
                    print(f"[DPO-MINE] autotuned micro_batch={mb} (mode={gen_mode}, nc={nc})")
                except Exception as e:
                    print(f"[WARN] DPO micro-batch autotune failed ({e}); using micro_batch={mb}")

            pair_rows: List[Dict[str, Any]] = []
            cand_rows: List[Dict[str, Any]] = []
            t0 = time.time()

            for start in range(0, len(train_df), input_bs):
                batch = train_df.iloc[start : start + input_bs]
                row_ids = batch["row_id"].astype(int).tolist() if "row_id" in batch.columns else list(range(start, start + len(batch)))
                inputs = batch[args.input_col].astype(str).tolist()

                prompts: List[str] = []
                ret_texts: List[str] = []
                ret_poss: List[List[int]] = []

                for rid, inp in zip(row_ids, inputs):
                    ex0, pos0 = _raw_examples_for_row(int(rid)) if str(args.align_context) == "retrieval_aware" else ([], [])
                    ptxt, pinfo, eff_ex = build_prompt_budgeted(
                        tokenizer=prep_tok,
                        input_text=inp,
                        system_prompt=system_prompt,
                        examples=ex0,
                        max_prompt_tokens=int(args.max_prompt_tokens),
                        max_input_tokens=int(args.max_input_tokens),
                        max_ex_in_tokens=int(args.max_ex_in_tokens),
                        max_ex_out_tokens=int(args.max_ex_out_tokens),
                        input_truncation=args.input_truncation,
                    )
                    eff_k = int(pinfo.get("EffectiveK", len(eff_ex)))
                    pos_eff = pos0[:eff_k]
                    ret_text = "\n".join([a + "\n" + b for a, b in eff_ex])

                    if gen_mode == "num_return_sequences":
                        prompts.append(ptxt)
                        ret_texts.append(ret_text)
                        ret_poss.append(pos_eff)
                    else:
                        prompts.extend([ptxt] * nc)
                        ret_texts.extend([ret_text] * nc)
                        ret_poss.extend([pos_eff] * nc)

                if gen_mode == "num_return_sequences":
                    gens, gen_tok_counts = generate_microbatched_multi(
                        model,
                        tok,
                        prompts,
                        max_new_tokens=int(args.gen_max_new_tokens),
                        num_return_sequences=nc,
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        max_prompt_tokens=int(args.max_prompt_tokens),
                        micro_batch_seqs=mb,
                    )
                    
                    # EMSE Patch: Assert output length to prevent silent OOM dropping bugs
                    expected = len(prompts) * nc
                    if len(gens) != expected or len(gen_tok_counts) != expected:
                        raise RuntimeError(
                            f"[DPO-MINE] Candidate length mismatch: expected {expected} "
                            f"(prompts={len(prompts)} * nc={nc}), got {len(gens)}"
                        )
                else:
                    gens, gen_tok_counts = generate_microbatched(
                        model,
                        tok,
                        prompts,
                        max_new_tokens=int(args.gen_max_new_tokens),
                        decode="sample",
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        max_prompt_tokens=int(args.max_prompt_tokens),
                        micro_batch=mb,
                    )

                # =========================================================
                # 🚀 [EMSE SPEED PATCH] CPU 24코어 멀티프로세싱 채점 🚀
                # =========================================================
                import multiprocessing as mp
                from concurrent.futures import ProcessPoolExecutor
                mp_ctx = mp.get_context("spawn")
                
                eval_args_list = []
                
                # 1. 점수 매길 텍스트들을 리스트로 모으기 (단일 루프)
                for j, (rid, inp) in enumerate(zip(row_ids, inputs)):
                    if gen_mode == "num_return_sequences":
                        base_idx = j * nc
                        cands = gens[base_idx : base_idx + nc]
                        cts = gen_tok_counts[base_idx : base_idx + nc]
                        rtxt = ret_texts[j] if ret_texts else ""
                    else:
                        cands = gens[j * nc : (j + 1) * nc]
                        cts = gen_tok_counts[j * nc : (j + 1) * nc]
                        rtxt = ret_texts[j * nc] if ret_texts else ""
                        
                    for orig_idx, (g, tcnt) in enumerate(zip(cands, cts)):
                        eval_args_list.append((inp, rtxt, g, int(tcnt), args.pref_profile, getattr(args, "pref_score_version", "v2_signalaware")))

                # 2. 32코어 중 24개의 워커를 생성하여 병렬 채점 실행
                regex_workers = int(getattr(args, "regex_processes", 24))
                if regex_workers <= 0: regex_workers = 24
                
                with ProcessPoolExecutor(max_workers=regex_workers, mp_context=mp_ctx) as pool:
                    results = list(pool.map(_dpo_score_worker, eval_args_list))

                # 3. 채점된 결과를 다시 row_id 기준으로 묶어서 평가
                result_idx = 0
                for j, (rid, inp) in enumerate(zip(row_ids, inputs)):
                    if gen_mode == "num_return_sequences":
                        rpos = ret_poss[j] if ret_poss else []
                        rtxt = ret_texts[j] if ret_texts else ""
                    else:
                        rpos = ret_poss[j * nc] if ret_poss else []
                        rtxt = ret_texts[j * nc] if ret_texts else ""
                        
                    scored: List[Tuple[float, Dict[str, float], str, int, int]] = []
                    for orig_idx in range(nc):
                        sc, comps, g, tcnt = results[result_idx]
                        scored.append((float(sc), comps, g, int(tcnt), int(orig_idx)))
                        result_idx += 1

                    # 여기서부터는 쌍(Pair)을 고르는 기존 로직 유지
                    scored.sort(key=lambda x: x[0], reverse=True)
                    best_sc, best_comps, best_text, best_tok, _best_idx = scored[0]
                    worst_sc, worst_comps, worst_text, worst_tok, _worst_idx = scored[-1]

                    cand_rows.append(
                        {
                            "row_id": int(rid),
                            "model": model_key,
                            "pref_profile": args.pref_profile,
                            "align_context": args.align_context,
                            "best_score": float(best_sc),
                            "worst_score": float(worst_sc),
                            "UAHE_per_1kTok_best": float(best_comps.get("UAHE_per_1kTok", float("nan"))),
                            "UAHE_per_1kTok_worst": float(worst_comps.get("UAHE_per_1kTok", float("nan"))),
                            "CondFilledRate_best": float(best_comps.get("CondFilledRate", float("nan"))),
                            "ContextUnattributedRate_best": float(best_comps.get("ContextUnattributedRate", float("nan"))),
                            "k": int(args.k),
                            "retrieved_pos": json.dumps(rpos, ensure_ascii=False),
                        }
                    )

                    if cand_full_f is not None:
                        ret_hash = sha1_text(rtxt)
                        for rank, (sc, comps, txt, tok_cnt, orig_idx) in enumerate(scored):
                            cand_full_f.write(
                                json.dumps(
                                    {
                                        "row_id": int(rid),
                                        "model": model_key,
                                        "pref_profile": args.pref_profile,
                                        "align_context": args.align_context,
                                        "pref_score_version": getattr(args, "pref_score_version", "v2_signalaware"),
                                        "k": int(args.k),
                                        "retrieved_pos": rpos,
                                        "input_text": inp,
                                        "retrieval_text_hash": ret_hash,
                                        "retrieval_text": rtxt,
                                        "candidate_index": int(orig_idx),
                                        "candidate_rank": int(rank),
                                        "candidate_text": txt,
                                        "resp_tokens": int(tok_cnt),
                                        "pref_score": float(sc),
                                        "components": comps,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )

                    # Pair acceptance filters
                    if float(best_sc) < float(args.quality_floor): continue

                    uahe_cap = float(getattr(args, "uahe_cap", -1.0))
                    if uahe_cap > 0 and float(best_comps.get("UAHE_per_1kTok", 0.0)) > uahe_cap: continue

                    min_cond = float(getattr(args, "min_cond_filled_rate", 0.0))
                    if min_cond > 0 and float(best_comps.get("CondFilledRate", best_comps.get("SecFilled", 0.0))) < min_cond: continue

                    need = max(1, int(getattr(args, "pairs_per_prompt", 1)))
                    chosen_uahe = float(best_comps.get("UAHE_per_1kTok", 0.0))
                    enforce_uahe = bool(getattr(args, "enforce_uahe_order", False))

                    rejects: List[Tuple[float, Dict[str, float], str, int]] = []
                    for sc, comps, txt, tok_cnt, _orig_idx in reversed(scored):
                        if txt.strip() == best_text.strip(): continue
                        if (float(best_sc) - float(sc)) < float(args.margin): continue
                        if enforce_uahe and chosen_uahe > float(comps.get("UAHE_per_1kTok", 0.0)): continue
                        rejects.append((float(sc), comps, txt, int(tok_cnt)))
                        if len(rejects) >= need: break

                    if not rejects: continue

                    for rj_sc, rj_comps, rj_text, _rj_tok in rejects:
                        if best_text.strip() == rj_text.strip(): continue
                        pair_rec: Dict[str, Any] = {
                            "row_id": int(rid), "prompt": inp, "chosen": best_text, "rejected": rj_text,
                            "chosen_score": float(best_sc), "rejected_score": float(rj_sc),
                        }
                        if bool(getattr(args, "save_pref_components", False)):
                            pair_rec["chosen_components"] = best_comps
                            pair_rec["rejected_components"] = rj_comps
                        pair_rows.append(pair_rec)

                # 로그 출력
                if (start // input_bs) % max(1, int(args.log_every)) == 0 and start > 0:
                    elapsed = time.time() - t0
                    print(
                        f"[DPO-MINE] {model_key} processed {min(start+input_bs,len(train_df))}/{len(train_df)} "
                        f"pairs={len(pair_rows)} elapsed={elapsed:.1f}s"
                    )

            with out_jsonl.open("w", encoding="utf-8") as f:
                for r in pair_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            pd.DataFrame(pair_rows).to_csv(out_csv, index=False)
            pd.DataFrame(cand_rows).to_csv(cand_csv, index=False)

            atomic_write_json(
                out_jsonl.with_suffix(".meta.json"),
                {
                    "command": "mine_dpo_pairs",
                    "timestamp": run_id,
                    "run_tag": run_tag,
                    "config_hash": cfg_hash,
                    "config_for_hash": cfg_for_hash,
                    "model_key": model_key,
                    "model_id": model_id,
                    "generator_adapter": str(args.generator_adapter),
                    "generator_adapter_path": str(adapter_path) if adapter_path else None,
                    "pairs": int(len(pair_rows)),
                    "candidates_csv": str(cand_csv),
                    "candidates_full_jsonl_gz": (str(cand_full_path) if cand_full_f is not None else None),
                    "retrieval_cache": (str(cache_path) if cache_path is not None else None),
                },
            )

            print(f"[DPO-MINE] Saved pairs: {out_jsonl} (n={len(pair_rows)})")
            print(f"[DPO-MINE] Saved pairs CSV: {out_csv}")
            print(f"[DPO-MINE] Saved candidate summary CSV: {cand_csv}")
            if cand_full_f is not None:
                print(f"[DPO-MINE] Saved full candidates: {cand_full_path}")

        finally:
            try:
                if cand_full_f is not None:
                    cand_full_f.close()
            except Exception:
                pass
            try:
                import torch
                del model
                del tok
                del prep_tok
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            cleanup_cuda()


def preference_score(
    *,
    input_text: str,
    retrieval_text: str,
    output_text: str,
    resp_tokens: int,
    pref_profile: Literal["balanced", "hard", "structure"],
    score_version: Literal["v1", "v2_signalaware"] = "v2_signalaware",
    input_sets: Optional[Dict[str, set]] = None,
    retrieval_sets: Optional[Dict[str, set]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Preference score for mining DPO pairs (no GT needed)."""
    sec = section_metrics(output_text)
    align = section_alignment_metrics2(input_text, output_text)

    if input_sets is None:
        input_sets = hard_entity_sets(input_text)
    if retrieval_sets is None:
        retrieval_sets = hard_entity_sets(retrieval_text)

    hattr = hard_attribution_metrics_precomputed(
        input_sets=input_sets,
        retrieval_sets=retrieval_sets,
        output_text=output_text,
        resp_tokens=int(resp_tokens),
    )

    sec_filled = float(sec.get("SecFilled", 0.0))
    sec_presence = float(sec.get("SecPresence", 0.0))
    placeholder = float(sec.get("PlaceholderRate", 1.0))
    unknown = float(sec.get("UnknownRate", 1.0))
    dup = float(sec.get("DupSectionCount", 0.0))

    ua_per = float(hattr.get("UAHE_per_1kTok", 0.0))
    iu_per = float(hattr.get("IUHE_per_1kTok", 0.0))

    cond_filled = float(align.get("CondFilledRate", sec_filled))
    signal_filled = float(align.get("SignalFilledRate", cond_filled))
    no_signal_filled = float(align.get("NoSignalFilledRate", cond_filled))

    tok = max(1.0, float(resp_tokens))
    length_pen = 0.0
    if tok < 200:
        length_pen = (200.0 - tok) / 2000.0
    elif tok > 1200:
        length_pen = (tok - 1200.0) / 4000.0

    dup_pen = min(0.2, dup * 0.03)
    ver = str(score_version).lower().strip()

    if ver == "v1":
        if pref_profile == "structure":
            score = 0.55 * sec_filled + 0.25 * sec_presence
            score -= 0.10 * placeholder + 0.08 * unknown
            score -= 0.10 * dup_pen
            score -= 0.05 * length_pen
            score -= min(0.25, ua_per * 0.03)
        elif pref_profile == "hard":
            score = 0.35 * sec_filled + 0.15 * sec_presence
            score -= 0.10 * placeholder + 0.08 * unknown
            score -= 0.10 * dup_pen
            score -= 0.05 * length_pen
            score -= min(0.60, ua_per * 0.06)
            score -= min(0.25, (iu_per - ua_per) * 0.02)
        else:  
            score = 0.45 * sec_filled + 0.20 * sec_presence
            score -= 0.10 * placeholder + 0.08 * unknown
            score -= 0.10 * dup_pen
            score -= 0.05 * length_pen
            score -= min(0.45, ua_per * 0.045)
    else:
        if pref_profile == "structure":
            score = 0.65 * cond_filled + 0.10 * sec_presence
            score += 0.10 * signal_filled + 0.05 * no_signal_filled
            score -= 0.08 * placeholder + 0.06 * unknown
            score -= 0.10 * dup_pen
            score -= 0.05 * length_pen
            score -= min(0.25, ua_per * 0.03)
        elif pref_profile == "hard":
            score = 0.45 * cond_filled + 0.10 * sec_presence
            score += 0.10 * signal_filled + 0.05 * no_signal_filled
            score -= 0.08 * placeholder + 0.06 * unknown
            score -= 0.10 * dup_pen
            score -= 0.05 * length_pen
            score -= min(0.60, ua_per * 0.06)
            score -= min(0.25, (iu_per - ua_per) * 0.02)
        else:  
            score = 0.55 * cond_filled + 0.10 * sec_presence
            score += 0.10 * signal_filled + 0.05 * no_signal_filled
            score -= 0.08 * placeholder + 0.06 * unknown
            score -= 0.10 * dup_pen
            score -= 0.05 * length_pen
            score -= min(0.45, ua_per * 0.045)

    score = float(max(0.0, min(1.0, score)))
    comps = {
        **sec,
        **align,
        **hattr,
        "PrefScore": score,
        "PrefScoreVersion": ver,
        "LengthPenaltyTok": float(length_pen),
    }
    return score, comps


def cmd_train_dpo(args: argparse.Namespace) -> None:
    """Train a DPO adapter from preference pairs."""
    project_root = Path(args.project_root).resolve()
    set_seed(int(args.seed))

    try:
        import torch
        import inspect
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel, prepare_model_for_kbit_training
        from datasets import load_dataset  # type: ignore
        from trl import DPOTrainer, DPOConfig  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "DPO training requires extra deps: torch, transformers, peft, datasets, trl. "
            f"Import failed: {e}"
        )

    splits = load_cached_splits(project_root, seed=int(args.seed))
    if splits is None:
        raise RuntimeError("Cached splits not found. Run data-prepare first.")

    system_prompt = SYSTEM_PROMPT_TEMPLATE_ON
    run_tag = _sanitize_run_tag(getattr(args, "run_tag", None))

    dense_cfg = DenseIndexConfig(emb_model=str(getattr(args, "dense_emb_model", DEFAULT_DENSE_EMB_MODEL)), normalize=True)
    lex_cfg = LexicalIndexConfig(
        backend=str(getattr(args, "lex_backend", "tfidf")),
        max_features=int(getattr(args, "tfidf_max_features", 200000)),
        ngram=int(getattr(args, "tfidf_ngram", 2)),
        bm25_k1=float(getattr(args, "bm25_k1", 1.5)),
        bm25_b=float(getattr(args, "bm25_b", 0.75)),
    )

    train_df = splits.train.reset_index(drop=True)
    train_inputs = train_df[args.input_col].astype(str).tolist()
    train_outputs = train_df[args.gt_col].astype(str).tolist()

    align_ctx = str(getattr(args, "align_context", "retrieval_free"))
    k_val = int(getattr(args, "k", 0))

    cache_map: Optional[Dict[int, Dict[str, Any]]] = None
    if align_ctx == "retrieval_aware" and k_val > 0:
        cache_path = build_retrieval_cache(
            project_root=project_root,
            seed=int(args.seed),
            input_col=args.input_col,
            split="train",
            retriever_type=str(getattr(args, "retriever_type", "dense")),
            retriever_mode=str(getattr(args, "retriever_mode", "similar")),
            k=k_val,
            dense_cfg=dense_cfg,
            lex_cfg=lex_cfg,
            dense_device=str(getattr(args, "dense_device", "cuda")),
            dense_batch_size=int(getattr(args, "dense_batch_size", 256)),
            leave_one_out=True,
            exclude_same_group=True,  # EMSE PATCH
            group_col="group_id",
            force=bool(getattr(args, "force_cache", False)),
        )
        cache_map = load_retrieval_cache(cache_path)
        print(f"[DPO] Using train retrieval cache: {cache_path}")

    def _examples_for_row(row_id: int) -> List[Tuple[str, str]]:
        if cache_map is None or k_val <= 0:
            return []
        r = cache_map.get(int(row_id))
        if not r:
            return []
        idxs = [int(x) for x in r.get("retrieved_pos", [])][:k_val]
        ex: List[Tuple[str, str]] = []
        ret_mask = str(getattr(args, "retrieval_mask", "none"))
        for pos in idxs:
            if 0 <= pos < len(train_inputs):
                ex_in, ex_out = train_inputs[pos], train_outputs[pos]
                if ret_mask == "hardmask":
                    ex_in, ex_out = mask_hard_entities(ex_in), mask_hard_entities(ex_out)
                ex.append((ex_in, ex_out))
        return ex

    def _resolve_pairs_for_model(model_key: str) -> Path:
        if getattr(args, "dpo_pairs_path", None):
            return Path(args.dpo_pairs_path).resolve()

        dpo_dir = project_root / DPO_DIR_REL / f"seed{int(args.seed)}" / args.pref_profile
        patt = f"dpo_pairs_{model_key}_seed{int(args.seed)}_{args.pref_profile}_{align_ctx}_*.jsonl"
        cand = sorted(dpo_dir.glob(patt), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cand:
            raise FileNotFoundError(f"No DPO pair file matched: {patt} under {dpo_dir}")
        return cand[0]

    for model_key in args.models:
        pairs_path = _resolve_pairs_for_model(model_key)
        if model_key not in MODEL_ID:
            raise KeyError(f"Unknown model key: {model_key}.")
        model_id = MODEL_ID[model_key]

        sft_root = project_root / ADAPTER_DIR_REL / "sft" / model_key
        sft_adapter = Path(args.sft_adapter) if getattr(args, "sft_adapter", None) else load_latest_adapter(sft_root)
        if sft_adapter is None or not sft_adapter.exists():
            raise RuntimeError(f"SFT adapter not found for {model_key}. Train SFT first or pass --sft-adapter")

        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        quant_config = None
        lora_type = getattr(args, "lora_type", "qlora")
        if lora_type == "qlora":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )

        load_kwargs = {"device_map": "auto", "trust_remote_code": True}
        if lora_type == "qlora" and quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
        else:
            load_kwargs["torch_dtype"] = compute_dtype

        base = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        tok = load_tokenizer_robust(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "left"
        tok.truncation_side = "left"

        if lora_type == "qlora":
            base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

        model = PeftModel.from_pretrained(base, str(sft_adapter), is_trainable=True)

        ds_full = load_dataset("json", data_files=str(pairs_path), split="train")

        def _format(row):
            inp = str(row["prompt"])
            rid = int(row.get("row_id", -1))
            ex = _examples_for_row(rid) if rid >= 0 else []
            row["prompt"] = build_prompt(inp, system_prompt, examples=ex)
            return row

        ds_full = ds_full.map(_format)
        
        cols_to_keep = {"prompt", "chosen", "rejected"}
        cols_to_remove = [c for c in ds_full.column_names if c not in cols_to_keep]
        if cols_to_remove:
            ds_full = ds_full.remove_columns(cols_to_remove)

        prompts = ds_full["prompt"]
        unique_prompts = list({p for p in prompts})
        rng = random.Random(int(args.seed))
        rng.shuffle(unique_prompts)
        
        eval_s = float(getattr(args, "eval_split", 0.1))
        eval_size = max(1, int(len(unique_prompts) * eval_s))
        eval_prompts = set(unique_prompts[:eval_size])
        
        train_idx = [i for i, p in enumerate(prompts) if p not in eval_prompts]
        eval_idx = [i for i, p in enumerate(prompts) if p in eval_prompts]
        
        if not train_idx or not eval_idx:
            ds_split = ds_full.train_test_split(test_size=eval_s, seed=int(args.seed))
            train_ds = ds_split["train"]
            eval_ds = ds_split["test"]
        else:
            train_ds = ds_full.select(train_idx)
            eval_ds = ds_full.select(eval_idx)

        cfg_for_hash = {
            "command": "train_dpo",
            "model_key": model_key,
            "seed": int(args.seed),
            "pref_profile": args.pref_profile,
            "align_context": align_ctx,
            "eval_split": eval_s,
            "lora_type": lora_type,
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "grad_accum_steps": int(args.grad_accum_steps),
            "beta": float(args.beta),
            "pairs_path": str(pairs_path),
            "k": k_val,
        }
        cfg_hash = _config_fingerprint(cfg_for_hash)
        run_id = now_id()

        out_root = ensure_dir(project_root / ADAPTER_DIR_REL / "dpo" / align_ctx / args.pref_profile / model_key)
        run_dir = ensure_dir(out_root / _join_nonempty(["RUN_DPO", run_id, run_tag, cfg_hash]))

        max_p_tokens = int(getattr(args, "max_prompt_tokens", 4096))
        max_n_tokens = int(getattr(args, "max_new_tokens", 1024))

        dpo_args_kwargs = dict(
            output_dir=str(run_dir),
            per_device_train_batch_size=int(args.batch_size),
            per_device_eval_batch_size=int(args.batch_size),
            gradient_accumulation_steps=int(args.grad_accum_steps),
            num_train_epochs=float(args.epochs),
            learning_rate=float(args.lr),
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=int(args.logging_steps),
            save_strategy="steps",
            save_steps=int(args.save_steps),
            evaluation_strategy="steps",
            eval_steps=int(args.save_steps),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
            report_to="none",
            remove_unused_columns=False,
            beta=float(args.beta),
            max_prompt_length=max_p_tokens,
            max_length=max_p_tokens + max_n_tokens,
        )
        
        try:
            cfg = DPOConfig(**dpo_args_kwargs)
        except TypeError:
            if "evaluation_strategy" in dpo_args_kwargs:
                dpo_args_kwargs["eval_strategy"] = dpo_args_kwargs.pop("evaluation_strategy")
                cfg = DPOConfig(**dpo_args_kwargs)
            else:
                raise

        trainer_kwargs = dict(
            model=model,
            ref_model=None,
            args=cfg,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
        )
        
        if "processing_class" in inspect.signature(DPOTrainer.__init__).parameters:
            trainer_kwargs["processing_class"] = tok
        else:
            trainer_kwargs["tokenizer"] = tok

        trainer = DPOTrainer(**trainer_kwargs)

        trainer.train()
        model.save_pretrained(str(run_dir))
        tok.save_pretrained(str(run_dir))

        atomic_write_json(
            run_dir / "train_config.json",
            {
                "command": "train_dpo",
                "timestamp": run_id,
                "run_tag": run_tag,
                "config_hash": cfg_hash,
                "config_for_hash": cfg_for_hash,
                "model_key": model_key,
                "model_id": model_id,
                "seed": int(args.seed),
                "pref_profile": args.pref_profile,
                "align_context": align_ctx,
                "pairs_path": str(pairs_path),
                "sft_adapter": str(sft_adapter),
            },
        )

        latest_txt = ensure_dir(out_root) / "LATEST_ADAPTER.txt"
        latest_txt.write_text(str(run_dir), encoding="utf-8")
        print(f"[DPO] Saved adapter: {run_dir}")


# ==============================
# CLI
# ==============================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="bugreport_pipeline_tse.py",
        description="TSE/EMSE-style bug report generation pipeline (GEN/SCORE split, retrieval caches, attribution metrics).",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # --- data-prepare ---
    p = sub.add_parser("data-prepare", help="Split raw Excel into train/val/test and cache CSVs (with row_id).")
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--raw-data-path", type=str, default=None)
    p.add_argument("--input-col", type=str, default=DEFAULT_INPUT_COL)
    p.add_argument("--gt-col", type=str, default=DEFAULT_GT_COL)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--keep-cols", type=str, default="")
    p.add_argument("--split-mode", type=str, choices=["group", "row"], default="group")
    p.add_argument("--group-col", type=str, default="group_id")

    # --- train-sft (optional) ---
    p = sub.add_parser("train-sft", help="Train an SFT adapter (optional; requires peft/bitsandbytes).")
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--models", nargs="+", required=True, choices=list(MODEL_ID.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--input-col", type=str, default=DEFAULT_INPUT_COL)
    p.add_argument("--gt-col", type=str, default=DEFAULT_GT_COL)
    p.add_argument("--template", type=str, choices=["on", "off"], default="on")
    p.add_argument("--run-tag", type=str, default=None)
    p.add_argument("--lora-type", type=str, choices=["qlora", "lora"], default="qlora")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=200)

    # --- mine-dpo-pairs (optional) ---
    p = sub.add_parser("mine-dpo-pairs", help="Mine DPO preference pairs (optional).")
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--models", nargs="+", required=True, choices=list(MODEL_ID.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--input-col", type=str, default=DEFAULT_INPUT_COL)
    p.add_argument("--gt-col", type=str, default=DEFAULT_GT_COL)
    p.add_argument("--run-tag", type=str, default=None)
    p.add_argument("--pref-profile", type=str, choices=["balanced", "hard", "structure"], default="balanced")
    p.add_argument("--align-context", type=str, choices=["retrieval_free", "retrieval_aware"], default="retrieval_free")
    p.add_argument("--generator-adapter", type=str, choices=["sft", "base"], default="sft")
    p.add_argument("--sft-adapter", type=str, default=None, help="Override SFT adapter path (else use latest).")
    
    
    # optional retrieval context for mining
    p.add_argument("--retriever-type", type=str, choices=["none", "dense", "lexical"], default="none")
    p.add_argument("--retriever-mode", type=str, choices=["none", "similar", "random"], default="none")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--retrieval-mask", type=str, choices=["none", "hardmask"], default="none")
    p.add_argument("--force-cache", action="store_true")

    p.add_argument("--dense-emb-model", type=str, default=DEFAULT_DENSE_EMB_MODEL)
    p.add_argument("--dense-device", type=str, choices=["cpu", "cuda", "auto"], default="cuda")
    p.add_argument("--dense-batch-size", type=int, default=256)
    p.add_argument("--lex-backend", type=str, choices=["tfidf", "bm25"], default="tfidf")
    p.add_argument("--tfidf-max-features", type=int, default=200000)
    p.add_argument("--tfidf-ngram", type=int, choices=[1, 2], default=2)
    p.add_argument("--bm25-k1", type=float, default=1.5)
    p.add_argument("--bm25-b", type=float, default=0.75)

    p.add_argument(
        "--pref-score-version",
        type=str,
        choices=["v1", "v2_signalaware"],
        default="v2_signalaware",
        help="Preference score definition version. v2_signalaware is recommended for EMSE.",
    )
    p.add_argument(
        "--pairs-per-prompt",
        type=int,
        default=2,
        help="Option2: mine multiple rejected pairs per prompt (chosen fixed as best).",
    )
    p.add_argument(
        "--uahe-cap",
        type=float,
        default=0.0,
        help="Optional filter: discard chosen candidates with UAHE_per_1kTok above this cap. <=0 disables.",
    )
    p.add_argument(
        "--min-cond-filled-rate",
        type=float,
        default=0.5,
        help="Optional filter: require chosen CondFilledRate >= this value (0 disables).",
    )
    p.add_argument(
        "--enforce-uahe-order",
        action="store_true",
        help="Optional: only accept pairs where chosen UAHE_per_1kTok <= rejected UAHE_per_1kTok.",
    )
    p.add_argument(
        "--save-candidates-jsonl-gz",
        action="store_true",
        help="Save all scored candidates (full text) to jsonl.gz for future re-mining without regeneration.",
    )

    p.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    p.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    p.add_argument("--max-ex-in-tokens", type=int, default=DEFAULT_MAX_EX_IN_TOKENS)
    p.add_argument("--max-ex-out-tokens", type=int, default=DEFAULT_MAX_EX_OUT_TOKENS)
    p.add_argument("--input-truncation", type=str, choices=["head", "tail", "head_tail"], default=DEFAULT_INPUT_TRUNCATION)

    p.add_argument("--precision", type=str, choices=["4bit", "fp16", "auto", 'bf16'], default="auto")
    p.add_argument("--micro-batch", type=int, default=0)
    p.add_argument(
        "--autotune-micro-batch",
        action="store_true",
        help="Autotune micro-batch for candidate generation (uses a single generate() call per test).",
    )

    p.add_argument(
        "--candidate-gen-mode",
        type=str,
        choices=["repeat", "num_return_sequences"],
        default="repeat",
        help=(
            "How to generate N candidates per input. "
            "'repeat' preserves legacy behavior; "
            "'num_return_sequences' is faster for large batches but may change the sampling stream."
        ),
    )
    p.add_argument(
        "--save-pref-components",
        action="store_true",
        help="Store per-candidate preference components for chosen/rejected in the pair file (larger output).",
    )

    p.add_argument("--num-candidates", type=int, default=8)
    p.add_argument("--input-batch-size", type=int, default=1)
    p.add_argument("--gen-max-new-tokens", type=int, default=768)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.98)
    p.add_argument("--quality-floor", type=float, default=0.35)
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)

    # --- train-dpo (optional) ---
    p = sub.add_parser("train-dpo", help="Train a DPO adapter (optional; requires trl/datasets).")
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--models", nargs="+", required=True, choices=list(MODEL_ID.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--input-col", type=str, default=DEFAULT_INPUT_COL)
    p.add_argument("--gt-col", type=str, default=DEFAULT_GT_COL)
    p.add_argument("--run-tag", type=str, default=None)
    p.add_argument("--pref-profile", type=str, choices=["balanced", "hard", "structure"], default="balanced")
    p.add_argument("--align-context", type=str, choices=["retrieval_free", "retrieval_aware"], default="retrieval_free")
    p.add_argument("--dpo-pairs-path", type=str, default=None)
    p.add_argument("--sft-adapter", type=str, default=None)
    p.add_argument("--merge-sft", action="store_true", help="Merge SFT adapter into base before DPO (optional).")
    p.add_argument("--lora-type", type=str, choices=["qlora", "lora"], default="qlora")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--eval-split", type=float, default=0.1)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)

    p.add_argument("--retriever-type", type=str, choices=["none", "dense", "lexical"], default="none")
    p.add_argument("--retriever-mode", type=str, choices=["none", "similar", "random"], default="none")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--retrieval-mask", type=str, choices=["none", "hardmask"], default="none")
    p.add_argument("--force-cache", action="store_true")
    p.add_argument("--dense-emb-model", type=str, default=DEFAULT_DENSE_EMB_MODEL)
    p.add_argument("--dense-device", type=str, choices=["cpu", "cuda", "auto"], default="cuda")
    p.add_argument("--dense-batch-size", type=int, default=256)
    p.add_argument("--lex-backend", type=str, choices=["tfidf", "bm25"], default="tfidf")
    p.add_argument("--tfidf-max-features", type=int, default=200000)
    p.add_argument("--tfidf-ngram", type=int, choices=[1, 2], default=2)
    p.add_argument("--bm25-k1", type=float, default=1.5)
    p.add_argument("--bm25-b", type=float, default=0.75)

    # --- retrieval-build ---
    p = sub.add_parser("retrieval-build", help="Build retrieval indexes (dense/lexical) once.")
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--input-col", type=str, default=DEFAULT_INPUT_COL)
    p.add_argument("--retriever-type", type=str, choices=["dense", "lexical", "both"], default="both")
    p.add_argument("--dense-emb-model", type=str, default=DEFAULT_DENSE_EMB_MODEL)
    p.add_argument("--dense-device", type=str, choices=["cpu", "cuda", "auto"], default="cuda")
    p.add_argument("--dense-batch-size", type=int, default=256)
    p.add_argument("--lex-backend", type=str, choices=["tfidf", "bm25"], default="tfidf")
    p.add_argument("--tfidf-max-features", type=int, default=200000)
    p.add_argument("--tfidf-ngram", type=int, choices=[1, 2], default=2)
    p.add_argument("--bm25-k1", type=float, default=1.5)
    p.add_argument("--bm25-b", type=float, default=0.75)
    p.add_argument("--force", action="store_true")

    # --- retrieval-cache ---
    p = sub.add_parser(
        "retrieval-cache",
        help="Precompute top-k retrieval results for a split (train corpus = train split).",
    )
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--input-col", type=str, default=DEFAULT_INPUT_COL)
    p.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    p.add_argument("--retriever-type", type=str, choices=["dense", "lexical"], required=True)
    p.add_argument("--retriever-mode", type=str, choices=["similar", "random"], default="similar")
    p.add_argument("--k-values", type=str, default="0,1,2")
    p.add_argument("--dense-emb-model", type=str, default=DEFAULT_DENSE_EMB_MODEL)
    p.add_argument("--dense-device", type=str, choices=["cpu", "cuda", "auto"], default="cuda")
    p.add_argument("--dense-batch-size", type=int, default=256)
    p.add_argument("--lex-backend", type=str, choices=["tfidf", "bm25"], default="tfidf")
    p.add_argument("--tfidf-max-features", type=int, default=200000)
    p.add_argument("--tfidf-ngram", type=int, choices=[1, 2], default=2)
    p.add_argument("--bm25-k1", type=float, default=1.5)
    p.add_argument("--bm25-b", type=float, default=0.75)
    p.add_argument("--random-seed-offset", type=int, default=7)
    p.add_argument("--leave-one-out", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-exclude-same-group", action="store_true", help="Disable group exclusion (NOT recommended).")
    p.add_argument("--group-col", type=str, default="group_id")

    # --- gen ---
    p = sub.add_parser(
        "gen",
        help="GEN stage: generate + lightweight metrics to jsonl.gz (no heavy scoring).",
    )
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-tag", type=str, default="none")
    p.add_argument(
        "--run-dir",
        type=str,
        default="auto",
        help="Run directory: auto|new creates a new results/runs/<timestamp>_<tag>_seed<seed>/; last uses LAST_RUN.txt; or provide a path.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing artifacts (ignore *.done).")
    p.add_argument("--lock-stale-hours", type=float, default=24.0, help="Remove lock if older than this (hours).")

    p.add_argument("--models", nargs="+", required=True, choices=list(MODEL_ID.keys()))
    p.add_argument("--input-col", type=str, default=DEFAULT_INPUT_COL)
    p.add_argument("--gt-col", type=str, default=DEFAULT_GT_COL)

    p.add_argument("--adapter", type=str, choices=["base", "sft", "dpo"], default="base")
    p.add_argument("--pref-profile", type=str, choices=["none", "balanced", "hard", "structure"], default="none")
    p.add_argument(
        "--align-context",
        type=str,
        choices=["none", "retrieval_free", "retrieval_aware"],
        default="none",
    )
    p.add_argument("--template", type=str, choices=["on", "off"], default="on")

    p.add_argument("--retriever-type", type=str, choices=["none", "dense", "lexical"], default="none")
    p.add_argument("--retriever-mode", type=str, choices=["none", "similar", "random"], default="none")
    p.add_argument("--k", type=int, default=0)
    p.add_argument("--retrieval-mask", type=str, choices=["none", "hardmask"], default="none")
    p.add_argument("--retrieval-cache", type=str, default="auto")

    p.add_argument("--dense-emb-model", type=str, default=DEFAULT_DENSE_EMB_MODEL)
    p.add_argument("--lex-backend", type=str, choices=["tfidf", "bm25"], default="tfidf")
    p.add_argument("--tfidf-max-features", type=int, default=200000)
    p.add_argument("--tfidf-ngram", type=int, choices=[1, 2], default=2)
    p.add_argument("--bm25-k1", type=float, default=1.5)
    p.add_argument("--bm25-b", type=float, default=0.75)

    p.add_argument("--decode", type=str, choices=["greedy", "sample"], default="greedy")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.98)

    p.add_argument("--precision", type=str, choices=["auto", "4bit", "fp16", "bf16"], default="auto")
    p.add_argument("--macro-batch", type=int, default=128)
    p.add_argument("--micro-batch", type=int, default=0)
    p.add_argument("--autotune-micro-batch", action="store_true")
    p.add_argument("--tokenizer-threads", type=int, default=30)
    p.add_argument("--regex-processes", type=int, default=20)
    p.add_argument("--pipeline-prefetch", type=int, default=8)

    p.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    p.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    p.add_argument("--max-ex-in-tokens", type=int, default=DEFAULT_MAX_EX_IN_TOKENS)
    p.add_argument("--max-ex-out-tokens", type=int, default=DEFAULT_MAX_EX_OUT_TOKENS)
    p.add_argument("--input-truncation", type=str, choices=["head", "tail", "head_tail"], default=DEFAULT_INPUT_TRUNCATION)

    p.add_argument("--limit", type=int, default=0)

    p.add_argument(
        "--no-export-csv",
        action="store_true",
        help="Do not create a companion *.csv.gz next to each *.jsonl.gz artifact.",
    )

    # --- gen-suite ---
    p = sub.add_parser(
        "gen-suite",
        help="Run a built-in grid of GEN conditions (main/ablation/mechanism) across models.",
    )
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-tag", type=str, default="none")
    p.add_argument(
        "--run-dir",
        type=str,
        default="auto",
        help="Run directory: auto|new creates a new results/runs/<timestamp>_<tag>_seed<seed>/; last uses LAST_RUN.txt; or provide a path.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--lock-stale-hours", type=float, default=24.0)

    p.add_argument("--suite", type=str, choices=["main", "ablation", "mechanism", "all"], default="main")
    p.add_argument("--models", nargs="+", required=True, choices=list(MODEL_ID.keys()))
    p.add_argument("--input-col", type=str, default=DEFAULT_INPUT_COL)
    p.add_argument("--gt-col", type=str, default=DEFAULT_GT_COL)

    p.add_argument("--adapter", type=str, choices=["base", "sft", "dpo"], default="base")
    p.add_argument("--pref-profile", type=str, choices=["none", "balanced", "hard", "structure"], default="none")
    p.add_argument("--align-context", type=str, choices=["none", "retrieval_free", "retrieval_aware"], default="none")

    p.add_argument("--retrieval-cache", type=str, default="auto")

    p.add_argument("--dense-emb-model", type=str, default=DEFAULT_DENSE_EMB_MODEL)
    p.add_argument("--lex-backend", type=str, choices=["tfidf", "bm25"], default="tfidf")
    p.add_argument("--tfidf-max-features", type=int, default=200000)
    p.add_argument("--tfidf-ngram", type=int, choices=[1, 2], default=2)
    p.add_argument("--bm25-k1", type=float, default=1.5)
    p.add_argument("--bm25-b", type=float, default=0.75)

    p.add_argument("--decode", type=str, choices=["greedy", "sample"], default="greedy")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.98)

    p.add_argument("--precision", type=str, choices=["auto", "4bit", "fp16", "bf16"], default="auto")
    p.add_argument("--macro-batch", type=int, default=128)
    p.add_argument("--micro-batch", type=int, default=0)
    p.add_argument("--autotune-micro-batch", action="store_true")
    p.add_argument("--tokenizer-threads", type=int, default=30)
    p.add_argument("--regex-processes", type=int, default=20)
    p.add_argument("--pipeline-prefetch", type=int, default=8)

    p.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    p.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    p.add_argument("--max-ex-in-tokens", type=int, default=DEFAULT_MAX_EX_IN_TOKENS)
    p.add_argument("--max-ex-out-tokens", type=int, default=DEFAULT_MAX_EX_OUT_TOKENS)
    p.add_argument("--input-truncation", type=str, choices=["head", "tail", "head_tail"], default=DEFAULT_INPUT_TRUNCATION)

    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--no-export-csv",
        action="store_true",
        help="Do not create a companion *.csv.gz next to each *.jsonl.gz artifact.",
    )

    # --- score ---
    p = sub.add_parser("score", help="SCORE stage: offline ROUGE/CTQRS/SBERT on GEN outputs.")
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="If set, reads GEN from <run_dir>/gen and writes scored outputs to <run_dir>/scored. "
        "Use 'last' to use LAST_RUN.txt.",
    )
    p.add_argument("--gen-path", type=str, default=None, help="Path or glob to GEN outputs (jsonl.gz). Optional if --run-dir is used.")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory for scored files (ignored if --run-dir is used).")
    p.add_argument("--run-tag", type=str, default="none")
    p.add_argument("--metrics", type=str, default="rouge,sbert,ctqrs")
    p.add_argument("--do-normalize-for-similarity", action="store_true")
    p.add_argument(
        "--no-export-csv",
        action="store_true",
        help="Do not create a companion *.csv.gz next to each *.jsonl.gz artifact.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--lock-stale-hours", type=float, default=24.0)
    p.add_argument(
        "--include-incomplete-gen",
        action="store_true",
        help="Also score GEN files without *.done (not recommended while GEN is running).",
    )

    p.add_argument("--ctqrs-path", type=str, default=str(Path("evaluation") / "perfect_ctqrs.py"))
    p.add_argument("--rouge-processes", type=int, default=32)
    p.add_argument("--ctqrs-processes", type=int, default=32)

    p.add_argument("--sbert-model", type=str, default=DEFAULT_SBERT_MODEL)
    p.add_argument("--sbert-device", type=str, choices=["cpu", "cuda", "auto"], default="cuda")
    p.add_argument("--sbert-batch-size", type=int, default=512)

    # --- aggregate ---
    p = sub.add_parser("aggregate", help="Aggregate scored outputs into integrated summary with CI.")
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--run-dir", type=str, default=None, help="If set, aggregates from <run_dir>/scored into <run_dir>/summary.")
    p.add_argument("--scored-glob", type=str, default=None, help="Glob for scored jsonl.gz (overrides --run-dir).")
    p.add_argument("--out-csv", type=str, default=None)
    p.add_argument(
        "--groupby",
        type=str,
        default="model,adapter_type,pref_profile,align_context,retriever_type,retriever_mode,k,template,retrieval_mask,seed",
    )
    p.add_argument("--stat", type=str, choices=["bootstrap_ci", "mean_ci", "median_iqr"], default="bootstrap_ci")
    p.add_argument("--bootstrap-iters", type=int, default=2000)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)

    # --- verify-gen ---
    p = sub.add_parser("verify-gen", help="Quickly verify GEN outputs in a run directory (integrity + counts).")
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--run-dir", type=str, default="last")
    p.add_argument("--max-files", type=int, default=0, help="If >0, verify only the first N GEN files (sorted by mtime desc).")
    p.add_argument("--sample-n", type=int, default=3, help="Show this many sample rows per GEN artifact.")
    p.add_argument("--full", action="store_true", help="Compute full-file means for a few key columns (reads whole file).")

    return ap


def cmd_data_prepare(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    raw_path = Path(args.raw_data_path).resolve() if args.raw_data_path else (project_root / DEFAULT_RAW_XLSX_REL)
    keep_cols = [c.strip() for c in str(args.keep_cols).split(",") if c.strip()]

    df = read_raw_dataset(raw_path, args.input_col, args.gt_col, keep_cols=keep_cols)
    splits = split_dataset(
        df,
        seed=int(args.seed),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        split_mode=getattr(args, "split_mode", "group"),
        group_col=getattr(args, "group_col", "group_id"),
    )

    cache_splits(project_root, splits, seed=int(args.seed))
    
    group_col = getattr(args, "group_col", "group_id")
    save_meta(
        project_root / SPLIT_DIR_REL / f"split_seed{int(args.seed)}.meta.json",
        {
            "command": "data-prepare",
            "timestamp": now_id(),
            "seed": int(args.seed),
            "raw_path": str(raw_path),
            "input_col": args.input_col,
            "gt_col": args.gt_col,
            "keep_cols": keep_cols,
            "split_mode": getattr(args, "split_mode", "group"),
            "group_col": group_col,
            "sizes": {"train": len(splits.train), "val": len(splits.val), "test": len(splits.test)},
            "unique_groups": {
                "train": int(splits.train[group_col].nunique()),
                "val": int(splits.val[group_col].nunique()),
                "test": int(splits.test[group_col].nunique()),
            },
        },
    )
    print(f"[DATA] Split saved under {project_root / SPLIT_DIR_REL}")


def cmd_retrieval_build(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    seed = int(args.seed)
    splits = load_cached_splits(project_root, seed)
    if splits is None:
        raise RuntimeError("Cached splits not found. Run data-prepare first.")
    train_df = splits.train.reset_index(drop=True)
    if args.input_col not in train_df.columns:
        raise KeyError(f"Missing column '{args.input_col}' in cached train split.")
    train_texts = train_df[args.input_col].astype(str).tolist()

    dense_cfg = DenseIndexConfig(emb_model=args.dense_emb_model, normalize=True)
    lex_cfg = LexicalIndexConfig(
        backend=str(args.lex_backend),
        max_features=int(args.tfidf_max_features),
        ngram=int(args.tfidf_ngram),
        bm25_k1=float(args.bm25_k1),
        bm25_b=float(args.bm25_b),
    )

    if args.retriever_type in ("dense", "both"):
        p = build_dense_index(
            project_root=project_root,
            seed=seed,
            train_texts=train_texts,
            cfg=dense_cfg,
            input_col=args.input_col,
            device=args.dense_device,
            batch_size=int(args.dense_batch_size),
            force=bool(args.force),
        )
        print(f"[RET] Dense index: {p}")

    if args.retriever_type in ("lexical", "both"):
        p = build_lexical_index(
            project_root=project_root,
            seed=seed,
            train_texts=train_texts,
            cfg=lex_cfg,
            input_col=args.input_col,
            force=bool(args.force),
        )
        print(f"[RET] Lexical index: {p}")


def cmd_retrieval_cache(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    seed = int(args.seed)

    dense_cfg = DenseIndexConfig(emb_model=args.dense_emb_model, normalize=True)
    lex_cfg = LexicalIndexConfig(
        backend=str(args.lex_backend),
        max_features=int(args.tfidf_max_features),
        ngram=int(args.tfidf_ngram),
        bm25_k1=float(args.bm25_k1),
        bm25_b=float(args.bm25_b),
    )

    ks = [int(x) for x in str(args.k_values).split(",") if x.strip()]
    ks = [k for k in ks if k > 0]
    if not ks:
        print("[RET] No k>0 provided; nothing to cache.")
        return

    exclude_same_group = not bool(getattr(args, "no_exclude_same_group", False))

    for k in ks:
        out = build_retrieval_cache(
            project_root=project_root,
            seed=seed,
            input_col=args.input_col,
            split=args.split,
            retriever_type=args.retriever_type,
            retriever_mode=args.retriever_mode,
            k=int(k),
            dense_cfg=dense_cfg,
            lex_cfg=lex_cfg,
            dense_device=args.dense_device,
            dense_batch_size=int(args.dense_batch_size),
            random_seed_offset=int(args.random_seed_offset),
            leave_one_out=bool(args.leave_one_out),
            exclude_same_group=exclude_same_group,
            group_col=str(getattr(args, "group_col", "group_id")),
            force=bool(args.force),
        )
        print(f"[RET] Cache saved: {out}")


def cmd_gen(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    run_dir, created = resolve_run_dir(
        project_root=project_root,
        run_dir_arg=args.run_dir,
        run_tag=args.run_tag,
        seed=int(args.seed),
        create=True,
    )
    print(f"[RUN] run_dir={run_dir} (created_new={created})")

    _ = gen_run(
        project_root=project_root,
        run_dir=run_dir,
        seed=int(args.seed),
        run_tag=args.run_tag,
        models=list(args.models),
        input_col=args.input_col,
        gt_col=args.gt_col,
        adapter_type=args.adapter,
        pref_profile=args.pref_profile,
        align_context=args.align_context,
        template=args.template,
        retriever_type=args.retriever_type,
        retriever_mode=args.retriever_mode,
        k=int(args.k),
        retrieval_mask=args.retrieval_mask,
        retrieval_cache=args.retrieval_cache,
        dense_emb_model=args.dense_emb_model,
        lex_backend=args.lex_backend,
        tfidf_max_features=int(args.tfidf_max_features),
        tfidf_ngram=int(args.tfidf_ngram),
        bm25_k1=float(args.bm25_k1),
        bm25_b=float(args.bm25_b),
        decode=args.decode,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        precision=args.precision,
        macro_batch=int(args.macro_batch),
        micro_batch=int(args.micro_batch),
        autotune_mb=bool(args.autotune_micro_batch),
        tokenizer_threads=int(args.tokenizer_threads),
        regex_processes=int(args.regex_processes),
        pipeline_prefetch=int(args.pipeline_prefetch),
        max_prompt_tokens=int(args.max_prompt_tokens),
        max_input_tokens=int(args.max_input_tokens),
        max_ex_in_tokens=int(args.max_ex_in_tokens),
        max_ex_out_tokens=int(args.max_ex_out_tokens),
        input_truncation=args.input_truncation,
        limit=int(args.limit),
        export_csv=(not bool(args.no_export_csv)),
        overwrite=bool(args.overwrite),
        lock_stale_hours=float(args.lock_stale_hours),
    )


def cmd_gen_suite(args: argparse.Namespace) -> None:
    suite = str(args.suite)
    suites: Dict[str, List[Dict[str, Any]]] = {
        "main": [
            {"template": "on", "retriever_type": "none", "retriever_mode": "none", "k": 0, "retrieval_mask": "none"},
            {"template": "on", "retriever_type": "dense", "retriever_mode": "similar", "k": 1, "retrieval_mask": "none"},
            {"template": "on", "retriever_type": "dense", "retriever_mode": "similar", "k": 2, "retrieval_mask": "none"},
        ],
        "ablation": [
            {"template": "on", "retriever_type": "dense", "retriever_mode": "random", "k": 2, "retrieval_mask": "none"},
            {"template": "off", "retriever_type": "dense", "retriever_mode": "similar", "k": 2, "retrieval_mask": "none"},
            {"template": "on", "retriever_type": "lexical", "retriever_mode": "similar", "k": 2, "retrieval_mask": "none"},
            {"template": "on", "retriever_type": "dense", "retriever_mode": "similar", "k": 2, "retrieval_mask": "hardmask"},
        ],
        "mechanism": [
            {"template": "on", "retriever_type": "dense", "retriever_mode": "similar", "k": 1, "retrieval_mask": "hardmask"},
            {"template": "on", "retriever_type": "dense", "retriever_mode": "random", "k": 1, "retrieval_mask": "none"},
            {"template": "on", "retriever_type": "lexical", "retriever_mode": "similar", "k": 1, "retrieval_mask": "none"},
        ],
    }

    if suite == "all":
        conds: List[Dict[str, Any]] = []
        seen = set()
        for s in ("main", "ablation", "mechanism"):
            for c in suites[s]:
                key = json.dumps(c, sort_keys=True)
                if key not in seen:
                    conds.append(c)
                    seen.add(key)
    else:
        conds = suites[suite]

    project_root = Path(args.project_root).resolve()
    run_dir, created = resolve_run_dir(
        project_root=project_root,
        run_dir_arg=args.run_dir,
        run_tag=f"{args.run_tag}_{suite}",
        seed=int(args.seed),
        create=True,
    )
    print(f"[RUN] run_dir={run_dir} (created_new={created})")

    for c in conds:
        run_tag = f"{args.run_tag}_{suite}"
        gen_run(
            project_root=project_root,
            run_dir=run_dir,
            seed=int(args.seed),
            run_tag=run_tag,
            models=list(args.models),
            input_col=args.input_col,
            gt_col=args.gt_col,
            adapter_type=args.adapter,
            pref_profile=args.pref_profile,
            align_context=args.align_context,
            template=c["template"],
            retriever_type=c["retriever_type"],
            retriever_mode=c["retriever_mode"],
            k=int(c["k"]),
            retrieval_mask=c["retrieval_mask"],
            retrieval_cache=args.retrieval_cache,
            dense_emb_model=args.dense_emb_model,
            lex_backend=args.lex_backend,
            tfidf_max_features=int(args.tfidf_max_features),
            tfidf_ngram=int(args.tfidf_ngram),
            bm25_k1=float(args.bm25_k1),
            bm25_b=float(args.bm25_b),
            decode=args.decode,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            precision=args.precision,
            macro_batch=int(args.macro_batch),
            micro_batch=int(args.micro_batch),
            autotune_mb=bool(args.autotune_micro_batch),
            tokenizer_threads=int(args.tokenizer_threads),
            regex_processes=int(args.regex_processes),
            pipeline_prefetch=int(args.pipeline_prefetch),
            max_prompt_tokens=int(args.max_prompt_tokens),
            max_input_tokens=int(args.max_input_tokens),
            max_ex_in_tokens=int(args.max_ex_in_tokens),
            max_ex_out_tokens=int(args.max_ex_out_tokens),
            input_truncation=args.input_truncation,
            limit=int(args.limit),
            export_csv=(not bool(args.no_export_csv)),
            overwrite=bool(args.overwrite),
            lock_stale_hours=float(args.lock_stale_hours),
        )


def cmd_score(args: argparse.Namespace) -> None:
    import glob

    project_root = Path(args.project_root).resolve()

    use_run_dir = args.run_dir is not None or args.gen_path is None
    run_dir: Optional[Path] = None

    if use_run_dir:
        run_dir_arg = args.run_dir if args.run_dir is not None else "last"
        run_dir, _ = resolve_run_dir(
            project_root=project_root,
            run_dir_arg=run_dir_arg,
            run_tag=args.run_tag,
            seed=0,
            create=True,
        )
        gen_glob = args.gen_path if args.gen_path else str(run_dir / "gen" / "gen_*.jsonl.gz")
        out_dir = run_dir / "scored"
    else:
        gen_glob = args.gen_path
        out_dir = Path(args.out_dir).resolve() if args.out_dir else (project_root / RESULTS_DIR_REL / "scored")

    if not gen_glob:
        raise ValueError("No --gen-path provided and no run_dir resolved.")

    paths = [Path(p) for p in glob.glob(gen_glob)]
    if len(paths) == 0:
        pth = Path(gen_glob)
        if pth.exists():
            paths = [pth]
    if not paths:
        raise FileNotFoundError(f"No GEN files matched: {gen_glob}")

    metrics = [m.strip().lower() for m in str(args.metrics).split(",") if m.strip()]

    _ = score_run(
        gen_paths=paths,
        out_dir=out_dir,
        run_tag=args.run_tag,
        metrics=metrics,
        do_normalize=bool(args.do_normalize_for_similarity),
        ctqrs_path=args.ctqrs_path,
        rouge_processes=int(args.rouge_processes),
        ctqrs_processes=int(args.ctqrs_processes),
        sbert_model=args.sbert_model,
        sbert_device=args.sbert_device,
        sbert_batch_size=int(args.sbert_batch_size),
        overwrite=bool(args.overwrite),
        lock_stale_hours=float(args.lock_stale_hours),
        only_done_gen=(not bool(args.include_incomplete_gen)),
        export_csv=(not bool(args.no_export_csv)),
    )


def cmd_aggregate(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()

    # Prefer run-dir workflow unless a scored-glob is explicitly provided without run-dir
    use_run_dir = args.run_dir is not None or args.scored_glob is None

    if use_run_dir:
        run_dir_arg = args.run_dir if args.run_dir is not None else "last"
        run_dir, _ = resolve_run_dir(
            project_root=project_root,
            run_dir_arg=run_dir_arg,
            run_tag="aggregate",
            seed=0,
            create=True,
        )
        scored_glob = args.scored_glob if args.scored_glob else str(run_dir / "scored" / "scored_*.jsonl.gz")
        out_csv = Path(args.out_csv).resolve() if args.out_csv else (run_dir / "summary" / "INTEGRATED_summary.csv")
    else:
        scored_glob = args.scored_glob
        out_csv = Path(args.out_csv).resolve() if args.out_csv else (project_root / RESULTS_DIR_REL / "summary" / "INTEGRATED_summary.csv")

    groupby = [c.strip() for c in str(args.groupby).split(",") if c.strip()]
    aggregate_run(
        scored_glob=scored_glob,
        out_csv=out_csv,
        groupby=groupby,
        stat=args.stat,
        bootstrap_iters=int(args.bootstrap_iters),
        ci=float(args.ci),
        seed=int(args.seed),
    )


# ==============================
# Verify utilities (GEN artifacts)
# ==============================

def _jsonl_gz_stats(
    path: Path,
    *,
    sample_n: int = 3,
    full: bool = False,
) -> Dict[str, Any]:
    samples: List[Dict[str, Any]] = []

    if not full:
        n = 0
        try:
            for r in jsonl_gz_iter(path):
                if len(samples) < int(sample_n):
                    samples.append(r)
                n += 1
                if len(samples) >= int(sample_n) and n >= int(sample_n):
                    break
        except Exception:
            pass
        return {"row_count": None, "content_sha1": None, "samples": samples, "means": None}

    sha1 = hashlib.sha1()
    n = 0

    cols = [
        "PromptTokens",
        "InputTokens",
        "ExamplesTokens",
        "EffectiveK",
        "SecFilled",
        "UAHE_per_1kTok",
        "IUHE_per_1kTok",
        "ContextSupportRate",
        "ContextUnattributedRate",
    ]
    sums = {c: 0.0 for c in cols}
    counts = {c: 0 for c in cols}
    truncated_cnt = 0

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sha1.update((line + "\n").encode("utf-8"))
            n += 1
            if len(samples) < int(sample_n):
                try:
                    samples.append(json.loads(line))
                except Exception:
                    samples.append({"__raw__": line[:200]})

            try:
                r = json.loads(line)
            except Exception:
                continue

            if bool(r.get("InputTruncated", False)):
                truncated_cnt += 1
            for c in cols:
                v = r.get(c, None)
                if v is None:
                    continue
                try:
                    fv = float(v)
                    if fv == fv:
                        sums[c] += fv
                        counts[c] += 1
                except Exception:
                    continue

    means = {}
    for c in cols:
        means[c] = (sums[c] / counts[c]) if counts[c] > 0 else float("nan")

    means["InputTruncatedRate"] = (float(truncated_cnt) / float(n)) if n > 0 else float("nan")

    return {"row_count": int(n), "content_sha1": sha1.hexdigest(), "samples": samples, "means": means}


def verify_gen_run(
    *,
    project_root: Path,
    run_dir: Path,
    max_files: int = 0,
    sample_n: int = 3,
    full: bool = False,
) -> None:
    run_dir = Path(run_dir).resolve()
    gen_dir = run_dir / "gen"
    if not gen_dir.exists():
        raise FileNotFoundError(f"GEN dir not found: {gen_dir}")

    files = sorted(gen_dir.glob("gen_*.jsonl.gz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if max_files and int(max_files) > 0:
        files = files[: int(max_files)]

    if not files:
        print(f"[VERIFY] No GEN artifacts found under: {gen_dir}")
        return

    print(f"[VERIFY] run_dir={run_dir}")
    print(f"[VERIFY] gen_files={len(files)} (full={full}, sample_n={sample_n})")

    for p in files:
        meta_p = meta_path_for(p)
        done_p = done_marker_path(p)
        lock_p = lock_path_for(p)
        tmp_p = tmp_path_for(p)

        meta = load_meta(meta_p) or {}
        cond_id = meta.get("condition_id", None)
        cond_cfg = meta.get("condition_config", {}) if isinstance(meta.get("condition_config", {}), dict) else {}
        model = cond_cfg.get("model_key", None)
        expected = meta.get("expected_rows", None)
        saved_n = meta.get("row_count", None)
        saved_sha1 = meta.get("content_sha1", None)

        print("\n" + "-" * 80)
        print(f"[VERIFY] {p.name}")
        print(f"  done={done_p.exists()}  lock={lock_p.exists()}  tmp={tmp_p.exists()}  meta={meta_p.exists()}")
        print(f"  model={model}  condition_id={cond_id}")
        print(f"  expected_rows(meta)={expected}  row_count(meta)={saved_n}  sha1(meta)={saved_sha1}")

        stats = _jsonl_gz_stats(p, sample_n=int(sample_n), full=bool(full))
        if full:
            print(f"  row_count(actual)={stats['row_count']}  sha1(actual)={stats['content_sha1']}")
            if saved_n is not None and int(saved_n) != int(stats["row_count"]):
                print("  [WARN] meta row_count != actual row_count")
            if saved_sha1 and stats["content_sha1"] and str(saved_sha1) != str(stats["content_sha1"]):
                print("  [WARN] meta sha1 != actual sha1")
            if stats.get("means"):
                m = stats["means"]
                print(
                    "  means: "
                    + " | ".join(
                        [
                            f"PromptTokens={m.get('PromptTokens', float('nan')):.1f}",
                            f"InputTokens={m.get('InputTokens', float('nan')):.1f}",
                            f"ExamplesTokens={m.get('ExamplesTokens', float('nan')):.1f}",
                            f"EffectiveK={m.get('EffectiveK', float('nan')):.2f}",
                            f"SecFilled={m.get('SecFilled', float('nan')):.3f}",
                            f"UAHE/1kTok={m.get('UAHE_per_1kTok', float('nan')):.3f}",
                            f"CtxSupport={m.get('ContextSupportRate', float('nan')):.3f}",
                            f"InputTruncRate={m.get('InputTruncatedRate', float('nan')):.2%}",
                        ]
                    )
                )

        for i, r in enumerate(stats.get("samples", [])[: int(sample_n)]):
            try:
                rid = r.get("row_id", "NA")
                ek = r.get("EffectiveK", "NA")
                sp = r.get("PromptTokens", "NA")
                sec = r.get("SecFilled", "NA")
                print(f"  sample[{i}] row_id={rid} EffectiveK={ek} PromptTokens={sp} SecFilled={sec}")
            except Exception:
                print(f"  sample[{i}] (unparsed)")


def cmd_verify_gen(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    run_dir, _ = resolve_run_dir(
        project_root=project_root,
        run_dir_arg=args.run_dir,
        run_tag="verify",
        seed=0,
        create=True,
    )
    verify_gen_run(
        project_root=project_root,
        run_dir=run_dir,
        max_files=int(args.max_files),
        sample_n=int(args.sample_n),
        full=bool(args.full),
    )


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    if args.command == "data-prepare":
        cmd_data_prepare(args)
    elif args.command == "train-sft":
        cmd_train_sft(args)
    elif args.command == "mine-dpo-pairs":
        cmd_mine_dpo_pairs(args)
    elif args.command == "train-dpo":
        cmd_train_dpo(args)
    elif args.command == "retrieval-build":
        cmd_retrieval_build(args)
    elif args.command == "retrieval-cache":
        cmd_retrieval_cache(args)
    elif args.command == "gen":
        cmd_gen(args)
    elif args.command == "gen-suite":
        cmd_gen_suite(args)
    elif args.command == "score":
        cmd_score(args)
    elif args.command == "aggregate":
        cmd_aggregate(args)
    elif args.command == "verify-gen":
        cmd_verify_gen(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()