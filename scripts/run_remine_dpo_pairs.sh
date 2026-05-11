#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

PYTHON="${PYTHON:-python3}"
REMINE_PY="${REMINE_PY:-$PROJECT_ROOT/code/remine_dpo_pairs.py}"

if [[ ! -f "$REMINE_PY" ]]; then
  echo "[ERROR] remine script not found: $REMINE_PY" >&2
  exit 1
fi

# =========================================================================
# ⚙️ 공통 환경 설정
# =========================================================================
PAIRS_PER_PROMPT="${PAIRS_PER_PROMPT:-2}"
SAVE_PREF_COMPONENTS="${SAVE_PREF_COMPONENTS:-1}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/data/dpo_pairs_remine}"

OVERWRITE="${OVERWRITE:-1}" # 일괄 덮어쓰기 허용 (필요시 0으로 변경)
DRY_RUN="${DRY_RUN:-0}"
NO_CSV="${NO_CSV:-0}"
ALLOW_NONCONTIGUOUS="${ALLOW_NONCONTIGUOUS:-0}"
LOCK_STALE_HOURS="${LOCK_STALE_HOURS:-24}"

LOG_DIR="$PROJECT_ROOT/results/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG_FILE="$LOG_DIR/remine_master_${TS}.log"

echo "========================================================================" | tee -a "$MAIN_LOG_FILE"
echo " 🚀 EMSE DPO Remining Master Pipeline" | tee -a "$MAIN_LOG_FILE"
echo " 🛡️ Strategy: Universal Quality Gate + Profile-Specific Margin" | tee -a "$MAIN_LOG_FILE"
echo "========================================================================" | tee -a "$MAIN_LOG_FILE"

# 3개의 프로필 순회 (이 루프로 총 8개의 파일이 알아서 매칭됩니다)
PROFILES=("balanced" "structure" "hard")

# 파일이 없을 경우 에러 방지
shopt -s nullglob

for PROF in "${PROFILES[@]}"; do
    echo "" | tee -a "$MAIN_LOG_FILE"
    echo "------------------------------------------------------------------------" | tee -a "$MAIN_LOG_FILE"
    echo " 🎯 Processing Profile: ${PROF^^}" | tee -a "$MAIN_LOG_FILE"
    echo "------------------------------------------------------------------------" | tee -a "$MAIN_LOG_FILE"

    # 해당 프로필이 이름에 포함된 모든 candidates_full 파일 찾기
    CAND_FILES=("$PROJECT_ROOT"/results/cache/dpo_candidates/*_"${PROF}"_*_candidates_full.jsonl.gz)
    
    if [[ ${#CAND_FILES[@]} -eq 0 ]]; then
        echo "[WARN] No candidate files found for profile '${PROF}'. Skipping..." | tee -a "$MAIN_LOG_FILE"
        continue
    fi

    echo "[INFO] Found ${#CAND_FILES[@]} candidate files for '${PROF}'." | tee -a "$MAIN_LOG_FILE"

    # =========================================================================
    # 🛡️ EMSE 논문 방어용 하이퍼파라미터 분기 (핵심 로직)
    # =========================================================================
    
    # 1. 공통 게이트 (Universal Gate): 모든 프로필에 동일하게 적용
    # - PrefScore 절대값 컷오프는 0.0으로 무력화하고, "정보 반영률(CondFilled)"이라는 절대 지표로 필터링
    QUALITY_FLOOR="0.0"
    MIN_COND_FILLED_RATE="0.60" 

    # 2. 프로필별 차등 설정 (Intent 기반 분기)
    if [[ "$PROF" == "balanced" ]]; then
        MARGIN="0.05"
        UAHE_CAP="-1.0"            # 환각 캡 미적용 (수식의 패널티로만 제어)
        ENFORCE_UAHE_ORDER="0"
    elif [[ "$PROF" == "structure" ]]; then
        MARGIN="0.05"
        UAHE_CAP="-1.0"            # 환각 캡 미적용
        ENFORCE_UAHE_ORDER="0"
    elif [[ "$PROF" == "hard" ]]; then
        MARGIN="0.03"              # 점수 스케일이 낮으므로 마진 축소
        UAHE_CAP="0.0"             # 🚨 환각 0개 무조건 강제 (Hard의 본질)
        ENFORCE_UAHE_ORDER="1"     # Chosen의 환각 <= Rejected 환각 강제
    else
        echo "[ERROR] Unknown profile: $PROF"
        continue
    fi
    # =========================================================================

    RUN_TAG="${PROF}_qf${QUALITY_FLOOR}_m${MARGIN}_ua${UAHE_CAP}_mc${MIN_COND_FILLED_RATE}_pp${PAIRS_PER_PROMPT}_eo${ENFORCE_UAHE_ORDER}"
    LOG_FILE="$LOG_DIR/remine_${TS}_${RUN_TAG}.log"

    ARGS=(
        "--project-root" "$PROJECT_ROOT"
        "--out-root" "$OUT_ROOT"
        "--run-tag" "$RUN_TAG"
        "--quality-floor" "$QUALITY_FLOOR"
        "--margin" "$MARGIN"
        "--pairs-per-prompt" "$PAIRS_PER_PROMPT"
        "--uahe-cap" "$UAHE_CAP"
        "--min-cond-filled-rate" "$MIN_COND_FILLED_RATE"
        "--lock-stale-hours" "$LOCK_STALE_HOURS"
    )

    if [[ "$ENFORCE_UAHE_ORDER" == "1" ]]; then
        ARGS+=("--enforce-uahe-order")
    fi
    if [[ "$SAVE_PREF_COMPONENTS" == "1" ]]; then
        ARGS+=("--save-pref-components")
    fi
    if [[ "$OVERWRITE" == "1" ]]; then
        ARGS+=("--overwrite")
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
        ARGS+=("--dry-run")
    fi
    if [[ "$NO_CSV" == "1" ]]; then
        ARGS+=("--no-csv")
    fi
    if [[ "$ALLOW_NONCONTIGUOUS" == "1" ]]; then
        ARGS+=("--allow-noncontiguous")
    fi

    {
        echo "[RUN] Policy: Universal Gate (QualityFloor=$QUALITY_FLOOR, CondFilled>=$MIN_COND_FILLED_RATE)"
        echo "[RUN] Policy: Profile Specifics (Margin=$MARGIN, UAHE_Cap=$UAHE_CAP, Enforce_UAHE_Order=$ENFORCE_UAHE_ORDER)"
    } | tee -a "$MAIN_LOG_FILE" "$LOG_FILE"

    export PYTHONUNBUFFERED=1

    echo "[INFO] Running remine_dpo_pairs.py for ${PROF}..." | tee -a "$MAIN_LOG_FILE"
    
    # Python 스크립트 실행 (로그는 파일에 저장하고 화면에는 핵심 내용만 출력)
    "$PYTHON" "$REMINE_PY" "${ARGS[@]}" --candidates-full "${CAND_FILES[@]}" 2>&1 | tee -a "$LOG_FILE"

done

echo "========================================================================" | tee -a "$MAIN_LOG_FILE"
echo " 🎉 ALL 8 ADAPTER SCENARIOS REMINED SUCCESSFULLY! Check $OUT_ROOT" | tee -a "$MAIN_LOG_FILE"
echo "========================================================================" | tee -a "$MAIN_LOG_FILE"