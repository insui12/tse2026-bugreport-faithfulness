#!/usr/bin/env bash

# =========================================================================
# 🚨 [1] 병렬 처리 충돌 방지 및 안전성 강제 설정 (OOM 및 Hang 방지)
# =========================================================================
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

set -euo pipefail
IFS=$'\n\t'

# =========================================================================
# 📂 [2] 프로젝트 환경 및 변수 초기화
# =========================================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
PIPELINE_PATH="${PROJECT_ROOT}/code/bugreport_pipeline_tse.py"

SEED=42
RUN_TAG="emse_re_eval_final"

# 🎯 모델 그룹 정의
ALL_MODELS=("qwen2.5-7b" "llama-3.2-3b" "mistral-7b-v0.3")
QWEN_ONLY=("qwen2.5-7b")

# 데이터 및 검색 설정
INPUT_COL="NEW_llama_output"
GT_COL="text"
DENSE_EMB_MODEL="BAAI/bge-large-en-v1.5"
DENSE_DEVICE="cuda"
DENSE_BATCH=256
LEX_BACKEND="tfidf"
TFIDF_NGRAM=2
TFIDF_MAX_FEATURES=200000
BM25_K1=1.5
BM25_B=0.75

# Generation & Scoring 설정
PRECISION="bf16"
DECODE="greedy"
MAX_NEW_TOKENS=1024
MAX_PROMPT_TOKENS=4096
MACRO_BATCH=128
MICRO_BATCH=128
INPUT_BATCH_SIZE=32
AUTOTUNE_MB=0
THREADS=28
PREFETCH=8
REGEX_PROCS=28

METRICS="rouge,sbert,ctqrs"
CTQRS_PATH="${PROJECT_ROOT}/code/evaluation/perfect_ctqrs.py"
SBERT_MODEL="sentence-transformers/all-mpnet-base-v2"
SBERT_DEVICE="cuda"
SBERT_BATCH=512
ROUGE_PROCS=15
CTQRS_PROCS=15

BOOT_ITERS=10000
CI=0.95
GROUPBY="model,adapter_type,pref_profile,align_context,retriever_type,retriever_mode,k,template,retrieval_mask,seed"

# ⚠️ 지정된 RUN_DIR 사용 (Qwen Base/SFT가 저장된 기존 디렉토리 이어서 사용)
RUN_DIR="${PROJECT_ROOT}/results/runs/20260228_074605_emse_re_eval_final_seed42" 
mkdir -p "${RUN_DIR}"

AUTO_FLAG=""
if [ "${AUTOTUNE_MB}" -eq 1 ]; then
    AUTO_FLAG="--autotune-micro-batch"
fi

echo "========================================================================"
echo " 🚀 EMSE 43-COMBINATIONS MASTER PIPELINE RESUME"
echo " 📂 RUN_DIR : ${RUN_DIR}"
echo "========================================================================"
echo ""

# =========================================================================
# ⛏️ [7] 4단계: DPO Training ONLY (재채굴된 쌍을 바탕으로 학습만 수행)
# =========================================================================
echo "[STAGE 4] DPO Training ONLY (Skipping Mining as remining is done)"

# 헬퍼 함수: mine-dpo-pairs 제거, train-dpo만 남김
train_dpo_only() {
    local ALIGN=$1; local PROF=$2; shift 2; local TGT_MODELS=("$@")
    local RET_TYPE="none"; local RET_MODE="none"; local K_VAL=0
    if [ "$ALIGN" == "retrieval_aware" ]; then
        RET_TYPE="dense"; RET_MODE="similar"; K_VAL=1
    fi
    echo "  -> [DPO TRAIN] Context: ${ALIGN} | Profile: ${PROF} | Models: ${TGT_MODELS[*]}"
    
    # 🚨 주의: 파이썬 파이프라인이 remine된 파일 경로를 자동으로 못 찾는다면,
    # data/dpo_pairs_remine 폴더의 파일들을 기존 data/dpo_pairs 폴더로 덮어씌워 주셔야 합니다.
    ${PYTHON} "${PIPELINE_PATH}" train-dpo \
        --project-root "${PROJECT_ROOT}" \
        --seed "${SEED}" \
        --run-tag "${RUN_TAG}_${ALIGN}_${PROF}" \
        --models "${TGT_MODELS[@]}" \
        --input-col "${INPUT_COL}" \
        --gt-col "${GT_COL}" \
        --pref-profile "${PROF}" \
        --align-context "${ALIGN}" \
        --retriever-type "${RET_TYPE}" \
        --retriever-mode "${RET_MODE}" \
        --k "${K_VAL}" \
        --dense-emb-model "${DENSE_EMB_MODEL}" \
        --max-prompt-tokens "${MAX_PROMPT_TOKENS}" \
        --lora-type "qlora" \
        --epochs 1 \
        --lr 5e-5 \
        --batch-size 1 \
        --grad-accum-steps 8 \
        --beta 0.1
}

# 1~3. Balanced (Free) -> 3개 모델
train_dpo_only "retrieval_free" "balanced" "${ALL_MODELS[@]}"

# 4~6. Balanced (Aware) -> 3개 모델
train_dpo_only "retrieval_aware" "balanced" "${ALL_MODELS[@]}"

# 7. Hard (Free) -> Qwen 전용
train_dpo_only "retrieval_free" "hard" "${QWEN_ONLY[@]}"

# 8. Structure (Free) -> Qwen 전용
train_dpo_only "retrieval_free" "structure" "${QWEN_ONLY[@]}"


# =========================================================================
# 🧪 [8] 5단계: GEN - 모델별 일괄 처리 (완료된 부분 제외)
# =========================================================================
echo "[STAGE 5] FULL GEN STAGE: Evaluating scenarios (Model by Model)..."

run_single_gen() {
    local ADAPTER=$1; local ALIGN_CTX=$2; local PREF_PROF=$3; local RET_TYPE=$4; local RET_MODE=$5; local K=$6; local TEMPLATE=$7; local RET_MASK=$8; local SUFFIX=$9; local TGT_MODEL=${10}
    echo "  -> [GEN] Model: ${TGT_MODEL} | Scenario: ${SUFFIX}"
    
    ${PYTHON} "${PIPELINE_PATH}" gen \
        --project-root "${PROJECT_ROOT}" --seed "${SEED}" --run-dir "${RUN_DIR}" --run-tag "${RUN_TAG}_${SUFFIX}" \
        --models "${TGT_MODEL}" --input-col "${INPUT_COL}" --gt-col "${GT_COL}" \
        --adapter "${ADAPTER}" --align-context "${ALIGN_CTX}" --pref-profile "${PREF_PROF}" --template "${TEMPLATE}" \
        --retriever-type "${RET_TYPE}" --retriever-mode "${RET_MODE}" --k "${K}" --retrieval-mask "${RET_MASK}" \
        --dense-emb-model "${DENSE_EMB_MODEL}" --lex-backend "${LEX_BACKEND}" \
        --precision "${PRECISION}" --decode "${DECODE}" --max-new-tokens "${MAX_NEW_TOKENS}" --max-prompt-tokens "${MAX_PROMPT_TOKENS}" \
        --macro-batch "${MACRO_BATCH}" --micro-batch "${MICRO_BATCH}" ${AUTO_FLAG} \
        --tokenizer-threads "${THREADS}" --pipeline-prefetch "${PREFETCH}" --regex-processes "${REGEX_PROCS}"
}

run_all_common_gens_for_model() {
    local MODEL=$1
    echo "============================================================"
    echo " 🚀 Starting ALL GEN tasks for model: ${MODEL}"
    echo "============================================================"

    # --- Base (2개) ---
    run_single_gen base none none none none 0 on none "base_none_k0" "${MODEL}"
    run_single_gen base none none dense similar 1 on none "base_dense_sim_k1" "${MODEL}"

    # --- SFT (7개) ---
    run_single_gen sft none none none none 0 on none "sft_none_k0" "${MODEL}"
    run_single_gen sft none none dense similar 1 on none "sft_dense_sim_k1" "${MODEL}"
    run_single_gen sft none none dense similar 2 on none "sft_dense_sim_k2" "${MODEL}"
    run_single_gen sft none none dense random 1 on none "sft_dense_rand_k1" "${MODEL}"
    run_single_gen sft none none lexical similar 1 on none "sft_lex_sim_k1" "${MODEL}"
    run_single_gen sft none none dense similar 1 off none "sft_dense_sim_k1_off" "${MODEL}"
    run_single_gen sft none none dense similar 1 on hardmask "sft_dense_sim_k1_hardmask" "${MODEL}"

    # --- DPO 공통 (4개) ---
    run_single_gen dpo retrieval_free balanced none none 0 on none "dpo_free_bal_none_k0" "${MODEL}"
    run_single_gen dpo retrieval_free balanced dense similar 1 on none "dpo_free_bal_dense_sim_k1" "${MODEL}"
    run_single_gen dpo retrieval_aware balanced none none 0 on none "dpo_aware_bal_none_k0" "${MODEL}"
    run_single_gen dpo retrieval_aware balanced dense similar 1 on none "dpo_aware_bal_dense_sim_k1" "${MODEL}"
}

# -------------------------------------------------------------------------
# 실행 순서: Qwen -> Mistral -> LLaMA
# -------------------------------------------------------------------------

# 1. Qwen (Base 2개, SFT 7개는 완료되었으므로 주석 처리. 새롭게 학습된 DPO 8개만 실행)
echo "============================================================"
echo " 🚀 Resuming GEN tasks for model: qwen2.5-7b"
echo "============================================================"
# run_single_gen base none none none none 0 on none "base_none_k0" "qwen2.5-7b"
# run_single_gen base none none dense similar 1 on none "base_dense_sim_k1" "qwen2.5-7b"
# run_single_gen sft none none none none 0 on none "sft_none_k0" "qwen2.5-7b"
# run_single_gen sft none none dense similar 1 on none "sft_dense_sim_k1" "qwen2.5-7b"
# run_single_gen sft none none dense similar 2 on none "sft_dense_sim_k2" "qwen2.5-7b"
# run_single_gen sft none none dense random 1 on none "sft_dense_rand_k1" "qwen2.5-7b"
# run_single_gen sft none none lexical similar 1 on none "sft_lex_sim_k1" "qwen2.5-7b"
# run_single_gen sft none none dense similar 1 off none "sft_dense_sim_k1_off" "qwen2.5-7b"
# run_single_gen sft none none dense similar 1 on hardmask "sft_dense_sim_k1_hardmask" "qwen2.5-7b"

# Qwen DPO 공통 시나리오 (4개)
run_single_gen dpo retrieval_free balanced none none 0 on none "dpo_free_bal_none_k0" "qwen2.5-7b"
run_single_gen dpo retrieval_free balanced dense similar 1 on none "dpo_free_bal_dense_sim_k1" "qwen2.5-7b"
run_single_gen dpo retrieval_aware balanced none none 0 on none "dpo_aware_bal_none_k0" "qwen2.5-7b"
run_single_gen dpo retrieval_aware balanced dense similar 1 on none "dpo_aware_bal_dense_sim_k1" "qwen2.5-7b"

# Qwen 전용 DPO 시나리오 (4개)
run_single_gen dpo retrieval_free hard none none 0 on none "dpo_free_hard_none_k0" "qwen2.5-7b"
run_single_gen dpo retrieval_free hard dense similar 1 on none "dpo_free_hard_dense_sim_k1" "qwen2.5-7b"
run_single_gen dpo retrieval_free structure none none 0 on none "dpo_free_struct_none_k0" "qwen2.5-7b"
run_single_gen dpo retrieval_free structure dense similar 1 on none "dpo_free_struct_dense_sim_k1" "qwen2.5-7b"

# 2. Mistral (전체 15개 시나리오 실행)
run_all_common_gens_for_model "mistral-7b-v0.3"

# 3. LLaMA (전체 15개 시나리오 실행)
run_all_common_gens_for_model "llama-3.2-3b"


# =========================================================================
# 📊 [9] 6단계: SCORE & AGGREGATE
# =========================================================================
echo "[STAGE 6] Scoring all 43 scenarios..."
${PYTHON} "${PIPELINE_PATH}" score \
    --project-root "${PROJECT_ROOT}" --run-dir "${RUN_DIR}" --metrics "${METRICS}" \
    --ctqrs-path "${CTQRS_PATH}" --sbert-model "${SBERT_MODEL}" --sbert-device "${SBERT_DEVICE}" --sbert-batch-size "${SBERT_BATCH}" \
    --rouge-processes "${ROUGE_PROCS}" --ctqrs-processes "${CTQRS_PROCS}" --do-normalize-for-similarity

echo "[STAGE 7] Aggregating results into INTEGRATED_summary.csv..."
${PYTHON} "${PIPELINE_PATH}" aggregate \
    --project-root "${PROJECT_ROOT}" --run-dir "${RUN_DIR}" --groupby "${GROUPBY}" \
    --stat "bootstrap_ci" --bootstrap-iters "${BOOT_ITERS}" --ci "${CI}" --seed "${SEED}"

echo "========================================================================"
echo " 🎉 MASTER PIPELINE COMPLETED SUCCESSFULLY! 🎉"
echo " 📁 Final Result Check: ${RUN_DIR}/summary/INTEGRATED_summary.csv"
echo "========================================================================"