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
PRECISION="auto"
DECODE="greedy"
MAX_NEW_TOKENS=1024
MAX_PROMPT_TOKENS=4096
MACRO_BATCH=128
MICRO_BATCH=32
INPUT_BATCH_SIZE=16 # 👈 새로 추가 (한 번에 8개의 프롬프트를 묶어서 처리)
AUTOTUNE_MB=0
THREADS=28
PREFETCH=8
REGEX_PROCS=0

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

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${PROJECT_ROOT}/results/runs/${TIMESTAMP}_${RUN_TAG}_seed${SEED}"
mkdir -p "${RUN_DIR}"

AUTO_FLAG=""
if [ "${AUTOTUNE_MB}" -eq 1 ]; then
    AUTO_FLAG="--autotune-micro-batch"
fi

echo "========================================================================"
echo " 🚀 EMSE 43-COMBINATIONS MASTER PIPELINE START"
echo " 📂 RUN_DIR : ${RUN_DIR}"
echo "========================================================================"
echo ""

# # =========================================================================
# # 🔄 [4] 1단계: Data Prepare & Indexing
# # =========================================================================
# echo "[STAGE 1] Data Prepare & Index Build"
# ${PYTHON} "${PIPELINE_PATH}" data-prepare \
#   --project-root "${PROJECT_ROOT}" --seed "${SEED}" --input-col "${INPUT_COL}" --gt-col "${GT_COL}" \
#   --split-mode "group" --group-col "group_id"

# ${PYTHON} "${PIPELINE_PATH}" retrieval-build \
#   --project-root "${PROJECT_ROOT}" --seed "${SEED}" --input-col "${INPUT_COL}" --retriever-type both \
#   --dense-emb-model "${DENSE_EMB_MODEL}" --dense-device "${DENSE_DEVICE}" --dense-batch-size "${DENSE_BATCH}" \
#   --lex-backend "${LEX_BACKEND}" --tfidf-ngram "${TFIDF_NGRAM}" --tfidf-max-features "${TFIDF_MAX_FEATURES}" \
#   --bm25-k1 "${BM25_K1}" --bm25-b "${BM25_B}"

# # =========================================================================
# # 🤖 [5] 2단계: SFT Training (기초 체력 확보)
# # =========================================================================
# echo "[STAGE 2] SFT Training on Clean Group-Split Train Set with lower LR (2e-4)..."
# ${PYTHON} "${PIPELINE_PATH}" train-sft \
#     --project-root "${PROJECT_ROOT}" \
#     --seed "${SEED}" \
#     --models "${ALL_MODELS[@]}" \
#     --input-col "${INPUT_COL}" \
#     --gt-col "${GT_COL}" \
#     --template "on" \
#     --lora-type "qlora" \
#     --lr 2e-4 \
#     --epochs 3 \
#     --batch-size 8 \
#     --grad-accum-steps 1 \
#     --lora-r 16 \
#     --lora-alpha 32 \
#     --lora-dropout 0.05 \
#     --max-seq-len 2048

# # =========================================================================
# # 🗄️ [6] 3단계: Build Retrieval Caches
# # =========================================================================
# echo "[STAGE 3] Building Retrieval Caches..."
# for RET_MODE in similar random; do
#     ${PYTHON} "${PIPELINE_PATH}" retrieval-cache \
#       --project-root "${PROJECT_ROOT}" --seed "${SEED}" --split test --input-col "${INPUT_COL}" \
#       --retriever-type dense --retriever-mode "${RET_MODE}" --k-values "1,2" \
#       --dense-emb-model "${DENSE_EMB_MODEL}" --dense-device "${DENSE_DEVICE}" --dense-batch-size "${DENSE_BATCH}"
# done

# ${PYTHON} "${PIPELINE_PATH}" retrieval-cache \
#   --project-root "${PROJECT_ROOT}" --seed "${SEED}" --split test --input-col "${INPUT_COL}" \
#   --retriever-type lexical --retriever-mode similar --k-values "1,2" \
#   --lex-backend "${LEX_BACKEND}" --tfidf-ngram "${TFIDF_NGRAM}" --tfidf-max-features "${TFIDF_MAX_FEATURES}"

# # DPO Train Cache (Group Exclusion)
# ${PYTHON} "${PIPELINE_PATH}" retrieval-cache \
#   --project-root "${PROJECT_ROOT}" --seed "${SEED}" --split train --input-col "${INPUT_COL}" \
#   --retriever-type dense --retriever-mode similar --k-values "1" \
#   --dense-emb-model "${DENSE_EMB_MODEL}" --dense-device "${DENSE_DEVICE}" --dense-batch-size "${DENSE_BATCH}" \
#   --leave-one-out


# =========================================================================
# ⛏️ [7] 4단계: DPO Mining & Training (정확히 8개 어댑터)
# =========================================================================
echo "[STAGE 4] DPO Mining & Training (Exactly 8 Adapters)"

# 헬퍼 함수
run_dpo() {
    local ALIGN=$1; local PROF=$2; shift 2; local TGT_MODELS=("$@")
    local RET_TYPE="none"; local RET_MODE="none"; local K_VAL=0
    if [ "$ALIGN" == "retrieval_aware" ]; then
        RET_TYPE="dense"; RET_MODE="similar"; K_VAL=1
    fi
    echo "  -> [DPO] Context: ${ALIGN} | Profile: ${PROF} | Models: ${TGT_MODELS[*]}"
    
    ${PYTHON} "${PIPELINE_PATH}" mine-dpo-pairs \
        --project-root "${PROJECT_ROOT}" --seed "${SEED}" --run-tag "${RUN_TAG}_${ALIGN}_${PROF}" \
        --models "${TGT_MODELS[@]}" --input-col "${INPUT_COL}" --gt-col "${GT_COL}" \
        --pref-profile "${PROF}" --align-context "${ALIGN}" \
        --retriever-type "${RET_TYPE}" --retriever-mode "${RET_MODE}" --k "${K_VAL}" --dense-emb-model "${DENSE_EMB_MODEL}" \
        --generator-adapter "sft" --num-candidates 4 --save-candidates-jsonl-gz --save-pref-components --enforce-uahe-order \
        --precision "${PRECISION}" --gen-max-new-tokens "${MAX_NEW_TOKENS}" --max-prompt-tokens "${MAX_PROMPT_TOKENS}" \
        --input-batch-size "${INPUT_BATCH_SIZE}" \
        --micro-batch "${MICRO_BATCH}" ${AUTO_FLAG}

    ${PYTHON} "${PIPELINE_PATH}" train-dpo \
        --project-root "${PROJECT_ROOT}" --seed "${SEED}" --run-tag "${RUN_TAG}_${ALIGN}_${PROF}" \
        --models "${TGT_MODELS[@]}" --input-col "${INPUT_COL}" --gt-col "${GT_COL}" \
        --pref-profile "${PROF}" --align-context "${ALIGN}" \
        --retriever-type "${RET_TYPE}" --retriever-mode "${RET_MODE}" --k "${K_VAL}" --dense-emb-model "${DENSE_EMB_MODEL}" \
        --max-prompt-tokens "${MAX_PROMPT_TOKENS}" --lora-type "qlora" --epochs 1 --lr 5e-5 --batch-size 8 --grad-accum-steps 1 --beta 0.1
}

# 1~3. Balanced (Free) -> 3개 모델
run_dpo "retrieval_free" "balanced" "${ALL_MODELS[@]}"

# 4~6. Balanced (Aware) -> 3개 모델
run_dpo "retrieval_aware" "balanced" "${ALL_MODELS[@]}"

# 7. Hard (Free) -> Qwen 전용
run_dpo "retrieval_free" "hard" "${QWEN_ONLY[@]}"

# 8. Structure (Free) -> Qwen 전용
run_dpo "retrieval_free" "structure" "${QWEN_ONLY[@]}"


# =========================================================================
# 🧪 [8] 5단계: GEN - 정확히 43개 시나리오 평가
# =========================================================================
echo "[STAGE 5] FULL GEN STAGE: Evaluating exactly 43 scenarios..."

# 헬퍼 함수 (타겟 모델만 인자로 받음)
run_gen() {
    local ADAPTER=$1; local ALIGN_CTX=$2; local PREF_PROF=$3; local RET_TYPE=$4; local RET_MODE=$5; local K=$6; local TEMPLATE=$7; local RET_MASK=$8; local SUFFIX=$9; shift 9; local TGT_MODELS=("$@")
    echo "  -> [GEN] Scenario: ${SUFFIX} | Models: ${TGT_MODELS[*]}"
    
    ${PYTHON} "${PIPELINE_PATH}" gen \
        --project-root "${PROJECT_ROOT}" --seed "${SEED}" --run-dir "${RUN_DIR}" --run-tag "${RUN_TAG}_${SUFFIX}" \
        --models "${TGT_MODELS[@]}" --input-col "${INPUT_COL}" --gt-col "${GT_COL}" \
        --adapter "${ADAPTER}" --align-context "${ALIGN_CTX}" --pref-profile "${PREF_PROF}" --template "${TEMPLATE}" \
        --retriever-type "${RET_TYPE}" --retriever-mode "${RET_MODE}" --k "${K}" --retrieval-mask "${RET_MASK}" \
        --dense-emb-model "${DENSE_EMB_MODEL}" --lex-backend "${LEX_BACKEND}" \
        --precision "${PRECISION}" --decode "${DECODE}" --max-new-tokens "${MAX_NEW_TOKENS}" --max-prompt-tokens "${MAX_PROMPT_TOKENS}" \
        --macro-batch "${MACRO_BATCH}" --micro-batch "${MICRO_BATCH}" ${AUTO_FLAG} \
        --tokenizer-threads "${THREADS}" --pipeline-prefetch "${PREFETCH}" --regex-processes "${REGEX_PROCS}"
}

# --- Base (6개) ---
run_gen base none none none none 0 on none "base_none_k0" "${ALL_MODELS[@]}"
run_gen base none none dense similar 1 on none "base_dense_sim_k1" "${ALL_MODELS[@]}"

# --- SFT (21개) ---
run_gen sft none none none none 0 on none "sft_none_k0" "${ALL_MODELS[@]}"
run_gen sft none none dense similar 1 on none "sft_dense_sim_k1" "${ALL_MODELS[@]}"
run_gen sft none none dense similar 2 on none "sft_dense_sim_k2" "${ALL_MODELS[@]}"
run_gen sft none none dense random 1 on none "sft_dense_rand_k1" "${ALL_MODELS[@]}"
run_gen sft none none lexical similar 1 on none "sft_lex_sim_k1" "${ALL_MODELS[@]}"
run_gen sft none none dense similar 1 off none "sft_dense_sim_k1_off" "${ALL_MODELS[@]}"
run_gen sft none none dense similar 1 on hardmask "sft_dense_sim_k1_hardmask" "${ALL_MODELS[@]}"

# --- DPO 공통 (12개) ---
run_gen dpo retrieval_free balanced none none 0 on none "dpo_free_bal_none_k0" "${ALL_MODELS[@]}"
run_gen dpo retrieval_free balanced dense similar 1 on none "dpo_free_bal_dense_sim_k1" "${ALL_MODELS[@]}"
run_gen dpo retrieval_aware balanced none none 0 on none "dpo_aware_bal_none_k0" "${ALL_MODELS[@]}"
run_gen dpo retrieval_aware balanced dense similar 1 on none "dpo_aware_bal_dense_sim_k1" "${ALL_MODELS[@]}"

# --- DPO Qwen 전용 (4개) ---
run_gen dpo retrieval_free hard none none 0 on none "dpo_free_hard_none_k0" "${QWEN_ONLY[@]}"
run_gen dpo retrieval_free hard dense similar 1 on none "dpo_free_hard_dense_sim_k1" "${QWEN_ONLY[@]}"
run_gen dpo retrieval_free structure none none 0 on none "dpo_free_struct_none_k0" "${QWEN_ONLY[@]}"
run_gen dpo retrieval_free structure dense similar 1 on none "dpo_free_struct_dense_sim_k1" "${QWEN_ONLY[@]}"


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