import os
import json
import random
import glob
from pathlib import Path
import shutil

# -----------------------------------------
# 🎯 목표 페어 개수 (Hard 프로필의 개수에 맞추세요. 예: 1000)
TARGET_N = 800  # <--- 이 숫자를 실제 가장 적은 파일의 개수로 수정하세요!
# -----------------------------------------

SEED = 42
random.seed(SEED)

SOURCE_DIR = Path("data/dpo_pairs_remine")
TARGET_DIR = Path("data/dpo_pairs")

# 기존 폴더가 있다면 백업
if TARGET_DIR.exists():
    shutil.move(str(TARGET_DIR), f"{TARGET_DIR}_backup")

TARGET_DIR.mkdir(parents=True, exist_ok=True)

# 모든 jsonl 파일 찾기
jsonl_files = glob.glob(f"{SOURCE_DIR}/**/*.jsonl", recursive=True)

for file_path in jsonl_files:
    src_p = Path(file_path)
    
    # 원본 데이터 읽기
    with open(src_p, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # 셔플 및 다운샘플링
    random.shuffle(lines)
    sampled_lines = lines[:TARGET_N]
    
    # 저장할 타겟 경로 만들기 (디렉토리 구조 유지)
    rel_path = src_p.relative_to(SOURCE_DIR)
    tgt_p = TARGET_DIR / rel_path
    tgt_p.parent.mkdir(parents=True, exist_ok=True)
    
    # 샘플링된 jsonl 저장
    with open(tgt_p, 'w', encoding='utf-8') as f:
        f.write("\n".join(sampled_lines) + "\n")
        
    print(f"✅ [Downsampled] {tgt_p.name} : {len(lines)} -> {len(sampled_lines)} pairs")