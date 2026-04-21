#!/usr/bin/env python3
"""Vocabulary 사용 분석 — 어떤 token이 실제로 쓰이는지
목표: 32000 vocab 중 사용 안 되는 것 제거해 임베딩 축소
"""
import json, os, sys
from collections import Counter
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

tok = AutoTokenizer.from_pretrained('tokenizer/')
print(f"Vocab size: {len(tok)}")

# 모든 데이터 로드
datasets = {
    'train_v34': 'data/train_final_v34.json',
    'train_v68': 'data/train_final_v68.json',
    'test_suite': 'data/test_suite.json',
    'gt_known': 'data/gt_known_scenarios.json',
    'gt_unknown': 'data/gt_unknown_scenarios.json',
    'ke_train': 'data/koelectra_converted_train.json',
    'ke_val': 'data/koelectra_converted_val.json',
}

all_tokens = Counter()
for name, path in datasets.items():
    if not os.path.exists(path):
        print(f"  {name}: SKIP (없음)")
        continue
    d = json.load(open(path))
    for item in d:
        utt = item.get('utterance', '')
        if not utt:
            continue
        ids = tok.encode(utt, add_special_tokens=True, truncation=True, max_length=32)
        for tid in ids:
            all_tokens[tid] += 1
    print(f"  {name}: {len(d)}개")

print(f"\n=== Vocabulary 사용 통계 ===")
print(f"총 unique tokens used: {len(all_tokens)} / {len(tok)} = {len(all_tokens)/len(tok)*100:.1f}%")
print(f"사용 안 됨:               {len(tok) - len(all_tokens)}")

# 사용 빈도 분포
counts = sorted(all_tokens.values(), reverse=True)
print(f"\n=== 빈도 분포 ===")
print(f"  Top-1:        {counts[0]}")
print(f"  Top-10:       {counts[9] if len(counts) > 9 else 'N/A'}")
print(f"  Top-100:      {counts[99] if len(counts) > 99 else 'N/A'}")
print(f"  Top-1000:     {counts[999] if len(counts) > 999 else 'N/A'}")
print(f"  Median:       {counts[len(counts)//2]}")

# 1회만 나오는 token
used_1x = sum(1 for c in counts if c == 1)
used_2x_plus = sum(1 for c in counts if c >= 2)
used_10x_plus = sum(1 for c in counts if c >= 10)
print(f"\n=== 빈도별 ===")
print(f"  1회만 사용:      {used_1x} tokens")
print(f"  2회 이상:        {used_2x_plus} tokens")
print(f"  10회 이상:       {used_10x_plus} tokens")

# 실질 vocab 제안: min_freq=2
print(f"\n=== Vocab Pruning 시나리오 ===")
for min_freq in [1, 2, 5, 10]:
    keep = sum(1 for c in counts if c >= min_freq)
    # 특수 토큰 (PAD, CLS, SEP, MASK) 보존 필수
    essential = 5  # PAD(0), CLS, SEP, MASK, UNK
    total_keep = keep + essential
    old_emb_mb = 32000 * 768 * 4 / 1048576
    new_emb_mb = total_keep * 768 * 4 / 1048576
    print(f"  min_freq={min_freq}: keep {total_keep}/32000, 임베딩 {old_emb_mb:.0f}MB → {new_emb_mb:.1f}MB ({(1-new_emb_mb/old_emb_mb)*100:.0f}% 감소)")

# 저장
with open('data/vocab_usage.json', 'w', encoding='utf-8') as f:
    json.dump({
        'total_vocab': len(tok),
        'used': len(all_tokens),
        'token_counts': dict(all_tokens),
    }, f, indent=2)
print(f"\n✓ data/vocab_usage.json 저장")
