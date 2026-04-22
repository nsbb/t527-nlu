#!/usr/bin/env python3
"""v2 vs 르엘 AI기대응답 similarity 평가.

지표:
- exact: 완전 일치
- word_overlap: 단어 Jaccard similarity
- semantic_close: key noun/verb가 일치하면 close
"""
import os, sys, csv, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deployment_pipeline_v2 import DeploymentPipelineV2


def word_tokens(s):
    # Korean words + digits
    s = re.sub(r'[^\w\s%도분시간초단계]', ' ', s)
    tokens = s.split()
    return set(t for t in tokens if len(t) >= 1)


def jaccard(a, b):
    if not a or not b:
        return 0
    return len(a & b) / len(a | b)


def main():
    print("Loading v2 pipeline...")
    p = DeploymentPipelineV2()
    print("Ready.\n")

    rows = []
    with open('data/ruel_scenarios_final.csv') as f:
        for row in csv.DictReader(f):
            rows.append(row)

    exact_matches = 0
    high_sim = 0  # jaccard >= 0.5
    medium_sim = 0  # jaccard >= 0.3
    total = 0

    low_sim_examples = []
    medium_sim_examples = []

    for row in rows:
        utt = row['사용자발화문'].split('/')[0].strip()
        expected = row['AI기대응답'].strip()
        # 괄호 안 주석 제거 (예: "(현재 거실 조명 밝기는 60%...)")
        expected_clean = re.sub(r'\([^)]*\)', '', expected).strip()
        p.reset_dst()
        r = p.process(utt, use_dst=True)
        actual = r['response']

        total += 1
        if actual == expected or actual == expected_clean:
            exact_matches += 1
            continue

        # Jaccard similarity
        exp_tok = word_tokens(expected_clean)
        act_tok = word_tokens(actual)
        sim = jaccard(exp_tok, act_tok)

        if sim >= 0.5:
            high_sim += 1
        elif sim >= 0.3:
            medium_sim += 1
            medium_sim_examples.append((row, actual, sim))
        else:
            low_sim_examples.append((row, actual, sim))

    print(f"=== Similarity Results ===")
    print(f"  Total scenarios: {total}")
    print(f"  Exact match:     {exact_matches} ({exact_matches/total*100:.1f}%)")
    print(f"  High sim (≥0.5): {high_sim} ({high_sim/total*100:.1f}%)")
    print(f"  Medium (≥0.3):   {medium_sim} ({medium_sim/total*100:.1f}%)")
    print(f"  Low (<0.3):      {len(low_sim_examples)} ({len(low_sim_examples)/total*100:.1f}%)")
    print(f"  usable (exact+high): {exact_matches+high_sim} ({(exact_matches+high_sim)/total*100:.1f}%)")

    print(f'\n=== Low similarity 예시 (최대 20개) ===')
    for row, actual, sim in low_sim_examples[:20]:
        print(f'  sim={sim:.2f} [{row["구분"]}/{row["세부기능"]}]')
        print(f'    utt: {row["사용자발화문"][:50]}')
        print(f'    기대: {row["AI기대응답"][:80]}')
        print(f'    실제: {actual[:80]}')
        print()


if __name__ == '__main__':
    main()
