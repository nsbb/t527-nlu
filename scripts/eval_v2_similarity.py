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


def parse_utt(raw):
    """CSV 발화문에서 첫 번째 대표 발화 추출.
    "/" 는 대안 발화 구분자지만, 인용부호 내부나 장치명 사이에 있으면 구분자가 아님.
    """
    # 공백-슬래시-공백은 항상 구분자
    if ' / ' in raw:
        return raw.split(' / ')[0].strip()

    parts = raw.split('/')
    if len(parts) == 1:
        return raw.strip()

    first = parts[0].strip()

    # 열린 인용부호가 닫히지 않은 경우 → "/" 가 인용부호 내부
    if first.count("'") % 2 == 1:
        rest = '/'.join(parts[1:])
        close_idx = rest.find("'")
        if close_idx >= 0:
            return (first + "'" + rest[close_idx + 1:]).strip()
        return (first + "'" + rest).strip()

    # 첫 번째 부분이 매우 짧으면 방/기기명 앞에 붙는 경우 → 뒤 부분에서 공백 이후 가져옴
    if len(first) <= 4 and len(parts) > 1:
        rest = '/'.join(parts[1:])
        space_idx = rest.find(' ')
        if space_idx >= 0:
            return (first + rest[space_idx:]).strip()

    return first


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
        utt = parse_utt(row['사용자발화문'])
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
