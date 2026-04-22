#!/usr/bin/env python3
"""v2 파이프라인으로 219 르엘 시나리오 end-to-end 테스트.

AI기대응답과 비교하여 어떤 발화에서 응답이 이상한지 분석.
"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deployment_pipeline_v2 import DeploymentPipelineV2


def main():
    print("Loading v2 pipeline...")
    p = DeploymentPipelineV2()
    print("Ready.\n")

    rows = []
    with open('data/ruel_scenarios_final.csv') as f:
        for row in csv.DictReader(f):
            rows.append(row)

    print(f"Evaluating {len(rows)} scenarios...\n")

    # 각 시나리오: 발화 → 응답 비교
    results = []
    for row in rows:
        utt = row['사용자발화문'].split('/')[0].strip()  # 슬래시로 여러 variant 있으면 첫 것만
        expected = row['AI기대응답']
        cat = f"{row['구분']}/{row['세부기능']}"
        p.reset_dst()
        r = p.process(utt, use_dst=True)
        results.append({
            'no': row['번호'], 'cat': cat, 'utt': utt,
            'expected': expected, 'actual': r['response'],
            'fn': r['fn'], 'exec': r['exec_type'], 'dir': r['param_direction'],
            'room': r['room'], 'value': r['value'],
        })

    # 카테고리별 요약
    print("=== 카테고리별 응답 샘플 ===\n")
    by_cat = {}
    for res in results:
        by_cat.setdefault(res['cat'], []).append(res)

    for cat in sorted(by_cat.keys()):
        examples = by_cat[cat][:3]
        print(f'[{cat}] ({len(by_cat[cat])}건 중 3건)')
        for e in examples:
            print(f'  "{e["utt"]}"')
            print(f'    fn={e["fn"]}/{e["exec"]}/{e["dir"]}, room={e["room"]}')
            print(f'    기대:  {e["expected"][:80]}')
            print(f'    실제: {e["actual"][:80]}')
        print()

    # CSV 저장
    with open('data/eval_v2_ruel_results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['no', 'cat', 'utt', 'expected', 'actual',
                                            'fn', 'exec', 'dir', 'room', 'value'])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f'Saved → data/eval_v2_ruel_results.csv')


if __name__ == '__main__':
    main()
