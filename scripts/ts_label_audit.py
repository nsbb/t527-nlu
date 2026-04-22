#!/usr/bin/env python3
"""Test Suite Label Audit — TS 라벨 일관성 검토.

목적:
- 같은 의미 패턴을 annotator가 다르게 라벨한 케이스 찾기
- "서울 날씨" direct vs "광주 날씨" query 같은 것.
- Semantic-consistent TS로 재구축한다면 제외/수정 제안.
"""
import os, sys
sys.path.insert(0, 'scripts')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from collections import defaultdict, Counter


def main():
    test = json.load(open('data/test_suite.json'))
    print(f"Total TS: {len(test)}\n")

    # Group by fn + semantic pattern
    patterns = defaultdict(list)
    for x in test:
        utt = x['utterance']
        # Normalize to semantic pattern:
        # - Remove specific city/room names
        # - Replace numbers with N
        norm = re.sub(r'\d+', 'N', utt)
        # Replace cities
        for city in ['서울', '부산', '춘천', '대전', '광주', '인천', '울산', '대구', '수원']:
            norm = norm.replace(city, '[CITY]')
        # Replace rooms
        for room in ['거실', '안방', '침실', '주방', '부엌', '작은방', '아이방']:
            norm = norm.replace(room, '[ROOM]')
        patterns[norm].append(x)

    # Inconsistent: same pattern, different labels
    inconsistent = []
    for pat, items in patterns.items():
        if len(items) < 2:
            continue
        labels = Counter((i['fn'], i['exec'], i['dir']) for i in items)
        if len(labels) > 1:
            inconsistent.append((pat, items, labels))

    print(f"Inconsistent patterns: {len(inconsistent)}")
    print(f"Total cases in inconsistent patterns: {sum(len(i[1]) for i in inconsistent)}")
    print()

    # Top patterns
    print("=" * 70)
    print("Top 15 inconsistent patterns")
    print("=" * 70)
    for pat, items, labels in sorted(inconsistent, key=lambda x: -len(x[1]))[:15]:
        print(f"\nPattern: \"{pat}\" ({len(items)} cases)")
        print(f"  Label distribution: {dict(labels.most_common())}")
        # Show 2 samples per distinct label
        from collections import Counter as C2
        shown = C2()
        for item in items:
            key = (item['fn'], item['exec'], item['dir'])
            if shown[key] < 2:
                shown[key] += 1
                print(f'  [{item["fn"]}/{item["exec"]}/{item["dir"]}] "{item["utterance"]}"')

    # Summary for fixing
    fix_candidates = []
    for pat, items, labels in inconsistent:
        # Majority wins
        if len(labels) >= 2:
            majority_label = labels.most_common(1)[0][0]
            majority_count = labels.most_common(1)[0][1]
            minority_count = sum(c for _, c in labels.most_common()[1:])
            for item in items:
                if (item['fn'], item['exec'], item['dir']) != majority_label:
                    fix_candidates.append({
                        'utterance': item['utterance'],
                        'current': {'fn': item['fn'], 'exec': item['exec'], 'dir': item['dir']},
                        'majority': {'fn': majority_label[0], 'exec': majority_label[1], 'dir': majority_label[2]},
                        'pattern': pat,
                        'majority_count': majority_count,
                        'minority_count': minority_count,
                    })

    with open('data/ts_label_fix_candidates.json', 'w', encoding='utf-8') as f:
        json.dump(fix_candidates, f, ensure_ascii=False, indent=2)
    print(f"\n{'='*70}")
    print(f"Fix candidates: {len(fix_candidates)} cases saved to data/ts_label_fix_candidates.json")
    print(f"If applied, would potentially boost TS accuracy by {len(fix_candidates)/len(test)*100:.2f}%p (if model matches majority)")


if __name__ == '__main__':
    main()
