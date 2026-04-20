#!/usr/bin/env python3
"""Suspect 카테고리 분류 — 진짜 라벨 오류 vs False Positive 구분

카테고리:
A. 확실한 오류 (control 발화인데 dir=none)
B. 확실한 오류 (동일 발화 train/test 충돌)
C. 모호 (query/status 문맥에서 action 키워드)
D. False Positive (규칙만 걸림, 문맥 맞음)
"""
import json, re, os, sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

report = json.load(open('data/label_audit_with_model.json'))

# Query/status indicators - 이게 있으면 action 키워드 있어도 dir=none 맞음
QUERY_MARKERS = [
    '확인', '상태', '보여', '알려', '뭐야', '뭐가', '몇', '얼마', '어때',
    '어디', '있나', '있어?', '있는지', '켜져', '꺼져', '잠겨', '열려',
    '닫혀', '열렸', '닫혔', '잠겼', '작동 중', '가능', '되나',
]

def has_query_marker(utt):
    return any(m in utt for m in QUERY_MARKERS)


def categorize_suspect(s, cross_conflicts_utts):
    utt = s['utterance']
    field = s['field']
    current = s['current']
    expected = s['expected']

    # Category B: Cross-conflict
    if utt in cross_conflicts_utts:
        return 'B_cross_conflict', '학습/테스트 라벨 충돌'

    # Category C: Query/status 문맥인데 규칙이 action 제안
    if field == 'param_direction' and expected in ('on', 'off', 'open', 'close', 'up', 'down', 'set'):
        if has_query_marker(utt):
            # 이건 query 발화, dir=none 맞음
            return 'D_false_positive_query', 'Query 문맥, dir=none 맞음'

    # Category A: 확실한 action 발화 (query marker 없음, control 키워드 있음)
    if field == 'param_direction' and current == 'none' and expected in ('on', 'off', 'open', 'close', 'set'):
        return 'A_clear_error_dir_none', '명확한 control 발화의 dir 누락'

    # direction 반대
    if field == 'param_direction':
        opposites = [('open', 'close'), ('close', 'open'), ('on', 'off'), ('off', 'on'),
                     ('up', 'down'), ('down', 'up')]
        if (current, expected) in opposites or (expected, current) in opposites:
            return 'A_clear_error_opposite', '반대 방향 라벨'

    # close/open vs up/down (블라인드 등)
    if field == 'param_direction' and (current, expected) in [('close', 'down'), ('open', 'up'), ('stop', 'up'), ('stop', 'down'), ('close', 'up'), ('open', 'down')]:
        return 'A_blind_direction_error', '블라인드/커튼 방향 라벨 오류'

    # exec 규칙
    if field == 'exec_type':
        if current == 'direct_respond' and expected == 'query_then_respond':
            if any(q in utt for q in ['얼마', '몇 도', '몇 시', '몇도', '몇시']):
                return 'A_direct_to_query', 'direct→query (수치 질의)'
        if current == 'query_then_respond' and expected == 'control_then_confirm':
            if '호출' in utt or '불러' in utt:
                return 'A_elevator_call_exec', '엘리베이터 호출 exec 오류'

    # Fallback
    return 'C_ambiguous', '검토 필요'


def main():
    # Cross-conflicts utts
    cross_utts = set(c['utterance'] for c in report['cross_conflicts'])

    # Categorize
    train_cat = defaultdict(list)
    test_cat = defaultdict(list)

    for s in report['train_suspects']:
        # Skip model_highconf (너무 많고 noise 많음)
        if s['source'] != 'rule': continue
        cat, reason = categorize_suspect(s, cross_utts)
        train_cat[cat].append({**s, 'reason': reason})

    for s in report['test_suspects']:
        if s['source'] != 'rule': continue
        cat, reason = categorize_suspect(s, cross_utts)
        test_cat[cat].append({**s, 'reason': reason})

    # 출력
    print(f"\n{'═'*70}")
    print(f"  Test Suite — {len(report['test_suspects'])}건 suspect → 카테고리")
    print('═'*70)
    for cat in sorted(test_cat):
        items = test_cat[cat]
        print(f"\n  [{cat}] {len(items)}건 — {items[0]['reason']}")
        for i in items[:5]:
            print(f"    \"{i['utterance']}\" {i['field']}: {i['current']} → {i['expected']}")
        if len(items) > 5:
            print(f"    ... 그 외 {len(items)-5}건")

    print(f"\n{'═'*70}")
    print(f"  Train Data — {len(report['train_suspects'])}건 → 카테고리")
    print('═'*70)
    for cat in sorted(train_cat):
        items = train_cat[cat]
        print(f"\n  [{cat}] {len(items)}건 — {items[0]['reason']}")
        for i in items[:5]:
            print(f"    \"{i['utterance']}\" {i['field']}: {i['current']} → {i['expected']}")
        if len(items) > 5:
            print(f"    ... 그 외 {len(items)-5}건")

    # 최종 요약 테이블
    print(f"\n{'═'*70}")
    print(f"  카테고리별 처리 권장")
    print('═'*70)
    RECOMMEND = {
        'A_clear_error_dir_none': '✅ 자동 수정 가능 (확실한 오류)',
        'A_clear_error_opposite': '✅ 자동 수정 가능 (반대 라벨)',
        'A_blind_direction_error': '✅ 자동 수정 가능 (블라인드 방향)',
        'A_direct_to_query': '✅ 자동 수정 가능',
        'A_elevator_call_exec': '✅ 자동 수정 가능',
        'B_cross_conflict': '⚠️ 수동 검토 (train↔test 충돌)',
        'C_ambiguous': '⚠️ 수동 검토 필요',
        'D_false_positive_query': '❌ 스킵 (규칙 False Positive)',
    }

    all_cats = set(train_cat.keys()) | set(test_cat.keys())
    print(f"\n  {'카테고리':<35} {'Train':>8} {'Test':>8} {'처리':15s}")
    print('  ' + '─' * 70)
    for cat in sorted(all_cats):
        tr = len(train_cat.get(cat, []))
        te = len(test_cat.get(cat, []))
        rec = RECOMMEND.get(cat, '?')
        print(f"  {cat:<35} {tr:>8} {te:>8}  {rec}")

    # Save categorized
    out = {
        'test': {k: v for k, v in test_cat.items()},
        'train': {k: v for k, v in train_cat.items()},
    }
    with open('data/suspects_categorized.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  상세 보고서: data/suspects_categorized.json")


if __name__ == '__main__':
    main()
