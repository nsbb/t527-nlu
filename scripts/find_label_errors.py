#!/usr/bin/env python3
"""Test suite 라벨 오류 체계적 탐지
규칙 기반: 발화에서 명백한 힌트 → 기대 라벨과 불일치 시 suspect
"""
import json, re, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

suite = json.load(open('data/test_suite.json'))
print(f"Total: {len(suite)}")

suspects = []

# 규칙 1: "닫아/잠가/잠금" → dir:close
close_keywords = ['닫아', '닫어', '닫자', '잠가', '잠궈', '잠금', '잠그', '잠궜']
# 규칙 2: "열어/여세요" → dir:open
open_keywords = ['열어', '열어봐', '여세요', '여는', '열기']
# 규칙 3: "꺼/끄기/중지/스톱" → dir:off
off_keywords = ['꺼줘', '꺼!', '끄기', '끄자', '끄세요', '중지']
# 규칙 4: "켜/틀어/가동" → dir:on
on_keywords = ['켜줘', '켜!', '켜세요', '틀어', '가동']
# 규칙 5: "호출/불러" → control (elevator)
call_keywords = ['호출', '불러', '오라고']

for i, t in enumerate(suite):
    utt = t['utterance']
    fn = t['fn']; exec_t = t['exec']; dir_t = t['dir']

    # Rule 1: 명백한 close지만 dir이 다른 경우
    if any(k in utt for k in close_keywords):
        if dir_t in ('open', 'on', 'up'):
            suspects.append((i, utt, f"close keyword → dir={dir_t}", {'dir': 'close'}))

    # Rule 2: 명백한 open지만 dir이 close인 경우
    if any(k in utt for k in open_keywords):
        if dir_t in ('close', 'off'):
            suspects.append((i, utt, f"open keyword → dir={dir_t}", {'dir': 'open'}))

    # Rule 3: 명백한 off지만 dir이 on/up
    # "꺼줘", "끄기" 등이 있는데 dir이 on 이면 이상
    if ('꺼' in utt and '꺼줘' not in utt.replace('도어락 꺼', '') and fn != 'door_control'):
        pass  # 단순 "꺼" 문자가 들어간 경우는 스킵 (현관 꺼짐 등 모호)
    elif any(k in utt for k in ['꺼줘', '끄기', '끄자', '끄세요']):
        if dir_t in ('on', 'up', 'open'):
            suspects.append((i, utt, f"off keyword → dir={dir_t}", {'dir': 'off'}))

    # Rule 4: elevator 호출 → control
    if fn == 'elevator_call' and any(k in utt for k in call_keywords):
        if exec_t == 'query_then_respond':
            suspects.append((i, utt, f"elevator call → exec=query", {'exec': 'control_then_confirm'}))

    # Rule 5: 설정/등록/예약 → dir:set (기본)
    if any(k in utt for k in ['등록해줘', '등록해', '설정해줘']):
        if dir_t == 'none':
            suspects.append((i, utt, f"set keyword → dir=none", {'dir': 'set'}))

# 분류
print(f"\nSuspect count: {len(suspects)}")
from collections import Counter
reasons = Counter(s[2] for s in suspects)
for r, c in reasons.most_common():
    print(f"  {c}x {r}")

print(f"\n=== 상세 (처음 30개) ===")
for i, utt, reason, fix in suspects[:30]:
    print(f"  [{i}] \"{utt}\"")
    print(f"       {reason} → suggest {fix}")

# Save suspects list
with open('data/test_suite_suspects.json', 'w', encoding='utf-8') as f:
    json.dump([{
        'idx': s[0], 'utterance': s[1], 'reason': s[2], 'suggested_fix': s[3]
    } for s in suspects], f, ensure_ascii=False, indent=2)
print(f"\n저장: data/test_suite_suspects.json")
