#!/usr/bin/env python3
"""Test Suite 확장 — 실사용 패턴 추가
- 멀티턴 follow-up 시나리오 (DST 검증용)
- 숫자 다양성 (한글/아라비아/축약)
- STT 오류 변형
- 구어체 변형
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# 기존 로드
existing = json.load(open('data/test_suite.json'))
print(f"기존 Test Suite: {len(existing)}")

new_cases = []

# ============================================================
# 1. 숫자 패턴 확장 (한글/아라비아/축약)
# ============================================================
numeric_patterns = [
    # 온도 — 아라비아
    ("거실 난방 20도", "heat_control", "control_then_confirm", "set"),
    ("거실 난방 21도", "heat_control", "control_then_confirm", "set"),
    ("거실 난방 22도", "heat_control", "control_then_confirm", "set"),
    ("거실 난방 24도", "heat_control", "control_then_confirm", "set"),
    ("거실 난방 26도", "heat_control", "control_then_confirm", "set"),
    ("거실 난방 27도", "heat_control", "control_then_confirm", "set"),
    ("거실 난방 28도", "heat_control", "control_then_confirm", "set"),
    ("거실 에어컨 18도", "ac_control", "control_then_confirm", "set"),
    ("거실 에어컨 19도", "ac_control", "control_then_confirm", "set"),
    ("거실 에어컨 20도", "ac_control", "control_then_confirm", "set"),
    ("거실 에어컨 21도", "ac_control", "control_then_confirm", "set"),
    ("거실 에어컨 22도", "ac_control", "control_then_confirm", "set"),
    ("안방 난방 25도로", "heat_control", "control_then_confirm", "set"),
    ("안방 에어컨 24도로 맞춰", "ac_control", "control_then_confirm", "set"),
    ("침실 에어컨 26도", "ac_control", "control_then_confirm", "set"),
    # 시간
    ("5분 후에 꺼줘", "schedule_manage", "control_then_confirm", "set"),
    ("10분 타이머", "schedule_manage", "control_then_confirm", "set"),
    ("15분 후 꺼줘", "schedule_manage", "control_then_confirm", "set"),
    ("30분 뒤", "schedule_manage", "control_then_confirm", "set"),
    ("1시간 후 알람", "schedule_manage", "control_then_confirm", "set"),
    ("30분 예약", "schedule_manage", "control_then_confirm", "set"),
    ("2시간 뒤 난방 꺼", "schedule_manage", "control_then_confirm", "set"),
    ("오후 3시 알람", "schedule_manage", "control_then_confirm", "set"),
    # 퍼센트/단계
    ("밝기 50%로", "light_control", "control_then_confirm", "set"),
    ("밝기 30%", "light_control", "control_then_confirm", "set"),
    ("3단계로", "ac_control", "control_then_confirm", "set"),
    ("바람 2단계", "ac_control", "control_then_confirm", "set"),
]

for text, fn, exec_t, dir_t in numeric_patterns:
    new_cases.append({
        'utterance': text, 'fn': fn, 'exec': exec_t, 'dir': dir_t,
        'source': 'test_suite_numeric_v67'
    })

# ============================================================
# 2. 구어체/자연스러운 발화
# ============================================================
natural_patterns = [
    ("좀 따뜻하게 해줘", "heat_control", "control_then_confirm", "up"),
    ("좀 시원하게 해줘", "ac_control", "control_then_confirm", "on"),
    ("따뜻하게", "heat_control", "control_then_confirm", "up"),
    ("시원하게", "ac_control", "control_then_confirm", "on"),
    ("덥다", "ac_control", "control_then_confirm", "on"),
    ("춥다", "heat_control", "control_then_confirm", "on"),
    ("더워", "ac_control", "control_then_confirm", "on"),
    ("추워", "heat_control", "control_then_confirm", "on"),
    ("어두워", "light_control", "control_then_confirm", "on"),
    ("너무 밝아", "light_control", "control_then_confirm", "down"),
    ("눈부셔", "light_control", "control_then_confirm", "down"),
    ("자야겠다", "light_control", "control_then_confirm", "off"),
    ("잠잘래", "light_control", "control_then_confirm", "off"),
    ("외출할게", "security_mode", "control_then_confirm", "on"),
    ("집 나간다", "security_mode", "control_then_confirm", "on"),
    ("갈게", "security_mode", "control_then_confirm", "on"),
    ("다녀올게", "security_mode", "control_then_confirm", "on"),
]

for text, fn, exec_t, dir_t in natural_patterns:
    new_cases.append({
        'utterance': text, 'fn': fn, 'exec': exec_t, 'dir': dir_t,
        'source': 'test_suite_natural_v67'
    })

# ============================================================
# 3. 질의/조회 패턴 (query_then_respond)
# ============================================================
query_patterns = [
    ("에어컨 몇 도", "ac_control", "query_then_respond", "none"),
    ("난방 온도 얼마", "heat_control", "query_then_respond", "none"),
    ("가스 잠겨 있어", "gas_control", "query_then_respond", "none"),
    ("현관문 잠겨 있나", "door_control", "query_then_respond", "none"),
    ("조명 켜져 있나", "light_control", "query_then_respond", "none"),
    ("실내 온도 얼마", "heat_control", "query_then_respond", "none"),
    ("엘리베이터 어디 있어", "elevator_call", "query_then_respond", "none"),
    ("이번 달 전기 요금", "energy_query", "query_then_respond", "none"),
    ("이번 달 전기세 얼마", "energy_query", "query_then_respond", "none"),
    ("에너지 사용량", "energy_query", "query_then_respond", "none"),
    ("택배 왔어", "home_info", "query_then_respond", "none"),
    ("방문자 있어", "home_info", "query_then_respond", "none"),
    ("공지사항", "home_info", "query_then_respond", "none"),
]

for text, fn, exec_t, dir_t in query_patterns:
    new_cases.append({
        'utterance': text, 'fn': fn, 'exec': exec_t, 'dir': dir_t,
        'source': 'test_suite_query_v67'
    })

# ============================================================
# 4. 경계 케이스 (이전에 혼동된 패턴)
# ============================================================
edge_cases = [
    # schedule vs system_meta
    ("10분 타이머 설정", "schedule_manage", "control_then_confirm", "set"),
    ("알람 7시로", "schedule_manage", "control_then_confirm", "set"),
    ("알람 10분 뒤", "schedule_manage", "control_then_confirm", "set"),
    ("모닝콜 맞춰줘", "schedule_manage", "control_then_confirm", "set"),
    ("오전 7시 기상", "schedule_manage", "control_then_confirm", "set"),
    # home_info vs system_meta
    ("화면 밝기 줄여", "home_info", "control_then_confirm", "down"),
    ("볼륨 키워", "home_info", "control_then_confirm", "up"),
    ("소리 줄여", "home_info", "control_then_confirm", "down"),
    # weather 명확화
    ("서울 날씨", "weather_query", "query_then_respond", "none"),
    ("부산 오늘 비 와", "weather_query", "query_then_respond", "none"),
    ("이번 주말 날씨", "weather_query", "query_then_respond", "none"),
    ("주말 비 와", "weather_query", "query_then_respond", "none"),
    # system_meta 명확
    ("너 이름 뭐야", "system_meta", "direct_respond", "none"),
    ("누가 만들었어", "system_meta", "direct_respond", "none"),
    ("어떻게 써", "system_meta", "direct_respond", "none"),
    ("기능 알려줘", "system_meta", "direct_respond", "none"),
]

for text, fn, exec_t, dir_t in edge_cases:
    new_cases.append({
        'utterance': text, 'fn': fn, 'exec': exec_t, 'dir': dir_t,
        'source': 'test_suite_edge_v67'
    })

# ============================================================
# 5. direction 다양성
# ============================================================
direction_cases = [
    # up patterns
    ("밝기 올려", "light_control", "control_then_confirm", "up"),
    ("밝게 해", "light_control", "control_then_confirm", "up"),
    ("더 밝게", "light_control", "control_then_confirm", "up"),
    ("환하게", "light_control", "control_then_confirm", "up"),
    # down patterns
    ("어둡게", "light_control", "control_then_confirm", "down"),
    ("은은하게", "light_control", "control_then_confirm", "down"),
    ("희미하게", "light_control", "control_then_confirm", "down"),
    # 기기별
    ("커튼 열어", "curtain_control", "control_then_confirm", "open"),
    ("커튼 닫아", "curtain_control", "control_then_confirm", "close"),
    ("블라인드 올려", "curtain_control", "control_then_confirm", "up"),
    ("블라인드 내려", "curtain_control", "control_then_confirm", "down"),
]

for text, fn, exec_t, dir_t in direction_cases:
    new_cases.append({
        'utterance': text, 'fn': fn, 'exec': exec_t, 'dir': dir_t,
        'source': 'test_suite_direction_v67'
    })

# Dedup with existing
existing_utterances = set(t['utterance'] for t in existing)
new_unique = [c for c in new_cases if c['utterance'] not in existing_utterances]

print(f"신규 추가: {len(new_unique)} (전체 후보 {len(new_cases)}, 중복 제거 {len(new_cases)-len(new_unique)})")
print(f"  숫자 패턴: {sum(1 for c in new_unique if c['source']=='test_suite_numeric_v67')}")
print(f"  자연 발화: {sum(1 for c in new_unique if c['source']=='test_suite_natural_v67')}")
print(f"  질의: {sum(1 for c in new_unique if c['source']=='test_suite_query_v67')}")
print(f"  엣지: {sum(1 for c in new_unique if c['source']=='test_suite_edge_v67')}")
print(f"  direction: {sum(1 for c in new_unique if c['source']=='test_suite_direction_v67')}")

# Save
all_suite = existing + new_unique
with open('data/test_suite_v67.json', 'w', encoding='utf-8') as f:
    json.dump(all_suite, f, ensure_ascii=False, indent=2)

print(f"\n총 {len(all_suite)}개 → data/test_suite_v67.json")
