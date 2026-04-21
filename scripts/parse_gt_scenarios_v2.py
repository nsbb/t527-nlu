#!/usr/bin/env python3
"""엑셀 GT → multi-head 라벨 v2 (원본 parse 버그 수정)

원본 `parse_gt_scenarios.py`의 발견된 문제:
  1. slash("/") 포함된 발화 (예: "닫아줘 / 열어줘") 에서 키워드 순서대로 매칭 → 뒤쪽 키워드 무시
  2. "밝게" 키워드 없음 (dir=up 매칭 불가)
  3. "어둡게"는 있지만 "밝게"는 빠짐
  4. stype 기본값 처리가 부정확 ("조절" → 'set')
  5. exec 보정 로직 일부 미비

개선:
  - 키워드 매칭 순서 reorder (exclusive 먼저, generic 나중)
  - "밝게" 추가 (dir=up)
  - slash 분리 처리 (첫 action만 사용, 대체 패턴 언급)
  - exec=query 패턴 명시화 (몇 도/얼마)
"""
import openpyxl, json, re
from collections import Counter

import os
path = 'data/260330 르엘 AI서비스 범위 및 시나리오(최종) (1).xlsx'
if not os.path.exists(path):
    path = '/home/nsbb/travail/claude/T527/docs/260330 르엘 AI서비스 범위 및 시나리오(최종) (1).xlsx'
wb = openpyxl.load_workbook(path, read_only=True)
ws = wb['월패드-AI서비스(0330)']


FUNC_TO_FN = {
    '조명': 'light_control', '난방': 'heat_control', '에어컨': 'ac_control',
    '환기': 'vent_control', '가스': 'gas_control', '도어락': 'door_control',
    '로비': 'door_control', '전동커튼': 'curtain_control', '엘리베이터': 'elevator_call',
    '외출/재택': 'security_mode', '비상': 'security_mode', '모닝콜': 'schedule_manage',
    '날씨': 'weather_query', '미세먼지': 'weather_query', '뉴스': 'news_query',
    '교통정보': 'traffic_query', '원격 검침': 'energy_query',
    '매뉴얼': 'system_meta', '비밀번호 변경': 'system_meta',
    '환경설정': 'home_info', '상태': 'home_info', '시간': 'home_info',
    '알림': 'home_info', '공지사항': 'home_info', '단지정보': 'home_info',
    '커뮤니티': 'home_info',
    '증시': 'market_query', '유가정보': 'market_query',
    '의료': 'medical_query',
    '전기차': 'vehicle_manage', '차량 출입 내역': 'vehicle_manage',
    '방문객 차량 등록': 'vehicle_manage',
}


ACTION_VERBS = ['닫', '잠', '열', '켜', '꺼', '끄', '틀', '올려', '낮춰', '높여',
                 '줄여', '세게', '약하게', '가동', '호출', '불러', '울려', '밝게',
                 '어둡게', '취소', '중지', '맞춰', '바꾸', '설정', '예약', '등록']


def normalize_utt(utt):
    """Slash 처리 — alternatives vs alias 구분."""
    if '/' not in utt:
        return utt
    parts = [p.strip() for p in utt.split('/')]
    # 각 파트에 action 동사가 있는지 확인
    parts_with_verb = [p for p in parts if any(v in p for v in ACTION_VERBS)]
    # 여러 파트에 동사 있음 → alternatives, 첫 파트만
    if len(parts_with_verb) >= 2:
        return parts_with_verb[0]
    # 한 파트에만 동사 → alias list, verb 있는 파트 사용
    if len(parts_with_verb) == 1:
        return parts_with_verb[0]
    # 둘 다 없음 (예: "모닝콜이 울리고 / 울릴 때 ...") → 전체 유지
    return utt


def guess_direction(utt, stype):
    """개선된 dir 추측"""
    # 첫 번째 action만 (slash 등 정제)
    clean = normalize_utt(utt)

    # Query markers — 있으면 dir=none
    query_markers = ['확인', '상태', '알려', '뭐야', '몇 도', '몇도', '얼마', '있나', '있어', '켜져 있', '꺼져 있', '잠겨']
    if any(m in clean for m in query_markers):
        # 상태 조회는 dir=none
        return 'none'

    # 명시적 stype이 조회면 dir=none
    if stype in ('조회',):
        return 'none'

    # 우선순위 순 (exclusive → generic)
    # 1. "밝게" — up (수정 포인트)
    if '밝게' in clean or '밝아' in clean:
        return 'up'
    # 2. "어둡게/은은/침침" — down
    if '어둡게' in clean or '은은' in clean or '침침' in clean or '아늑' in clean:
        return 'down'
    # 3. 끝 close (닫/잠) — open보다 먼저
    if any(k in clean for k in ['닫아', '닫어', '닫기', '잠가', '잠궈', '잠그', '잠금']):
        return 'close'
    # 4. open (열어, 열기) — 단 "열려 있" 같은 query 제외 (위에서 처리됨)
    if any(k in clean for k in ['열어', '열기', '열어봐']):
        return 'open'
    # 5. off (꺼/끄/취소/중지)
    if any(k in clean for k in ['꺼', '끄기', '끄자', '끄세요', '취소', '중지', '중단']):
        return 'off'
    # 6. on (켜/틀어/가동/호출/불러)
    if any(k in clean for k in ['켜', '틀어', '작동', '실행', '가동', '호출', '불러', '울려']):
        return 'on'
    # 7. up (올려/높여/세게/키워/최대)
    if any(k in clean for k in ['올려', '높여', '세게', '키워', '최대']):
        return 'up'
    # 8. down (낮춰/줄여/약하게/최소)
    if any(k in clean for k in ['낮춰', '줄여', '약하게', '최소']):
        return 'down'
    # 9. stop (멈춰)
    if '멈춰' in clean:
        return 'stop'
    # 10. set (모드/설정/예약/맞춰/도로/목표/바꾸)
    if any(k in clean for k in ['모드', '설정', '예약', '맞춰', '도로', '목표', '바꾸', '등록', '제습', '송풍', '자동']):
        return 'set'

    # stype fallback
    if stype == '전원': return 'on'
    if stype == '조절': return 'set'
    if stype == '호출': return 'on'
    if stype == '예약': return 'set'
    return 'none'


def guess_param_type(utt):
    """개선된 param_type 추측"""
    clean = normalize_utt(utt)
    for kw, pt in [('온도', 'temperature'), ('도로', 'temperature'), ('도 ', 'temperature'),
                    ('밝기', 'brightness'), ('밝게', 'brightness'), ('어둡게', 'brightness'),
                    ('은은', 'brightness'), ('아늑', 'brightness'),
                    ('모드', 'mode'), ('제습', 'mode'), ('송풍', 'mode'), ('자동', 'mode'),
                    ('풍량', 'speed'), ('세게', 'speed'), ('약하게', 'speed'), ('바람', 'speed'),
                    ('볼륨', 'speed'), ('소리', 'speed')]:
        if kw in clean:
            return pt
    return 'none'


def guess_judge(utt, func):
    clean = normalize_utt(utt)
    if func == '미세먼지' or any(w in clean for w in ['창문', '환기해도', '공기']):
        return 'air_quality'
    if any(w in clean for w in ['입고', '옷', '겉옷', '반팔', '코트']):
        return 'clothing'
    if any(w in clean for w in ['세차', '나가도', '캠핑', '소풍', '야외']):
        return 'outdoor_activity'
    if any(w in clean for w in ['주유해도', '떨어질까', '오를까']):
        return 'cost_trend'
    return 'none'


REAL_UNKNOWN_KEYWORDS = ['지원하지 않', '제공하지 않', '불가', '제공할 수 없',
                          '진행해주세요', '추천은 제공', '조언은 제공', '예측 정보는',
                          '보호 대상', '진단은 지원', '구매는 지원', '예약 기능은']
CLARIFY_KEYWORDS = ['공간명을 포함', '다시 말씀', '정해서 말씀', '필요한 것을',
                     '조절이 필요한']

MANUAL_OVERRIDES = {
    '나 지금 나갈건데': {'fn': 'security_mode', 'exec_type': 'control_then_confirm', 'param_direction': 'on'},
    '네 이름을 바꾸고 싶어': {'fn': 'system_meta', 'exec_type': 'direct_respond'},
}


def classify_scenario(cat, func, stype, utt, resp):
    fn = FUNC_TO_FN.get(func, 'home_info')
    direction = guess_direction(utt, stype)
    param_type = guess_param_type(utt)
    judge = guess_judge(utt, func)

    # exec_type 결정
    if judge != 'none':
        exec_type = 'query_then_judge'
    elif stype == '조회':
        # 특수: 매뉴얼/시간/비번 등은 direct_respond
        if func in ('매뉴얼', '시간', '비밀번호 변경'):
            exec_type = 'direct_respond'
        # 자주 쓰이는 "할 수 있" 류
        elif any(w in utt for w in ['할 수 있', '어떻게 조절', '설정가능', '뭐야?', '뭐 할']):
            exec_type = 'direct_respond'
        else:
            exec_type = 'query_then_respond'
    elif stype in ('전원', '조절', '호출'):
        exec_type = 'control_then_confirm'
    elif stype == '예약':
        if any(w in utt for w in ['있어', '어떻게 되', '확인', '뭐야']):
            exec_type = 'query_then_respond'
            direction = 'none'
        else:
            exec_type = 'control_then_confirm'
    elif stype == '예외':
        exec_type = 'direct_respond'
    else:
        exec_type = 'query_then_respond'

    # clarify
    if any(kw in resp for kw in CLARIFY_KEYWORDS):
        exec_type = 'clarify'
        direction = 'none'

    # unknown
    is_unknown = False
    if any(kw in resp for kw in REAL_UNKNOWN_KEYWORDS):
        is_unknown = True

    # Manual overrides
    for prefix, override in MANUAL_OVERRIDES.items():
        if utt.startswith(prefix):
            fn = override.get('fn', fn)
            exec_type = override.get('exec_type', exec_type)
            direction = override.get('param_direction', direction)
            is_unknown = False
            break

    labels = {
        'fn': fn,
        'exec_type': exec_type,
        'param_direction': direction,
        'param_type': param_type,
        'judge': judge,
    }

    if is_unknown:
        labels = {'fn': 'unknown', 'exec_type': 'direct_respond',
                  'param_direction': 'none', 'param_type': 'none', 'judge': 'none'}

    return labels, is_unknown


# 파싱
known = []
unknown = []
sid = 0

for row in ws.iter_rows(min_row=2, values_only=True):
    cat = (row[0] or '').strip()
    func = (row[1] or '').strip()
    stype = (row[2] or '').strip()
    utt = (row[3] or '').strip()
    resp = (row[4] or '').strip()
    dev = (row[5] or '').strip()
    if not utt or dev != 'O':
        continue

    labels, is_unk = classify_scenario(cat, func, stype, utt, resp)

    entry = {
        'scenario_id': sid,
        'cat': cat, 'func': func, 'stype': stype,
        'utterance': utt, 'response': resp[:100],
        'labels': labels,
    }
    (unknown if is_unk else known).append(entry)
    sid += 1

with open('data/gt_known_scenarios_v2.json', 'w', encoding='utf-8') as f:
    json.dump(known, f, ensure_ascii=False, indent=2)
with open('data/gt_unknown_scenarios_v2.json', 'w', encoding='utf-8') as f:
    json.dump(unknown, f, ensure_ascii=False, indent=2)

print(f"Known: {len(known)}개, Unknown: {len(unknown)}개\n")

# v1 vs v2 비교
gt1_known = json.load(open('data/gt_known_scenarios.json'))
gt1_unknown = json.load(open('data/gt_unknown_scenarios.json'))
gt1_all = {s['utterance']: s['labels'] for s in gt1_known + gt1_unknown}

gt2_all = {s['utterance']: s['labels'] for s in known + unknown}

diffs = []
for utt in gt2_all:
    if utt in gt1_all:
        for k in ['fn', 'exec_type', 'param_direction', 'param_type']:
            if gt1_all[utt].get(k) != gt2_all[utt].get(k):
                diffs.append((utt, k, gt1_all[utt].get(k), gt2_all[utt].get(k)))

print(f"=== v1 → v2 차이 ({len(diffs)}개) ===")
for utt, k, v1, v2 in diffs[:30]:
    print(f"  \"{utt}\" {k}: {v1} → {v2}")
if len(diffs) > 30:
    print(f"  ... 그 외 {len(diffs) - 30}개")
