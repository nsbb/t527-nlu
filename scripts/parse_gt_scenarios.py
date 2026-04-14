#!/usr/bin/env python3
"""엑셀 GT 219개 시나리오 → known/unknown 분리 + multi-head 라벨 부여
수동 보정 포함"""
import openpyxl, json
from collections import Counter

path = 'data/260330 르엘 AI서비스 범위 및 시나리오(최종) (1).xlsx'
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


def guess_direction(utt, stype):
    if stype in ('조회',):
        return 'none'
    for kw, d in [('켜', 'on'), ('틀어', 'on'), ('작동', 'on'), ('실행', 'on'),
                   ('호출', 'on'), ('불러', 'on'), ('울려', 'on'),
                   ('꺼', 'off'), ('끄', 'off'), ('취소', 'off'),
                   ('열어', 'open'), ('열기', 'open'),
                   ('닫아', 'close'), ('닫기', 'close'), ('잠가', 'close'), ('잠금', 'close'),
                   ('올려', 'up'), ('높여', 'up'), ('세게', 'up'), ('키워', 'up'), ('최대', 'up'),
                   ('낮춰', 'down'), ('줄여', 'down'), ('어둡게', 'down'), ('은은', 'down'), ('아늑', 'down'),
                   ('멈춰', 'stop'),
                   ('도로', 'set'), ('맞춰', 'set'), ('설정', 'set'),
                   ('등록', 'set'), ('제습', 'set'), ('송풍', 'set'), ('자동', 'set'),
                   ('모드', 'set'), ('바꾸', 'set'), ('예약', 'set'), ('목표', 'set')]:
        if kw in utt:
            return d
    if stype == '전원': return 'on'
    if stype == '조절': return 'set'
    if stype == '호출': return 'on'
    if stype == '예약': return 'set'
    return 'none'


def guess_param_type(utt):
    for kw, pt in [('온도', 'temperature'), ('도로', 'temperature'), ('도 ', 'temperature'),
                    ('밝기', 'brightness'), ('밝게', 'brightness'), ('어둡게', 'brightness'),
                    ('은은', 'brightness'), ('아늑', 'brightness'),
                    ('모드', 'mode'), ('제습', 'mode'), ('송풍', 'mode'), ('자동', 'mode'),
                    ('풍량', 'speed'), ('세게', 'speed'), ('약하게', 'speed'), ('바람', 'speed'),
                    ('볼륨', 'speed'), ('소리', 'speed')]:
        if kw in utt:
            return pt
    return 'none'


def guess_judge(utt, func):
    if func == '미세먼지' or any(w in utt for w in ['창문', '환기해도', '공기']):
        return 'air_quality'
    if any(w in utt for w in ['입고', '옷', '겉옷', '반팔', '코트']):
        return 'clothing'
    if any(w in utt for w in ['세차', '나가도', '캠핑', '소풍', '야외']):
        return 'outdoor_activity'
    if any(w in utt for w in ['주유해도', '떨어질까', '오를까']):
        return 'cost_trend'
    return 'none'


# 진짜 unknown (미지원 기능): 응답이 "지원하지 않", "제공하지 않" 등
# 단, clarify("공간명을 포함하여")나 실제 동작하는 예외는 제외
REAL_UNKNOWN_KEYWORDS = ['지원하지 않', '제공하지 않', '불가', '제공할 수 없',
                          '진행해주세요', '추천은 제공', '조언은 제공', '예측 정보는',
                          '보호 대상', '진단은 지원', '구매는 지원', '예약 기능은']
CLARIFY_KEYWORDS = ['공간명을 포함', '다시 말씀', '정해서 말씀', '필요한 것을',
                     '조절이 필요한']

# 수동 보정 목록 (utterance prefix → 강제 분류)
MANUAL_OVERRIDES = {
    '나 지금 나갈건데': {'fn': 'security_mode', 'exec_type': 'control_then_confirm', 'param_direction': 'on'},
    '네 이름을 바꾸고 싶어': {'fn': 'system_meta', 'exec_type': 'direct_respond'},  # 미지원이지만 system_meta 응답
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
        # 매뉴얼/시간 등 고정 응답 → direct_respond
        if func in ('매뉴얼', '시간', '비밀번호 변경'):
            exec_type = 'direct_respond'
        # 환경설정 조회 중 "할 수 있어?" 류 → direct_respond
        elif any(w in utt for w in ['할 수 있', '어떻게 조절', '설정가능', '뭐야?', '뭐 할']):
            exec_type = 'direct_respond'
        else:
            exec_type = 'query_then_respond'
    elif stype in ('전원', '조절', '호출'):
        exec_type = 'control_then_confirm'
    elif stype == '예약':
        # "예약 있어?" = query, "예약 설정" = control
        if any(w in utt for w in ['있어', '어떻게 되', '확인', '뭐야']):
            exec_type = 'query_then_respond'
            direction = 'none'
        else:
            exec_type = 'control_then_confirm'
    elif stype == '예외':
        exec_type = 'direct_respond'
    else:
        exec_type = 'query_then_respond'

    # clarify 판정
    if any(kw in resp for kw in CLARIFY_KEYWORDS):
        exec_type = 'clarify'
        direction = 'none'

    # unknown 판정 (진짜 미지원 기능)
    is_unknown = False
    if any(kw in resp for kw in REAL_UNKNOWN_KEYWORDS):
        is_unknown = True

    # 수동 보정 적용
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
        labels['fn'] = 'unknown'
        labels['exec_type'] = 'direct_respond'
        labels['param_direction'] = 'none'
        labels['param_type'] = 'none'
        labels['judge'] = 'none'

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

    if is_unk:
        unknown.append(entry)
    else:
        known.append(entry)
    sid += 1

# 저장
with open('data/gt_known_scenarios.json', 'w', encoding='utf-8') as f:
    json.dump(known, f, ensure_ascii=False, indent=2)
with open('data/gt_unknown_scenarios.json', 'w', encoding='utf-8') as f:
    json.dump(unknown, f, ensure_ascii=False, indent=2)

# 통계
print(f"Known: {len(known)}개, Unknown: {len(unknown)}개\n")

fn_c = Counter(s['labels']['fn'] for s in known)
print("--- known fn ---")
for k, v in fn_c.most_common(): print(f"  {k}: {v}")

exec_c = Counter(s['labels']['exec_type'] for s in known)
print("\n--- known exec ---")
for k, v in exec_c.most_common(): print(f"  {k}: {v}")

dir_c = Counter(s['labels']['param_direction'] for s in known)
print("\n--- known direction ---")
for k, v in dir_c.most_common(): print(f"  {k}: {v}")

judge_c = Counter(s['labels']['judge'] for s in known)
print("\n--- known judge ---")
for k, v in judge_c.most_common(): print(f"  {k}: {v}")

print(f"\n--- unknown ({len(unknown)}개) ---")
for e in unknown:
    print(f"  [{e['func']}] {e['utterance'][:35]:35s} → {e['response'][:40]}")

# clarify 확인
clarify = [s for s in known if s['labels']['exec_type'] == 'clarify']
print(f"\n--- clarify ({len(clarify)}개) ---")
for e in clarify:
    print(f"  [{e['func']}] {e['utterance'][:40]}")
