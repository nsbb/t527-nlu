#!/usr/bin/env python3
"""부족 fn 증강 — seed 발화에서 변형 생성
어미 변형 + Room 교차 + 존댓말/반말 + STT 오류 주입

외부 LLM 없이 규칙 기반으로 빠르게 생성
"""
import json, random, re, copy
from collections import Counter

random.seed(42)

# ============================================================
# 변형 규칙
# ============================================================

# 어미 변형 패턴
VERB_ENDINGS = {
    '켜줘': ['켜줘', '켜봐', '켜라', '켜주세요', '켜줄래', '켜줄래?', '키다', '켜', '좀 켜줘', '켜줘요'],
    '꺼줘': ['꺼줘', '꺼봐', '꺼라', '꺼주세요', '꺼줄래', '끄다', '꺼', '좀 꺼줘', '꺼줘요'],
    '열어줘': ['열어줘', '열어봐', '열어라', '열어주세요', '열어줄래', '열어', '좀 열어줘'],
    '닫아줘': ['닫아줘', '닫아봐', '닫아라', '닫아주세요', '닫아줄래', '닫아', '좀 닫아줘'],
    '올려줘': ['올려줘', '올려봐', '올려라', '올려주세요', '올려줄래', '올려', '좀 올려줘', '높여줘'],
    '낮춰줘': ['낮춰줘', '낮춰봐', '낮춰라', '낮춰주세요', '낮춰줄래', '낮춰', '좀 낮춰줘', '줄여줘'],
    '맞춰줘': ['맞춰줘', '맞춰봐', '맞춰주세요', '맞춰줄래', '설정해줘', '해줘', '맞춰'],
    '해줘': ['해줘', '해봐', '해라', '해주세요', '해줄래', '해줘요', '해', '좀 해줘'],
    '알려줘': ['알려줘', '알려봐', '알려주세요', '알려줄래', '말해줘', '알려줘요', '알려'],
    '어때': ['어때', '어때?', '어떠니', '어떠냐', '어떤가요', '어떻게 돼', '어때요'],
    '있어': ['있어', '있어?', '있나', '있니', '있나요', '있습니까', '있음?'],
    '뭐야': ['뭐야', '뭐야?', '뭐니', '뭐냐', '뭔가요', '뭐예요', '뭐임'],
}

# Room 변형
ROOMS = {
    'living': ['거실', '거실에', '거실의'],
    'kitchen': ['주방', '주방에', '부엌', '부엌에'],
    'bedroom_main': ['안방', '안방에', '안방의', '큰방'],
    'bedroom_sub': ['작은방', '작은방에', '침실', '침실에', '아이방'],
    'all': ['전체', '전부', '모든', '다'],
}

ROOM_KEYWORDS = ['거실', '주방', '부엌', '안방', '큰방', '작은방', '침실', '아이방', '전체', '전부']

# 디바이스별 간접 표현
INDIRECT_EXPRESSIONS = {
    'ac_control': [
        ('찜통이야', 'control_then_confirm', 'on'),
        ('너무 더워', 'control_then_confirm', 'on'),
        ('덥다 덥다', 'control_then_confirm', 'on'),
        ('열대야네', 'control_then_confirm', 'on'),
        ('더워 죽겠다', 'control_then_confirm', 'on'),
        ('시원하게 좀', 'control_then_confirm', 'on'),
        ('에어컨 좀', 'control_then_confirm', 'on'),
        ('냉방 좀', 'control_then_confirm', 'on'),
        ('에어컨 있으면 좋겠다', 'control_then_confirm', 'on'),
        ('땀이 줄줄', 'control_then_confirm', 'on'),
    ],
    'heat_control': [
        ('좀 춥다', 'control_then_confirm', 'on'),
        ('너무 추워', 'control_then_confirm', 'on'),
        ('얼어 죽겠다', 'control_then_confirm', 'on'),
        ('시베리아야', 'control_then_confirm', 'on'),
        ('추워 추워', 'control_then_confirm', 'on'),
        ('따뜻하게 좀', 'control_then_confirm', 'on'),
        ('손 시려', 'control_then_confirm', 'on'),
        ('발 시려', 'control_then_confirm', 'on'),
        ('한기가 든다', 'control_then_confirm', 'on'),
    ],
    'light_control': [
        ('어두워', 'control_then_confirm', 'on'),
        ('안 보여', 'control_then_confirm', 'on'),
        ('깜깜해', 'control_then_confirm', 'on'),
        ('동굴이야', 'control_then_confirm', 'on'),
        ('눈이 부셔', 'control_then_confirm', 'down'),
        ('너무 밝아', 'control_then_confirm', 'down'),
        ('눈부셔', 'control_then_confirm', 'down'),
        ('분위기 좀 내자', 'control_then_confirm', 'down'),
    ],
    'vent_control': [
        ('공기가 탁해', 'control_then_confirm', 'on'),
        ('답답해', 'control_then_confirm', 'on'),
        ('냄새 나', 'control_then_confirm', 'on'),
        ('환기 좀', 'control_then_confirm', 'on'),
        ('공기 좀 바꾸자', 'control_then_confirm', 'on'),
        ('숨이 막혀', 'control_then_confirm', 'on'),
        ('시끄러워', 'control_then_confirm', 'off'),
        ('환기 소리 거슬려', 'control_then_confirm', 'off'),
    ],
    'security_mode': [
        ('나 나간다', 'control_then_confirm', 'on'),
        ('외출할게', 'control_then_confirm', 'on'),
        ('잠깐 나갔다 올게', 'control_then_confirm', 'on'),
        ('출근한다', 'control_then_confirm', 'on'),
        ('나 들어왔어', 'control_then_confirm', 'set'),
        ('다녀왔습니다', 'control_then_confirm', 'set'),
    ],
    'gas_control': [
        ('가스 잠갔나', 'query_then_respond', 'none'),
        ('가스 괜찮아?', 'query_then_respond', 'none'),
        ('가스 안전한가', 'query_then_respond', 'none'),
    ],
    'elevator_call': [
        ('내려갈 건데', 'control_then_confirm', 'on'),
        ('나가야 하는데', 'control_then_confirm', 'on'),
        ('엘베 좀', 'control_then_confirm', 'on'),
        ('엘리베이터 오고 있어?', 'query_then_respond', 'none'),
    ],
}

# STT 오류 패턴
STT_ERRORS = {
    '에어컨': ['에어콘', '애어컨', '에어컨'],
    '난방': ['남방', '난반', '날방'],
    '환기': ['환기', '한기'],
    '커튼': ['커턴', '커튼'],
    '엘리베이터': ['엘레베이터', '엘리배이터', '엘베'],
    '조명': ['조명', '조멍'],
    '보안': ['보안', '보완'],
    '에너지': ['에널지', '에너지'],
    '주차': ['주차', '주챠'],
    '도어락': ['도어락', '도어록', '도얼락'],
    '가스': ['까스', '가쓰'],
    '밸브': ['밸브', '벨브', '밸부'],
}


def apply_verb_ending(text):
    """어미 변형 적용"""
    for pattern, variants in VERB_ENDINGS.items():
        if pattern in text:
            new_ending = random.choice(variants)
            return text.replace(pattern, new_ending, 1)
    return text


def apply_room_swap(text, fn):
    """Room 교체 — 디바이스 제어 fn만"""
    device_fns = ['light_control', 'heat_control', 'ac_control', 'vent_control', 'curtain_control']
    if fn not in device_fns:
        return text, None

    # 기존 room 제거
    clean = text
    for kw in ROOM_KEYWORDS:
        clean = clean.replace(kw, '').strip()
        clean = re.sub(r'\s+', ' ', clean)

    # 새 room 추가
    room_key = random.choice(list(ROOMS.keys()))
    room_word = random.choice(ROOMS[room_key])
    new_text = f"{room_word} {clean}"
    return new_text, room_key


def apply_stt_error(text):
    """STT 오류 주입 (10% 확률로 각 키워드)"""
    result = text
    for correct, errors in STT_ERRORS.items():
        if correct in result and random.random() < 0.15:
            error = random.choice([e for e in errors if e != correct] or errors)
            result = result.replace(correct, error, 1)
    return result


def apply_politeness(text):
    """존댓말/반말 변형"""
    variants = []
    # 반말화
    for polite, casual in [('주세요', '줘'), ('하세요', '해'), ('입니다', '이야'),
                            ('됩니까', '돼?'), ('줄래요', '줄래')]:
        if polite in text:
            variants.append(text.replace(polite, casual))
    # 존댓말화
    for casual, polite in [('줘', '주세요'), ('해', '해주세요'), ('봐', '봐주세요'),
                            ('라', '주세요')]:
        if text.endswith(casual):
            variants.append(text[:-len(casual)] + polite)
    return random.choice(variants) if variants else text


def get_exec_and_direction(intent_seed, fn):
    """seed의 exec_type과 direction 추정"""
    utt = intent_seed['utterance']
    exec_type = intent_seed.get('exec_type', 'control_then_confirm')

    # direction 추정
    direction = 'none'
    if exec_type == 'query_then_respond':
        direction = 'none'
    elif exec_type == 'query_then_judge':
        direction = 'none'
    elif exec_type == 'direct_respond':
        direction = 'none'
    elif exec_type == 'clarify':
        direction = 'none'
    else:
        # control_then_confirm — 발화에서 direction 추정
        for kw, d in [('켜', 'on'), ('틀어', 'on'), ('작동', 'on'), ('실행', 'on'), ('호출', 'on'),
                      ('꺼', 'off'), ('끄', 'off'), ('중단', 'off'), ('취소', 'off'),
                      ('열어', 'open'), ('열려', 'open'),
                      ('닫아', 'close'), ('닫혀', 'close'), ('잠가', 'close'), ('잠금', 'close'),
                      ('올려', 'up'), ('높여', 'up'), ('세게', 'up'), ('밝게', 'up'),
                      ('낮춰', 'down'), ('줄여', 'down'), ('약하게', 'down'), ('어둡게', 'down'),
                      ('멈춰', 'stop'), ('중지', 'stop'),
                      ('도로', 'set'), ('모드', 'set'), ('설정', 'set'), ('맞춰', 'set'),
                      ('예약', 'set'), ('등록', 'set')]:
            if kw in utt:
                direction = d
                break
        if direction == 'none' and exec_type == 'control_then_confirm':
            direction = 'on'  # default

    # param_type 추정
    param_type = 'none'
    for kw, pt in [('온도', 'temperature'), ('도로', 'temperature'), ('도 ', 'temperature'),
                   ('밝기', 'brightness'), ('밝게', 'brightness'), ('어둡게', 'brightness'),
                   ('은은하게', 'brightness'),
                   ('모드', 'mode'), ('제습', 'mode'), ('송풍', 'mode'), ('자동', 'mode'),
                   ('풍량', 'speed'), ('세게', 'speed'), ('약하게', 'speed'), ('바람', 'speed'),
                   ('볼륨', 'speed'), ('소리', 'speed')]:
        if kw in utt:
            param_type = pt
            break

    # judge 추정
    judge = 'none'
    if exec_type == 'query_then_judge':
        for kw, j in [('입고', 'clothing'), ('옷', 'clothing'), ('겉옷', 'clothing'),
                      ('세차', 'outdoor_activity'), ('나가도', 'outdoor_activity'),
                      ('캠핑', 'outdoor_activity'), ('외출', 'outdoor_activity'),
                      ('미세먼지', 'air_quality'), ('환기해도', 'air_quality'), ('창문', 'air_quality'),
                      ('요금', 'cost_trend'), ('가격', 'cost_trend'), ('주유', 'cost_trend')]:
            if kw in utt:
                judge = j
                break

    return exec_type, direction, param_type, judge


def augment_seed(seed, fn, target_count):
    """하나의 seed에서 target_count개 변형 생성"""
    results = []
    exec_type, direction, param_type, judge = get_exec_and_direction(seed, fn)
    base_utt = seed['utterance']

    for _ in range(target_count):
        utt = base_utt
        d = direction
        pt = param_type
        j = judge
        et = exec_type
        room = 'none'

        # 변형 적용 (랜덤 조합)
        r = random.random()
        if r < 0.25:
            utt = apply_verb_ending(utt)
        elif r < 0.45:
            utt, new_room = apply_room_swap(utt, fn)
            if new_room:
                room = new_room
        elif r < 0.60:
            utt = apply_politeness(utt)
        elif r < 0.75:
            utt = apply_stt_error(utt)
        else:
            # 복합: 어미 + room
            utt = apply_verb_ending(utt)
            utt, new_room = apply_room_swap(utt, fn)
            if new_room:
                room = new_room

        # 중복 방지를 위한 미세 변형
        if random.random() < 0.3:
            prefixes = ['', '좀 ', '지금 ', '빨리 ', '']
            utt = random.choice(prefixes) + utt
        if random.random() < 0.2:
            suffixes = ['', ' 좀', ' 지금', ' 바로', '']
            utt = utt + random.choice(suffixes)

        utt = re.sub(r'\s+', ' ', utt).strip()

        results.append({
            'utterance': utt,
            'flat_intent': f"{fn}_{et}",
            'labels': {
                'fn': fn,
                'exec_type': et,
                'param_direction': d,
                'param_type': pt,
                'judge': j,
            },
            'source': 'augment_rule'
        })

    return results


def augment_indirect(fn, count):
    """간접 표현 생성"""
    expressions = INDIRECT_EXPRESSIONS.get(fn, [])
    if not expressions:
        return []

    results = []
    for _ in range(count):
        expr, et, d = random.choice(expressions)
        utt = expr

        # Room 추가 (50%)
        if random.random() < 0.5 and fn in ['ac_control', 'heat_control', 'light_control', 'vent_control']:
            room_key = random.choice(list(ROOMS.keys()))
            room_word = random.choice(ROOMS[room_key])
            utt = f"{room_word} {utt}" if random.random() < 0.5 else f"{utt} {room_word}"

        # 어미 변형
        if random.random() < 0.3:
            suffixes = ['', '요', '야', '네', '다', '어']
            utt = utt.rstrip('야네다어요') + random.choice(suffixes)

        utt = re.sub(r'\s+', ' ', utt).strip()

        results.append({
            'utterance': utt,
            'flat_intent': f"{fn}_{et}",
            'labels': {
                'fn': fn,
                'exec_type': et,
                'param_direction': d,
                'param_type': 'none',
                'judge': 'none',
            },
            'source': 'augment_indirect'
        })

    return results


def main():
    with open('data/augment_seeds.json', encoding='utf-8') as f:
        seeds = json.load(f)

    # 현재 CNN 데이터의 fn 분포
    current = {
        'ac_control': 591, 'gas_control': 67, 'elevator_call': 62,
        'security_mode': 182, 'energy_query': 155, 'system_meta': 320,
        'vehicle_manage': 168, 'curtain_control': 171, 'door_control': 154,
        'vent_control': 354, 'market_query': 362, 'medical_query': 280,
        'light_control': 932, 'heat_control': 698, 'home_info': 844,
        'schedule_manage': 444, 'news_query': 586, 'traffic_query': 504,
    }

    # 목표
    targets = {
        'ac_control': 1300, 'gas_control': 800, 'elevator_call': 700,
        'security_mode': 900, 'energy_query': 900, 'system_meta': 1000,
        'vehicle_manage': 800, 'curtain_control': 900, 'door_control': 750,
        'vent_control': 1100, 'market_query': 1000, 'medical_query': 800,
        'light_control': 1100, 'heat_control': 1000, 'home_info': 1200,
        'schedule_manage': 750, 'news_query': 800, 'traffic_query': 800,
    }

    all_augmented = []

    for fn, seed_list in seeds.items():
        if not seed_list:
            continue

        cur = current.get(fn, 0)
        tgt = targets.get(fn, 800)
        need = max(0, tgt - cur)

        if need == 0:
            print(f"{fn}: 이미 충분 ({cur}/{tgt})")
            continue

        # 간접 표현 20%
        indirect_count = min(need // 5, len(INDIRECT_EXPRESSIONS.get(fn, [])) * 30)
        indirect_data = augment_indirect(fn, indirect_count)
        all_augmented.extend(indirect_data)

        # 나머지는 seed 기반 변형
        rule_count = need - indirect_count
        per_seed = max(1, rule_count // len(seed_list))

        for seed in seed_list:
            augmented = augment_seed(seed, fn, per_seed)
            all_augmented.extend(augmented)

        actual = indirect_count + per_seed * len(seed_list)
        print(f"{fn}: {cur} → +{actual} (seed {len(seed_list)}, indirect {indirect_count})")

    # 저장
    output_path = 'data/train_augmented_v1.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_augmented, f, ensure_ascii=False, indent=2)

    # 통계
    fn_c = Counter(d['labels']['fn'] for d in all_augmented)
    print(f"\n=== 증강 결과: {len(all_augmented)}개 ===")
    for k, v in fn_c.most_common():
        print(f"  {k}: +{v}")


if __name__ == '__main__':
    main()
