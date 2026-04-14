#!/usr/bin/env python3
"""시나리오당 100개 균등 증강
204개 known 시나리오 × 100개 = ~20,400개
"""
import json, random, re
from collections import Counter

random.seed(42)

# ============================================================
# 변형 규칙
# ============================================================
VERB_VARIANTS = {
    '켜줘': ['켜줘','켜봐','켜라','켜주세요','켜줄래','켜','좀 켜줘','켜줘요','켜주실래요'],
    '꺼줘': ['꺼줘','꺼봐','꺼라','꺼주세요','꺼줄래','꺼','좀 꺼줘','꺼줘요'],
    '열어줘': ['열어줘','열어봐','열어라','열어주세요','열어줄래','열어','좀 열어줘'],
    '닫아줘': ['닫아줘','닫아봐','닫아라','닫아주세요','닫아줄래','닫아','좀 닫아줘'],
    '올려줘': ['올려줘','올려봐','올려주세요','올려줄래','올려','좀 올려줘','높여줘'],
    '낮춰줘': ['낮춰줘','낮춰봐','낮춰주세요','낮춰줄래','낮춰','좀 낮춰줘','줄여줘'],
    '맞춰줘': ['맞춰줘','맞춰봐','맞춰주세요','맞춰줄래','설정해줘','해줘','맞춰'],
    '해줘': ['해줘','해봐','해라','해주세요','해줄래','해줘요','해','좀 해줘'],
    '해 줘': ['해 줘','해봐','해 주세요','해줄래','해줘요','해','좀 해줘'],
    '알려줘': ['알려줘','알려봐','알려주세요','알려줄래','말해줘','알려줘요','알려'],
    '어때': ['어때','어때?','어떠니','어떠냐','어떤가요','어떻게 돼','어때요'],
    '있어': ['있어','있어?','있나','있니','있나요','있음?'],
    '있어?': ['있어?','있나?','있니?','있나요?','있음?'],
    '뭐야': ['뭐야','뭐야?','뭐니','뭐냐','뭔가요','뭐예요','뭐임'],
    '뭐야?': ['뭐야?','뭐니?','뭐냐?','뭔가요?','뭐예요?'],
    '얼마야': ['얼마야','얼마야?','얼마냐','얼맨가요','얼마예요'],
    '어디': ['어디','어디야','어딨어','어디에','어디 있어'],
    '불러줘': ['불러줘','불러봐','불러주세요','불러','호출해줘','호출해'],
    '찾아줘': ['찾아줘','찾아봐','찾아주세요','검색해줘','알아봐줘'],
    '틀어줘': ['틀어줘','틀어봐','틀어주세요','틀어','좀 틀어줘'],
    '확인': ['확인','확인해줘','확인해봐','확인해주세요','체크해줘','봐줘'],
}

ROOMS = ['거실','안방','주방','침실','작은방','아이방']
ROOM_PARTICLES = ['','에','의',' ']

STT_ERRORS = {
    '에어컨':['에어콘','애어컨'],'난방':['남방','난반'],'환기':['한기','환기'],
    '커튼':['커턴'],'엘리베이터':['엘레베이터','엘베'],'조명':['조멍'],
    '가스':['까스'],'밸브':['벨브','밸부'],'도어락':['도어록','도얼락'],
    '볼륨':['볼름','볼류'],'비밀번호':['비번','비밀번'],'모닝콜':['몬잉콜','모닝꼴'],
}

INDIRECT = {
    'ac_control': [
        ('찜통이야','on'),('너무 더워','on'),('덥다','on'),('땀이 줄줄','on'),
        ('시원하게 좀','on'),('열대야네','on'),('더워 죽겠다','on'),
        ('에어컨 좀','on'),('냉방 좀','on'),
    ],
    'heat_control': [
        ('좀 춥다','on'),('너무 추워','on'),('얼어 죽겠다','on'),('시베리아야','on'),
        ('따뜻하게 좀','on'),('손 시려','on'),('발 시려','on'),('한기가 든다','on'),
        ('추워 추워','on'),('북극이야','on'),('냉동실이야','on'),
    ],
    'light_control': [
        ('어두워','on'),('안 보여','on'),('깜깜해','on'),('동굴이야','on'),
        ('눈이 부셔','down'),('너무 밝아','down'),('눈부셔','down'),
    ],
    'vent_control': [
        ('공기가 탁해','on'),('답답해','on'),('냄새 나','on'),('환기 좀','on'),
        ('숨이 막혀','on'),('시끄러워','off'),
    ],
    'security_mode': [
        ('나 나간다','on'),('외출할게','on'),('출근한다','on'),('잠깐 나갔다 올게','on'),
        ('나 들어왔어','set'),('다녀왔습니다','set'),
    ],
    'elevator_call': [
        ('내려갈 건데','on'),('나가야 하는데','on'),('엘베 좀','on'),
    ],
}

PREFIXES = ['','좀 ','지금 ','빨리 ','혹시 ','']
SUFFIXES = ['',' 좀',' 지금',' 바로',' 해줘','']


def apply_verb_variant(text):
    for pattern, variants in VERB_VARIANTS.items():
        if pattern in text:
            return text.replace(pattern, random.choice(variants), 1)
    return text


def apply_room_swap(text):
    for room in ROOMS:
        if room in text:
            new_room = random.choice(ROOMS)
            particle = random.choice(ROOM_PARTICLES)
            return text.replace(room, new_room + particle, 1)
    return text


def apply_room_add(text, fn):
    device_fns = ['light_control','heat_control','ac_control','vent_control','curtain_control']
    if fn not in device_fns:
        return text
    if any(r in text for r in ROOMS):
        return text
    room = random.choice(ROOMS)
    particle = random.choice(['','에','의',' '])
    return f"{room}{particle} {text}"


def apply_stt_error(text):
    result = text
    for correct, errors in STT_ERRORS.items():
        if correct in result and random.random() < 0.2:
            result = result.replace(correct, random.choice(errors), 1)
    return result


def apply_politeness(text):
    # 반말화
    for p, c in [('주세요','줘'),('하세요','해'),('줄래요','줄래'),('해주세요','해줘')]:
        if p in text:
            return text.replace(p, c)
    # 존댓말화
    if text.endswith('줘'):
        return text[:-1] + '주세요'
    if text.endswith('해'):
        return text + '주세요'
    return text


def generate_variants(seed_utt, fn, labels, count=100):
    results = []
    base = seed_utt

    # 간접 표현 (있으면 20% 할당)
    indirect_pool = INDIRECT.get(fn, [])
    indirect_count = min(count // 5, len(indirect_pool) * 5) if indirect_pool else 0
    rule_count = count - indirect_count

    # 간접 표현 생성
    for _ in range(indirect_count):
        expr, d = random.choice(indirect_pool)
        utt = expr
        if random.random() < 0.4:
            utt = apply_room_add(utt, fn)
        if random.random() < 0.15:
            utt = apply_stt_error(utt)
        utt = re.sub(r'\s+', ' ', utt).strip()
        new_labels = dict(labels)
        new_labels['param_direction'] = d
        results.append((utt, new_labels))

    # 규칙 기반 변형
    for _ in range(rule_count):
        utt = base
        r = random.random()
        if r < 0.25:
            utt = apply_verb_variant(utt)
        elif r < 0.40:
            utt = apply_room_swap(utt)
        elif r < 0.50:
            utt = apply_room_add(utt, fn)
        elif r < 0.60:
            utt = apply_politeness(utt)
        elif r < 0.70:
            utt = apply_stt_error(utt)
        elif r < 0.80:
            utt = apply_verb_variant(utt)
            utt = apply_room_swap(utt)
        elif r < 0.90:
            utt = apply_verb_variant(utt)
            utt = apply_stt_error(utt)
        else:
            utt = apply_politeness(utt)
            utt = apply_room_add(utt, fn)

        # 접두/접미
        if random.random() < 0.2:
            utt = random.choice(PREFIXES) + utt
        if random.random() < 0.15:
            utt = utt + random.choice(SUFFIXES)

        utt = re.sub(r'\s+', ' ', utt).strip()
        if utt:
            results.append((utt, labels))

    return results


def main():
    with open('data/gt_known_scenarios.json', encoding='utf-8') as f:
        scenarios = json.load(f)

    PER_SCENARIO = 100
    all_data = []

    for s in scenarios:
        fn = s['labels']['fn']
        labels = s['labels']
        utt = s['utterance']

        # OOO, 000 같은 플레이스홀더 처리
        clean_utt = utt
        clean_utt = re.sub(r'\(.*?\)', '', clean_utt)  # (공간명 지정이 없을 때) 제거
        clean_utt = re.sub(r'OO+', '', clean_utt)
        clean_utt = re.sub(r'0{3,}', '', clean_utt)
        clean_utt = re.sub(r'\s+', ' ', clean_utt).strip()

        if not clean_utt or len(clean_utt) < 2:
            clean_utt = utt  # fallback

        # seed 자체도 포함
        all_data.append({
            'utterance': clean_utt,
            'scenario_id': s['scenario_id'],
            'labels': labels,
            'source': 'gt_seed',
        })

        # 변형 생성
        variants = generate_variants(clean_utt, fn, labels, PER_SCENARIO - 1)
        for v_utt, v_labels in variants:
            all_data.append({
                'utterance': v_utt,
                'scenario_id': s['scenario_id'],
                'labels': v_labels,
                'source': 'gt_augment',
            })

    # 저장
    with open('data/train_gt_augmented.json', 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    # 통계
    fn_c = Counter(d['labels']['fn'] for d in all_data)
    exec_c = Counter(d['labels']['exec_type'] for d in all_data)
    dir_c = Counter(d['labels']['param_direction'] for d in all_data)
    judge_c = Counter(d['labels']['judge'] for d in all_data)

    print(f"총 {len(all_data)}개 생성 ({len(scenarios)} scenarios × ~{PER_SCENARIO})")
    print(f"\n--- fn ({len(fn_c)}개) ---")
    for k, v in fn_c.most_common():
        print(f"  {k:20s}: {v:5d} ({v/len(all_data)*100:.1f}%)")
    print(f"\n--- exec ---")
    for k, v in exec_c.most_common():
        print(f"  {k}: {v}")
    print(f"\n--- direction ---")
    for k, v in dir_c.most_common():
        print(f"  {k}: {v}")
    print(f"\n--- judge ---")
    for k, v in judge_c.most_common():
        print(f"  {k}: {v}")

    # 시나리오당 수 확인
    sid_c = Counter(d['scenario_id'] for d in all_data)
    print(f"\n시나리오당: min={min(sid_c.values())}, max={max(sid_c.values())}, avg={sum(sid_c.values())/len(sid_c):.0f}")


if __name__ == '__main__':
    main()
