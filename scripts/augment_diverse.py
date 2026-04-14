#!/usr/bin/env python3
"""고다양성 증강 — 시나리오당 최소 50개 유니크 목표
기존 augment_per_scenario.py의 변형 규칙 대폭 강화
"""
import json, random, re
from collections import Counter

random.seed(42)

# ============================================================
# 1. 어미 변형 (더 많은 패턴)
# ============================================================
VERB_PATTERNS = [
    # 동사 어미
    ('켜줘', ['켜줘','켜봐','켜주세요','켜줄래','켜줄래?','켜','키다','좀 켜줘','켜줘요','켜주실래요','켜주실래','켜주라','켜달라','켜달라고','켜놔','켜놓아']),
    ('꺼줘', ['꺼줘','꺼봐','꺼주세요','꺼줄래','꺼','끄다','좀 꺼줘','꺼줘요','꺼주실래요','꺼달라','꺼놔','꺼놓아']),
    ('열어줘', ['열어줘','열어봐','열어주세요','열어줄래','열어','좀 열어줘','열어줘요','열어달라','열어놔']),
    ('닫아줘', ['닫아줘','닫아봐','닫아주세요','닫아줄래','닫아','좀 닫아줘','닫아달라','닫아놔']),
    ('올려줘', ['올려줘','올려봐','올려주세요','올려줄래','올려','좀 올려줘','높여줘','높여봐','올려주라']),
    ('낮춰줘', ['낮춰줘','낮춰봐','낮춰주세요','낮춰줄래','낮춰','좀 낮춰줘','줄여줘','줄여봐']),
    ('맞춰줘', ['맞춰줘','맞춰봐','맞춰주세요','설정해줘','설정해봐','맞춰줄래','해줘','맞춰']),
    ('해줘', ['해줘','해봐','해주세요','해줄래','해줘요','해','좀 해줘','해주라','해달라']),
    ('해 줘', ['해 줘','해봐','해 주세요','해줄래','해줘요','해','좀 해줘']),
    ('불러줘', ['불러줘','불러봐','불러주세요','호출해줘','호출해','불러','좀 불러줘']),
    ('찾아줘', ['찾아줘','찾아봐','찾아주세요','검색해줘','알아봐줘','찾아','좀 찾아줘']),
    ('틀어줘', ['틀어줘','틀어봐','틀어주세요','틀어','좀 틀어줘','틀어달라']),
    ('알려줘', ['알려줘','알려봐','알려주세요','알려줄래','말해줘','알려','가르쳐줘']),
    # 의문 어미
    ('어때', ['어때','어때?','어떠니','어떠냐','어떤가요','어떻게 돼','어때요','어떠세요','어떻습니까']),
    ('어때?', ['어때?','어떠니?','어떠냐?','어떤가요?','어떻게 돼?','어때요?']),
    ('있어?', ['있어?','있나?','있니?','있나요?','있음?','있어','있습니까?','있냐?']),
    ('있어', ['있어','있나','있니','있나요','있음?','있어?']),
    ('뭐야', ['뭐야','뭐야?','뭐니','뭐냐','뭔가요','뭐예요','뭐임','무엇인가요']),
    ('뭐야?', ['뭐야?','뭐니?','뭐냐?','뭔가요?','뭐예요?','뭐임?']),
    ('얼마야', ['얼마야','얼마야?','얼마냐','얼마인가요','얼마예요','얼마임']),
    ('얼마야?', ['얼마야?','얼마냐?','얼마인가요?','얼마예요?']),
    ('어디', ['어디','어디야','어딨어','어디에','어디 있어']),
    ('돼?', ['돼?','될까?','되나?','돼','되겠어?','괜찮아?','괜찮을까?']),
    ('괜찮아?', ['괜찮아?','괜찮을까?','괜찮나?','되나?','돼?']),
    ('할 수 있어?', ['할 수 있어?','가능해?','가능한가요?','할 수 있나?','되나?']),
]

# 2. 조사 변형
PARTICLE_VARIANTS = [
    ('을 ', ['을 ','를 ',' ']),
    ('를 ', ['를 ','을 ',' ']),
    ('에서 ', ['에서 ','에 ','서 ',' ']),
    ('으로 ', ['으로 ','로 ',' ']),
    ('이 ', ['이 ','가 ',' ']),
    ('가 ', ['가 ','이 ',' ']),
    ('은 ', ['은 ','는 ',' ']),
    ('는 ', ['는 ','은 ',' ']),
]

# 3. 동의어/유의어
SYNONYMS = {
    '에어컨': ['에어컨','냉방','냉방기','에어콘','AC'],
    '난방': ['난방','히터','보일러','온돌'],
    '조명': ['조명','라이트','전등','등'],
    '불': ['불','조명','라이트','전등'],
    '환기': ['환기','공기순환','공기청정','환풍'],
    '커튼': ['커튼','블라인드','전동커튼'],
    '엘리베이터': ['엘리베이터','엘베','승강기','리프트'],
    '도어락': ['도어락','현관문','현관','도어'],
    '볼륨': ['볼륨','소리','음량','사운드'],
    '날씨': ['날씨','기상','기온','날'],
    '온도': ['온도','기온','템퍼','체감온도'],
    '켜져': ['켜져','작동하고','돌아가고','동작하고'],
    '꺼져': ['꺼져','멈춰','중단되어'],
    '밝기': ['밝기','광량','조도','빛'],
    '뉴스': ['뉴스','소식','기사','보도'],
    '주식': ['주식','증시','주가','증권'],
    '유가': ['유가','기름값','유류','연료비'],
    '병원': ['병원','의원','클리닉','진료소'],
    '미세먼지': ['미세먼지','공기질','대기질','먼지'],
}

# 4. 문장 패턴 변형 (의문→명령, 명령→의문 등)
SENTENCE_PATTERNS = {
    # "X 어때?" → "X 알려줘" / "X 확인해줘"
    '어때': ['어때','좀 알려줘','확인해줘','어떤지 봐줘','상태 좀'],
    # "X 있어?" → "X 확인해줘" / "X 알려줘"
    '있어?': ['있어?','있는지 확인해줘','있는지 알려줘','있나 봐줘'],
}

# 5. Room 변형
ROOMS = ['거실','안방','주방','침실','작은방','아이방']
ROOM_PARTICLES = ['','에','의',' ']

# 6. 접두/접미어
PREFIXES = ['','좀 ','지금 ','빨리 ','혹시 ','야 ','어이 ','제발 ','','','']
SUFFIXES = ['',' 좀',' 지금',' 바로',' 해줘',' 제발','','','']

# 7. STT 오류
STT_MAP = {
    '에어컨':['에어콘','애어컨'],'난방':['남방','난반'],'환기':['한기'],
    '커튼':['커턴'],'엘리베이터':['엘레베이터','엘베'],'조명':['조멍'],
    '가스':['까스'],'밸브':['벨브','밸부'],'도어락':['도어록','도얼락'],
    '볼륨':['볼름'],'모닝콜':['몬잉콜'],'비밀번호':['비번'],
}


def apply_random_transform(text, fn):
    """랜덤하게 1~3개 변형 동시 적용"""
    transforms = random.sample([
        'verb', 'particle', 'synonym', 'prefix', 'suffix',
        'stt', 'room_swap', 'room_add', 'sentence_pattern',
        'word_drop', 'word_repeat',
    ], k=random.randint(1, 3))

    for t in transforms:
        if t == 'verb':
            for pattern, variants in VERB_PATTERNS:
                if pattern in text:
                    text = text.replace(pattern, random.choice(variants), 1)
                    break

        elif t == 'particle':
            p, variants = random.choice(PARTICLE_VARIANTS)
            if p in text:
                text = text.replace(p, random.choice(variants), 1)

        elif t == 'synonym':
            for word, syns in SYNONYMS.items():
                if word in text:
                    text = text.replace(word, random.choice(syns), 1)
                    break

        elif t == 'prefix':
            text = random.choice(PREFIXES) + text

        elif t == 'suffix':
            text = text + random.choice(SUFFIXES)

        elif t == 'stt':
            for correct, errors in STT_MAP.items():
                if correct in text and random.random() < 0.3:
                    text = text.replace(correct, random.choice(errors), 1)
                    break

        elif t == 'room_swap':
            for room in ROOMS:
                if room in text:
                    new_room = random.choice(ROOMS)
                    p = random.choice(ROOM_PARTICLES)
                    text = text.replace(room, new_room + p, 1)
                    break

        elif t == 'room_add':
            if fn in ['light_control','heat_control','ac_control','vent_control','curtain_control']:
                if not any(r in text for r in ROOMS):
                    room = random.choice(ROOMS)
                    p = random.choice(ROOM_PARTICLES)
                    text = f"{room}{p} {text}"

        elif t == 'sentence_pattern':
            for pat, variants in SENTENCE_PATTERNS.items():
                if pat in text:
                    text = text.replace(pat, random.choice(variants), 1)
                    break

        elif t == 'word_drop':
            words = text.split()
            if len(words) > 2:
                idx = random.randint(0, len(words)-1)
                words.pop(idx)
                text = ' '.join(words)

        elif t == 'word_repeat':
            words = text.split()
            if words:
                idx = random.randint(0, len(words)-1)
                words.insert(idx, words[idx])
                text = ' '.join(words)

    return re.sub(r'\s+', ' ', text).strip()


def augment_scenario(seed_utt, fn, labels, target=100):
    """다양한 변형 생성 — 유니크 목표 달성까지 반복"""
    unique_set = {seed_utt}
    results = [(seed_utt, labels)]

    # 간접 표현 (있으면)
    INDIRECT = {
        'ac_control': [('찜통이야','on'),('너무 더워','on'),('덥다','on'),('시원하게 좀','on'),('열대야네','on'),('더워 죽겠다','on'),('땀이 줄줄','on')],
        'heat_control': [('좀 춥다','on'),('너무 추워','on'),('얼어 죽겠다','on'),('시베리아야','on'),('따뜻하게 좀','on'),('손 시려','on'),('북극이야','on'),('냉동실이야','on')],
        'light_control': [('어두워','on'),('안 보여','on'),('깜깜해','on'),('동굴이야','on'),('눈이 부셔','down'),('너무 밝아','down')],
        'vent_control': [('공기가 탁해','on'),('답답해','on'),('냄새 나','on'),('숨이 막혀','on'),('시끄러워','off')],
        'security_mode': [('나 나간다','on'),('외출할게','on'),('출근한다','on'),('나 들어왔어','set'),('다녀왔습니다','set')],
        'elevator_call': [('내려갈 건데','on'),('나가야 하는데','on'),('엘베 좀','on')],
    }

    indirect_pool = INDIRECT.get(fn, [])
    for expr, d in indirect_pool:
        new_labels = dict(labels)
        new_labels['param_direction'] = d
        for room in [''] + ROOMS[:3]:
            utt = f"{room} {expr}".strip() if room else expr
            if utt not in unique_set:
                unique_set.add(utt)
                results.append((utt, new_labels))

    # 반복 변형 — 유니크 목표까지
    max_attempts = target * 10
    attempts = 0
    while len(unique_set) < target and attempts < max_attempts:
        utt = apply_random_transform(seed_utt, fn)
        if utt and utt not in unique_set and len(utt) > 1:
            unique_set.add(utt)
            results.append((utt, labels))
        attempts += 1

    return results[:target]


def main():
    with open('data/gt_known_scenarios.json', encoding='utf-8') as f:
        scenarios = json.load(f)

    PER_SCENARIO = 100
    all_data = []

    for s in scenarios:
        fn = s['labels']['fn']
        labels = s['labels']
        utt = s['utterance']
        utt = re.sub(r'\(.*?\)', '', utt)
        utt = re.sub(r'OO+', '', utt)
        utt = re.sub(r'0{3,}', '', utt)
        utt = re.sub(r'\s+', ' ', utt).strip()
        if len(utt) < 2: utt = s['utterance']

        variants = augment_scenario(utt, fn, labels, PER_SCENARIO)
        for v_utt, v_labels in variants:
            all_data.append({
                'utterance': v_utt,
                'scenario_id': s['scenario_id'],
                'labels': v_labels,
                'source': 'gt_augment_v2',
            })

    # 시나리오당 유니크 확인
    by_sid = {}
    for d in all_data:
        sid = d['scenario_id']
        if sid not in by_sid: by_sid[sid] = set()
        by_sid[sid].add(d['utterance'].strip())

    uniq_counts = [len(v) for v in by_sid.values()]
    print(f"총 {len(all_data)}개 생성 ({len(scenarios)} scenarios)")
    print(f"유니크: min={min(uniq_counts)}, max={max(uniq_counts)}, avg={sum(uniq_counts)/len(uniq_counts):.0f}")
    print(f"유니크 <20: {sum(1 for u in uniq_counts if u < 20)}개")
    print(f"유니크 >=50: {sum(1 for u in uniq_counts if u >= 50)}개")

    with open('data/train_gt_augmented_v2.json', 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    fn_c = Counter(d['labels']['fn'] for d in all_data)
    print(f"\n--- fn ---")
    for k, v in fn_c.most_common():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
