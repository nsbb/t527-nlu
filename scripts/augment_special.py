#!/usr/bin/env python3
"""특수 데이터 생성: judge 보충, clarify, 오류패턴 수정 데이터"""
import json, random

random.seed(42)

# ============================================================
# 1. Judge 보충 (weather_query + query_then_judge)
# ============================================================
JUDGE_TEMPLATES = {
    'outdoor_activity': [
        "오늘 {activity}해도 돼?", "이번 주말 {activity} 가도 되나?",
        "지금 {activity}하기 좋아?", "{activity} 괜찮을까?",
        "오늘 밖에 나가도 돼?", "야외활동 해도 될까?",
        "비 올까?", "우산 가져가야 해?",
        "{activity} 일정 잡아도 될까?", "외출해도 괜찮아?",
    ],
    'clothing': [
        "오늘 뭐 입고 나가?", "뭐 입을까?", "겉옷 필요해?",
        "코트 입어야 해?", "반팔 입어도 돼?", "옷 어떻게 입혀?",
        "아이 옷 뭐 입혀?", "자켓 필요할까?", "외투 챙겨야 해?",
        "반바지 입어도 될까?", "두꺼운 옷 입어야 해?", "얇게 입어도 돼?",
    ],
    'air_quality': [
        "창문 열어도 괜찮아?", "환기해도 돼?", "미세먼지 어때?",
        "공기 괜찮아?", "마스크 써야 해?", "창문 열어도 될까?",
        "공기질 어때?", "먼지 많아?", "밖에 공기 좋아?",
        "아이 밖에 내보내도 돼?", "빨래 밖에 널어도 돼?",
    ],
    'cost_trend': [
        "오늘 주유해도 되나?", "기름값 떨어질까?", "주유 지금 해도 돼?",
        "유가 어떻게 될까?", "기다리면 떨어질까?",
    ],
}

ACTIVITIES = ['세차', '조깅', '산책', '등산', '캠핑', '소풍', '운동', '자전거', '낚시', '골프']

def gen_judge_data():
    results = []
    for judge_type, templates in JUDGE_TEMPLATES.items():
        target = 150 if judge_type != 'cost_trend' else 80
        for _ in range(target):
            tmpl = random.choice(templates)
            if '{activity}' in tmpl:
                tmpl = tmpl.replace('{activity}', random.choice(ACTIVITIES))
            results.append({
                'utterance': tmpl,
                'flat_intent': f'weather_{judge_type}',
                'labels': {
                    'fn': 'weather_query',
                    'exec_type': 'query_then_judge',
                    'param_direction': 'none',
                    'param_type': 'none',
                    'judge': judge_type,
                },
                'source': 'special_judge'
            })
    return results


# ============================================================
# 2. Clarify 데이터
# ============================================================
def gen_clarify_data():
    templates = [
        # 공간 미지정
        ("불 켜줘", "light_control"), ("불 꺼줘", "light_control"),
        ("조명 켜", "light_control"), ("조명 좀", "light_control"),
        ("난방 올려줘", "heat_control"), ("난방 좀", "heat_control"),
        ("에어컨 켜줘", "ac_control"), ("에어컨 틀어", "ac_control"),
        ("송풍해줘", "ac_control"), ("제습 해줘", "ac_control"),
        ("커튼 열어줘", "curtain_control"), ("커튼 닫아", "curtain_control"),
        # 기기 미지정
        ("전부 켜", "home_info"), ("다 꺼줘", "home_info"),
        ("전체 다 켜", "home_info"), ("싹 다 꺼", "home_info"),
    ]
    results = []
    endings = ['', '줘', '봐', '주세요', '줄래']
    for _ in range(200):
        base, fn = random.choice(templates)
        # 어미 변형
        for old, new in [('줘', random.choice(endings)), ('해', random.choice(['해', '해줘', '해봐']))]:
            if base.endswith(old):
                base = base[:-len(old)] + new
                break
        results.append({
            'utterance': base,
            'flat_intent': f'{fn}_clarify',
            'labels': {
                'fn': fn,
                'exec_type': 'clarify',
                'param_direction': 'none',
                'param_type': 'none',
                'judge': 'none',
            },
            'source': 'special_clarify'
        })
    return results


# ============================================================
# 3. 오류 패턴 수정 데이터 (르엘 평가에서 발견된 혼동 패턴)
# ============================================================
def gen_confusion_fixes():
    """르엘 평가에서 틀린 패턴 → 정답 데이터 추가"""
    fixes = [
        # "~예약 있어?" → device_query (schedule이 아닌 해당 디바이스)
        ("난방 예약 있어?", "heat_control", "query_then_respond"),
        ("난방 예약 확인해줘", "heat_control", "query_then_respond"),
        ("난방 예약 뭐 있어", "heat_control", "query_then_respond"),
        ("난방 예약 어떻게 돼있어", "heat_control", "query_then_respond"),
        ("조명 예약 있어?", "light_control", "query_then_respond"),
        ("조명 예약 확인해줘", "light_control", "query_then_respond"),
        ("조명 예약 뭐있어", "light_control", "query_then_respond"),
        ("에어컨 예약 있어?", "ac_control", "query_then_respond"),
        ("에어컨 예약 확인", "ac_control", "query_then_respond"),
        ("환기 예약 있어?", "vent_control", "query_then_respond"),
        ("환기 예약 확인해줘", "vent_control", "query_then_respond"),
        ("커튼 예약 있어?", "curtain_control", "query_then_respond"),
        # "취침모드 예약" → light_control
        ("취침모드 예약 어떻게 되어있어", "light_control", "query_then_respond"),
        ("취침모드 설정 확인해줘", "light_control", "query_then_respond"),
        # weather_query 판단형 — "세차/창문/환기해도 돼?" = weather, not device
        ("세차해도 돼?", "weather_query", "query_then_judge"),
        ("세차 가도 되나", "weather_query", "query_then_judge"),
        ("세차해도 되겠어?", "weather_query", "query_then_judge"),
        ("세차하기 좋아?", "weather_query", "query_then_judge"),
        ("창문 열어도 괜찮아?", "weather_query", "query_then_judge"),
        ("창문 열어도 되나", "weather_query", "query_then_judge"),
        ("창문 열어놔도 돼?", "weather_query", "query_then_judge"),
        ("환기해도 돼?", "weather_query", "query_then_judge"),
        ("환기해도 되나", "weather_query", "query_then_judge"),
        ("환기시켜도 괜찮아?", "weather_query", "query_then_judge"),
        # "밖에 더워?" = weather query, not ac_control
        ("지금 밖에 더워?", "weather_query", "query_then_respond"),
        ("밖에 춥나?", "weather_query", "query_then_respond"),
        ("밖에 더운가?", "weather_query", "query_then_respond"),
        ("바깥 날씨 더워?", "weather_query", "query_then_respond"),
        # home_info 경계
        ("볼륨 조절할 수 있어?", "home_info", "direct_respond"),
        ("볼륨 조절 가능해?", "home_info", "direct_respond"),
        ("소리 조절 돼?", "home_info", "direct_respond"),
        ("전기요금 아끼게 해줘", "home_info", "direct_respond"),
        ("절전 모드 해줘", "home_info", "direct_respond"),
        # 시베리아 = heat_control
        ("시베리아야", "heat_control", "control_then_confirm"),
        ("시베리아인줄", "heat_control", "control_then_confirm"),
        ("북극이야", "heat_control", "control_then_confirm"),
        ("냉동실이야", "heat_control", "control_then_confirm"),
    ]

    results = []
    # 각 패턴을 5~10번 반복 (어미 변형 포함)
    for utt, fn, exec_type in fixes:
        for _ in range(8):
            v = utt
            # 어미 변형
            if random.random() < 0.4:
                for old, new_list in [('해줘', ['해봐', '해주세요', '해']),
                                       ('있어?', ['있나?', '있어', '있니']),
                                       ('돼?', ['될까?', '되나?', '돼', '되겠어?']),
                                       ('괜찮아?', ['괜찮을까?', '괜찮나?', '괜찮겠어?'])]:
                    if old in v:
                        v = v.replace(old, random.choice(new_list))
                        break

            direction = 'none'
            if exec_type == 'control_then_confirm':
                direction = 'on'
            judge = 'none'
            if exec_type == 'query_then_judge':
                for kw, j in [('세차', 'outdoor_activity'), ('창문', 'air_quality'),
                              ('환기', 'air_quality'), ('옷', 'clothing'), ('입고', 'clothing'),
                              ('주유', 'cost_trend')]:
                    if kw in utt:
                        judge = j
                        break

            results.append({
                'utterance': v,
                'flat_intent': f'{fn}_{exec_type}',
                'labels': {
                    'fn': fn,
                    'exec_type': exec_type,
                    'param_direction': direction,
                    'param_type': 'none',
                    'judge': judge,
                },
                'source': 'special_confusion_fix'
            })
    return results


def main():
    judge_data = gen_judge_data()
    clarify_data = gen_clarify_data()
    confusion_data = gen_confusion_fixes()

    all_special = judge_data + clarify_data + confusion_data

    with open('data/train_special.json', 'w', encoding='utf-8') as f:
        json.dump(all_special, f, ensure_ascii=False, indent=2)

    from collections import Counter
    src_c = Counter(d['source'] for d in all_special)
    print(f"Special data: {len(all_special)}개")
    for k, v in src_c.most_common():
        print(f"  {k}: {v}")

    fn_c = Counter(d['labels']['fn'] for d in all_special)
    print(f"\nfn distribution:")
    for k, v in fn_c.most_common():
        print(f"  {k}: {v}")

    judge_c = Counter(d['labels']['judge'] for d in all_special)
    print(f"\njudge distribution:")
    for k, v in judge_c.most_common():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
