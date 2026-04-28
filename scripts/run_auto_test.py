#!/usr/bin/env python3
"""자동 종합 테스트 — 정석/변형/비유/오타/멀티턴/집상태 시나리오 전체 커버.
실행: python3 scripts/run_auto_test.py
"""
import os, sys, re, json
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deployment_pipeline_v2 import DeploymentPipelineV2

pipeline = DeploymentPipelineV2()

# ────────────────────────────────────────────────
# 테스트 정의: (입력, 기대 fn, 기대 dir, 기대 room, 설명)
# room None = 무시, dir None = 무시
# ────────────────────────────────────────────────
SINGLE_TESTS = [
    # ── 정석 직접 명령 ─────────────────────────────────────
    ("거실 불 켜줘",           "light_control",  "on",   "living",       "정석_조명_on"),
    ("거실 불 꺼줘",           "light_control",  "off",  "living",       "정석_조명_off"),
    ("에어컨 켜줘",            "ac_control",     "on",   None,           "정석_에어컨_on"),
    ("에어컨 꺼줘",            "ac_control",     "off",  None,           "정석_에어컨_off"),
    ("난방 켜줘",              "heat_control",   "on",   None,           "정석_난방_on"),
    ("난방 꺼줘",              "heat_control",   "off",  None,           "정석_난방_off"),
    ("환기 켜줘",              "vent_control",   "on",   None,           "정석_환기_on"),
    ("가스 잠궈줘",            "gas_control",    "close",None,           "정석_가스_close"),
    ("도어락 열어줘",          "door_control",   "open", None,           "정석_도어락_open"),
    ("커튼 닫아줘",            "curtain_control","close",None,           "정석_커튼_close"),
    ("오늘 날씨 어때",         "weather_query",  None,   None,           "정석_날씨"),
    ("오늘 뉴스 알려줘",       "news_query",     None,   None,           "정석_뉴스"),

    # ── STT 오타/발음 변형 ────────────────────────────────
    ("에어콘 켜줘",            "ac_control",     "on",   None,           "STT_에어콘"),
    ("에어건 켜줘",            "ac_control",     "on",   None,           "STT_에어건"),
    ("남방 켜줘",              "heat_control",   "on",   None,           "STT_남방"),
    ("완기 틀어줘",            "vent_control",   "on",   None,           "STT_완기"),
    ("조멍 켜줘",              "light_control",  "on",   None,           "STT_조멍"),
    ("거실불 켜줘",            "light_control",  "on",   "living",       "STT_거실불붙임"),
    ("불끄고자",               "light_control",  "off",  None,           "STT_불끄고자"),
    ("도어록 열어",            "door_control",   "open", None,           "STT_도어록"),
    ("엘베 불러줘",            "elevator_call",  "on",   None,           "STT_엘베"),
    ("볼름 높여줘",            "unknown",        None,   None,           "STT_볼름_unknown"),
    ("에어컨은 꺼줘",          "ac_control",     "off",  None,           "STT_에어컨은조사"),
    ("난벙 켜줘",              "heat_control",   "on",   None,           "STT_난벙"),
    ("주방불 꺼",              "light_control",  "off",  "kitchen",      "STT_주방불붙임"),
    ("환풍기 켜줘",            "vent_control",   "on",   None,           "STT_환풍기"),
    ("보일러 올려줘",          "heat_control",   "up",   None,           "STT_보일러_up"),

    # ── 간접/비유 표현 ─────────────────────────────────────
    ("너무 더워",              "ac_control",     "on",   None,           "비유_너무더워"),
    ("너무 추워",              "heat_control",   "on",   None,           "비유_너무추워"),
    ("방이 너무 어두워",        "light_control",  "on",   None,           "비유_어두워"),
    ("눈이 부셔",              "light_control",  "down", None,           "비유_눈부심"),
    ("공기가 탁해",            "vent_control",   "on",   None,           "비유_공기탁해"),
    ("얼어 죽겠어",            "heat_control",   "on",   None,           "비유_얼어죽겠어"),
    ("쪄 죽겠어",              "ac_control",     "on",   None,           "비유_쪄죽겠어"),
    ("냉장고 같아",            "heat_control",   "on",   None,           "비유_냉장고같아"),
    ("사우나야",               "ac_control",     "on",   None,           "비유_사우나"),
    ("눈이 침침해",            "light_control",  "on",   None,           "비유_눈침침"),
    ("머리가 띵해",            "vent_control",   "on",   None,           "비유_머리띵"),
    ("한기가 도네",            "heat_control",   "on",   None,           "비유_한기"),
    ("더위 먹겠다",            "ac_control",     "on",   None,           "비유_더위먹겠다"),
    ("땀이 뻘뻘",              "ac_control",     "on",   None,           "비유_땀뻘뻘"),
    ("눈이 멀겠어",            "light_control",  "down", None,           "비유_눈멀겠어"),
    ("동굴 같아",              "light_control",  "on",   None,           "비유_동굴같아"),
    ("밀폐된 것 같아",         "vent_control",   "on",   None,           "비유_밀폐"),
    ("이가 딱딱 부딪혀",       "heat_control",   "on",   None,           "비유_이딱딱"),
    ("그래도 더워",            "ac_control",     "on",   None,           "비유_그래도더워"),
    ("아직도 추워",            "heat_control",   "on",   None,           "비유_아직도추워"),
    ("여전히 더워",            "ac_control",     "on",   None,           "비유_여전히더워"),

    # ── 강도/설정 ──────────────────────────────────────────
    ("온도 22도로 맞춰줘",     "heat_control",   "set",  None,           "설정_온도22도"),
    ("에어컨 세게 틀어줘",     "ac_control",     "on",   None,           "설정_에어컨세게"),
    ("환기 세게 틀어줘",       "vent_control",   "on",   None,           "설정_환기세게"),
    ("환기 약하게 틀어줘",     "vent_control",   "on",   None,           "설정_환기약하게"),
    ("불 밝게 해줘",           "light_control",  "up",   None,           "설정_불밝게"),
    ("불 어둡게 해줘",         "light_control",  "down", None,           "설정_불어둡게"),
    ("제습모드로 해줘",        "ac_control",     "set",  None,           "설정_제습모드"),

    # ── 부정/취소 표현 ────────────────────────────────────
    ("괜찮아 그냥 둬",         "unknown",        None,   None,           "취소_괜찮아그냥둬"),
    ("됐어",                   "unknown",        None,   None,           "취소_됐어"),
    ("그냥 둬",                "unknown",        None,   None,           "취소_그냥둬"),
    ("에어컨 켜지 마",         "ac_control",     None,   None,           "부정_에어컨켜지마"),

    # ── OOD (Out of Domain) ────────────────────────────────
    ("피자 시켜줘",            "unknown",        None,   None,           "OOD_피자"),
    ("유튜브 틀어줘",          "unknown",        None,   None,           "OOD_유튜브"),
    ("오늘 주식 어때",         "market_query",   None,   None,           "OOD_주식"),
    ("아파트 시세",            "home_info",      None,   None,           "OOD_아파트시세"),

    # ── v77: 한국어 NLU 한계 기반 신규 케이스 ─────────────────
    # 취소/허락 표현
    ("안 꺼도 돼",             "unknown",        None,   None,           "v77_안꺼도돼"),
    ("안 켜도 돼",             "unknown",        None,   None,           "v77_안켜도돼"),
    ("안 끄지 않아도 돼",      "unknown",        None,   None,           "v77_이중부정"),
    ("이거 켜도 돼?",          "unknown",        None,   None,           "v77_켜도돼허락"),
    # 상태 확인 (fn=디바이스, dir=none)
    ("에어컨 켜져 있나요",     "ac_control",     "none", None,           "v77_상태_에어컨켜져"),
    ("불 꺼져 있어요",         "light_control",  "none", None,           "v77_상태_불꺼져"),
    ("난방이 켜져 있나요",     "heat_control",   "none", None,           "v77_상태_난방켜져"),
    # 더위/추위 비유 보강
    ("이 방 왜 이렇게 후텁지근해", "ac_control", "on",   None,           "v77_후텁지근"),
    ("한기가 싸하네",          "heat_control",   "on",   None,           "v77_한기싸하네"),
    # 조도 비유 보강
    ("어둑어둑하게 해줘",      "light_control",  "down", None,           "v77_어둑어둑"),
    # 쾌적/바람 → 환기
    ("쾌적하게 해줘",          "vent_control",   "on",   None,           "v77_쾌적하게"),
    ("잠깐 바람 좀",           "vent_control",   "on",   None,           "v77_바람좀"),
    # OOD 기기 보강
    ("TV 꺼줘",                "unknown",        None,   None,           "v77_TV꺼"),
    ("전기장판 켜줘",          "unknown",        None,   None,           "v77_전기장판"),
    ("창문 열어주시겠어요",    "unknown",        None,   None,           "v77_창문열어"),
    ("잠금 걸어줘",            "door_control",   "close",None,           "v77_잠금걸어"),
    # 혼잣말/자연발화
    ("불 끄고 자야지",         "light_control",  "off",  None,           "v77_자야지"),
    ("에어컨은 끄고 선풍기 켜줘","ac_control",   "off",  None,           "v77_에어컨끄고복합"),
]

# ── 멀티턴 시나리오 ──────────────────────────────────────────
# (발화, 기대fn, 기대dir, 기대room, 설명)
MULTITURN_SCENARIOS = [
    {
        "name": "방 상속 3턴",
        "turns": [
            ("거실 불 켜줘",       "light_control", "on",  "living",  "1턴_거실명시"),
            ("좀 더 밝게 해줘",    "light_control", "up",  "living",  "2턴_거실상속"),
            ("이제 꺼줘",          "light_control", "off", "living",  "3턴_거실상속유지"),
        ]
    },
    {
        "name": "냉난방 충돌 (간접)",
        "turns": [
            ("에어컨 켜줘",        "ac_control",   "on",  None,      "1턴_AC켜기"),
            ("너무 추워",          "heat_control", "on",  None,      "2턴_추워_AC꺼야함"),  # AC off만
        ]
    },
    {
        "name": "냉난방 충돌 (명시)",
        "turns": [
            ("에어컨 켜줘",        "ac_control",   "on",  None,      "1턴_AC켜기"),
            ("난방 켜줘",          "heat_control", "on",  None,      "2턴_난방명시_AC끄고켜기"),
        ]
    },
    {
        "name": "여러 방 순차 제어",
        "turns": [
            ("거실 불 켜줘",       "light_control", "on",  "living",   "1턴_거실"),
            ("안방도 켜줘",        "light_control", "on",  "bedroom_main", "2턴_안방"),
            ("둘 다 꺼줘",         "light_control", "off", None,       "3턴_전체"),  # room='all' or 'none'
        ]
    },
    {
        "name": "DST 리셋 후 다른 방",
        "turns": [
            ("주방 환기 켜줘",     "vent_control",  "on",  "kitchen",  "1턴_주방"),
            ("꺼줘",               "vent_control",  "off", "kitchen",  "2턴_주방상속"),
        ]
    },
    {
        "name": "STT변형 포함 멀티턴",
        "turns": [
            ("안방 에어콘 틀어",       "ac_control",    "on",  "bedroom_main", "1턴_STT에어콘"),
            ("에어컨 쎄게 틀어줘",     "ac_control",    "on",  "bedroom_main", "2턴_명시적으로쎄게"),  # 명시적 에어컨으로 변경
        ]
    },
    {
        "name": "비유 포함 멀티턴",
        "turns": [
            ("거실이 너무 더워",   "ac_control",    "on",  "living",   "1턴_비유더워"),
            ("그래도 더워",        "ac_control",    "on",  "living",   "2턴_그래도더워"),
            ("에어컨 쎄게 틀어줘", "ac_control",    "on",  "living",   "3턴_쎄게"),
        ]
    },
]


# ────────────────────────────────────────────────
def check(r, exp_fn, exp_dir, exp_room):
    ok_fn   = (r['fn'] == exp_fn)
    ok_dir  = (exp_dir is None) or (r['param_direction'] == exp_dir)
    ok_room = (exp_room is None) or (r['room'] == exp_room)
    return ok_fn, ok_dir, ok_room


def run_single_tests():
    print("\n" + "="*60)
    print("■ 단일 발화 테스트")
    print("="*60)
    results = []
    pipeline.reset_dst()
    for text, exp_fn, exp_dir, exp_room, label in SINGLE_TESTS:
        pipeline.reset_dst()
        r = pipeline.process(text, use_dst=False)
        ok_fn, ok_dir, ok_room = check(r, exp_fn, exp_dir, exp_room)
        ok = ok_fn and ok_dir and ok_room
        results.append((ok, label, text, r, exp_fn, exp_dir, exp_room))
        status = "✓" if ok else "✗"
        if not ok:
            detail = []
            if not ok_fn:   detail.append(f"fn={r['fn']} (기대:{exp_fn})")
            if not ok_dir:  detail.append(f"dir={r['param_direction']} (기대:{exp_dir})")
            if not ok_room: detail.append(f"room={r['room']} (기대:{exp_room})")
            print(f"  {status} [{label}] \"{text}\"")
            print(f"      → {', '.join(detail)}")

    pass_n = sum(1 for ok,*_ in results if ok)
    total  = len(results)
    print(f"\n  단일 결과: {pass_n}/{total} ({100*pass_n/total:.1f}%)")
    return results


def run_multiturn_tests():
    print("\n" + "="*60)
    print("■ 멀티턴 시나리오")
    print("="*60)
    all_ok = 0; all_total = 0
    failures = []

    for scenario in MULTITURN_SCENARIOS:
        pipeline.reset_dst()
        print(f"\n  [{scenario['name']}]")
        s_ok = 0
        for text, exp_fn, exp_dir, exp_room, label in scenario['turns']:
            r = pipeline.process(text, use_dst=True)
            ok_fn, ok_dir, ok_room = check(r, exp_fn, exp_dir, exp_room)
            ok = ok_fn and ok_dir and ok_room
            s_ok += ok; all_total += 1; all_ok += ok
            status = "✓" if ok else "✗"
            detail = f"fn={r['fn']}, dir={r['param_direction']}, room={r['room']}"
            print(f"    {status} {label}: \"{text}\" → {detail}")
            if not ok:
                failures.append((scenario['name'], label, text, r, exp_fn, exp_dir, exp_room))

    print(f"\n  멀티턴 결과: {all_ok}/{all_total} ({100*all_ok/all_total:.1f}%)")
    return failures


if __name__ == '__main__':
    single_results = run_single_tests()
    mt_failures    = run_multiturn_tests()

    # 실패 요약
    single_fails = [(l,t,r,ef,ed,er) for ok,l,t,r,ef,ed,er in single_results if not ok]
    print("\n" + "="*60)
    print(f"■ 전체 요약")
    print(f"  단일 실패: {len(single_fails)}건")
    print(f"  멀티턴 실패: {len(mt_failures)}건")

    if single_fails:
        print("\n  단일 실패 목록:")
        for label, text, r, ef, ed, er in single_fails:
            print(f"    [{label}] \"{text}\"")
            print(f"      실제: fn={r['fn']}, dir={r['param_direction']}, room={r['room']}")
            print(f"      기대: fn={ef}, dir={ed}, room={er}")
