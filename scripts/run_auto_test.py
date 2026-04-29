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
    ("볼름 높여줘",            "home_info",      "up",   None,           "STT_볼름_preprocess_home_info"),
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
    ("눈이 침침해",            "light_control",  "up",   None,           "v92_눈침침_up"),
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
    # 상태 확인 → fn=디바이스 유지, exec=query, dir=none (v77 rule)
    ("에어컨 켜져 있나요",     "ac_control",     None,   None,           "v90_상태_에어컨켜져"),
    ("불 꺼져 있어요",         "light_control",  None,   None,           "v90_상태_불꺼져"),
    ("난방이 켜져 있나요",     "heat_control",   None,   None,           "v90_상태_난방켜져"),
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

    # ── v80: 시간 표현 + 기기 → schedule_manage 오예측 교정 ────
    ("5분 후에 불 꺼줘",       "light_control",  "off",  None,           "v80_5분후불"),
    ("30분 있다가 에어컨 꺼줘","ac_control",     "off",  None,           "v80_30분있다가에어컨"),
    ("잠시 후에 불 꺼줘",      "light_control",  "off",  None,           "v80_잠시후불"),
    ("1시간 후에 난방 꺼줘",   "heat_control",   "off",  None,           "v80_1시간후난방"),
    ("10분 뒤에 환기 꺼줘",    "vent_control",   "off",  None,           "v80_10분뒤환기"),

    # ── v81: 동음이의어/미지원기기/조건부 교정 ───────────────────
    ("TV 틀어줘",              "unknown",        None,   None,           "v81_TV틀어"),
    ("선풍기 틀어줘",          "unknown",        None,   None,           "v81_선풍기틀어"),
    ("비 오면 창문 닫아줘",    "unknown",        None,   None,           "v81_비오면창문"),

    # ── v83: 신체감각 온도/습도 표현 + 잠가야겠다 ──────────────────
    ("쌀쌀하다",               "heat_control",   "on",   None,           "v83_쌀쌀"),
    ("서늘해",                 "heat_control",   "on",   None,           "v83_서늘"),
    ("으슬으슬",               "heat_control",   "on",   None,           "v83_으슬으슬"),
    ("끈적끈적",               "ac_control",     "on",   None,           "v83_끈적끈적"),
    ("땀이 나",                "ac_control",     "on",   None,           "v83_땀이나"),
    ("문이나 잠가야겠다",      "door_control",   "close",None,           "v83_잠가야겠다"),

    # ── v84: 소원형/부정 표현 교정 ────────────────────────────────
    ("따뜻해졌으면 해",        "heat_control",   "on",   None,           "v84_따뜻해졌으면"),
    ("좀 시원했으면 좋겠어",   "ac_control",     "on",   None,           "v84_시원했으면"),
    ("어둡게 하지 말아줘",     "light_control",  "on",   None,           "v84_어둡게하지마"),
    ("서늘해지게 해줘",        "ac_control",     "on",   None,           "v84_서늘해지게"),

    # ── v85: 온도질의/가스누출/기기감탄 교정 ───────────────────────
    ("지금 몇 도야",           "home_info",      None,   None,           "v85_지금몇도"),
    ("현재 온도 알려줘",       "home_info",      None,   None,           "v85_현재온도"),
    ("습도 얼마야",            "home_info",      None,   None,           "v85_습도얼마"),
    ("가스 새는 것 같아",      "gas_control",    "close",None,           "v85_가스새는"),
    ("난방이 빨리 드네",       "unknown",        None,   None,           "v85_난방빨리드네"),

    # ── v86: 이중부정/켜볼까/창문외풍 ────────────────────────────────
    ("끄지 않게 해줘",         "light_control",  "on",   None,           "v86_이중부정_끄지않게"),
    ("에어컨 끄지 말아줘",     "ac_control",     "on",   None,           "v86_이중부정_에어컨끄지마"),
    ("난방 끄지 마",           "heat_control",   "on",   None,           "v86_이중부정_난방끄지마"),
    ("불 켜볼까요",            "light_control",  "on",   None,           "v86_켜볼까요"),
    ("난방 한번 켜볼까",       "heat_control",   "on",   None,           "v86_난방한번켜볼까"),
    ("창문 외풍이 심해",       "unknown",        None,   None,           "v86_창문외풍"),

    # ── v87: 영어혼용/청유형/STT변형/OOD ────────────────────────
    ("에어컨 off 해줘",        "ac_control",     "off",  None,           "v87_영어off"),
    ("불 off",                 "light_control",  "off",  None,           "v87_불off"),
    ("불 좀 켜자",             "light_control",  "on",   None,           "v87_켜자청유형"),
    ("에어컨 꺼도 될까",       "ac_control",     "off",  None,           "v87_꺼도될까"),
    ("불 켜도 될까요",         "light_control",  "on",   None,           "v87_켜도될까요"),
    ("아이가 자는데 조용히 해줘","unknown",       None,   None,           "v87_조용히OOD"),
    ("보일라 켜줘",            "heat_control",   "on",   None,           "v87_보일라STT"),

    # ── v89: 기기상태조회/날씨관찰/필요없음 ─────────────────────
    # 상태조회: fn=device 유지 (v77 rule), exec=query_then_respond, dir=none
    ("에어컨 켜져 있어?",      "ac_control",     None,   None,           "v89_상태조회_AC"),
    ("난방 지금 켜져 있어?",   "heat_control",   None,   None,           "v89_상태조회_난방"),
    ("가스 잠겨 있어?",        "gas_control",    None,   None,           "v89_상태조회_가스"),
    ("오늘 날씨 좀 쌀쌀하네",  "weather_query",  None,   None,           "v89_날씨관찰_쌀쌀"),
    ("요즘 날씨 많이 더워",    "weather_query",  None,   None,           "v89_날씨관찰_더워"),
    ("요즘 건조해서 가습기 틀어야지","unknown",  None,   None,           "v89_가습기OOD"),
    ("바람이 불어서 환기 필요 없겠다","unknown", None,   None,           "v89_필요없겠다"),
    ("거실이 너무 밝아서 눈이 아파","light_control","down",None,         "v89_눈아파light_down"),
    ("눈이 뻑뻑해",            "vent_control",   "on",   None,           "v89_눈뻑뻑_vent"),

    # ── v88: 부정감각/완곡/조건부 OOD ────────────────────────────
    ("전혀 안 추워",           "unknown",        None,   None,           "v88_전혀안추워"),
    ("별로 안 더워",           "unknown",        None,   None,           "v88_별로안더워"),
    ("딱히 춥진 않은데",       "unknown",        None,   None,           "v88_딱히춥진않은데"),
    ("에어컨 안 켜도 되겠어",  "unknown",        None,   None,           "v88_안켜도되겠어"),
    ("난방 안 해도 될 것 같아","unknown",        None,   None,           "v88_안해도될것같아"),
    ("좀 덥긴 한데 에어컨까진","unknown",        None,   None,           "v88_에어컨까진"),
    ("에어컨 켜면 좀 시원해질까","unknown",      None,   None,           "v88_수사적조건"),
    ("에어컨이 켜져 있다면 꺼줘","ac_control",   "off",  None,           "v88_조건부명령꺼줘"),

    # ── v90: 찜질방비유/방문객귀가/감탄OOD/실외관찰 ──────────────
    ("방이 찜질방 같아",          "ac_control",   "on",   None,           "v90_찜질방비유"),
    ("조명 좀 밝혀줄 수 있을까요","light_control","up",   None,           "v90_밝혀줄_up"),
    ("취소해줘",                  "unknown",       None,   None,           "v90_취소해줘"),
    ("밖이 더 시원한 것 같은데",  "unknown",       None,   None,           "v90_실외관찰"),

    # ── v91: 더워졌어/기기고장/더위민감도 ────────────────────────
    ("청소하다 보니 더워졌어",    "ac_control",   "on",   None,           "v91_더워졌어"),
    ("에어컨에서 냄새 나는 것 같아","unknown",    None,   None,           "v91_에어컨냄새"),
    ("할아버지 더위 많이 타시는데","ac_control",  "on",   None,           "v91_더위타시는데"),
    ("낭방 켜줘",                  "heat_control", "on",   None,           "v91_낭방STT"),

    # ── v92: 일정/관리/실외/취침 확장 ────────────────────────────
    ("난방이 이상한 것 같아",     "unknown",       None,   None,           "v92_난방이상"),
    ("아침에 일어났어 불 켜줘",   "light_control", "on",   None,           "v92_일어났어불켜"),
    ("관리비 나왔어",             "unknown",       None,   None,           "v92_관리비OOD"),
    ("바깥 공기 좀 마시고 싶어",  "unknown",       None,   None,           "v92_바깥공기OOD"),
    ("집이 너무 어두워서 눈이 침침해","light_control","up", None,          "v92_눈침침_up"),
    ("이제 자야할 것 같아",       "light_control", "off",  None,           "v92_자야할_off"),
    # ── v93: 끄기의무/청유/환기청유/어둡불만 ─────────────────────
    ("불 꺼야 할 것 같아",        "light_control", "off",  None,           "v93_꺼야할_off"),
    ("불 끌까요",                 "light_control", "off",  None,           "v93_끌까요"),
    ("환기 시킬까요",             "vent_control",  "on",   None,           "v93_환기시킬까요"),
    ("여기 꽤 어둡네",            "light_control", "on",   None,           "v93_어둡네_light"),
    # ── v94: 볼륨/귀가/습도/현관 + preprocess 버그 수정 ──────────────────
    ("볼륨 올려줘",               "home_info",     "up",   None,           "v94_볼륨up"),
    ("볼륨 낮춰줘",               "home_info",     "down", None,           "v94_볼륨down"),
    ("조용히 해줘",               "home_info",     "down", None,           "v94_조용히_home"),
    ("귀가",                     "security_mode",  "on",   None,           "v94_귀가"),
    ("습도 어때",                 "weather_query",  None,   None,           "v94_습도날씨"),
    ("간접등 켜줘",               "light_control",  "on",   None,           "v94_간접등_on"),
    ("불 끄지 마",               "light_control",  "on",   None,           "v94_끄지마_on"),
    # ── v95: 극존칭/능력부정/허락요청/환기 ──────────────────────────────
    ("불 켜 주시겠어요?",          "light_control",  "on",   None,           "v95_켜주시겠어요"),
    ("조명 낮춰 주실 수 있을까요?", "light_control",  "down", None,           "v95_낮춰주실수있"),
    ("조명을 못 켜겠어요",         "unknown",        None,   None,           "v95_못켜겠어"),
    ("난방 꺼도 될까요?",         "heat_control",   "off",  None,           "v95_꺼도될까요"),
    ("환기 해봐",                 "vent_control",   "on",   None,           "v95_환기해봐"),
    ("불 끄면 안 될까요?",        "unknown",         None,   None,           "v95_끄면안될까"),

    # ── v97: 조건부/hearsay/비유/가스잠금 ──────────────────────────
    ("손님 오기 전에 좀 시원하게 해줘", "ac_control", "on",  None,           "v97_조건부시원하게"),
    ("밥 먹고 나서 에어컨 꺼",          "ac_control", "off", None,           "v97_조건부에어컨꺼"),
    ("출근하고 나면 가스 잠가줘",       "gas_control","close",None,          "v97_조건부가스잠금"),
    ("우리 남편이 더워한대",            "ac_control", "on",  None,           "v97_더워한대"),
    ("일어나면 불 켜줘",                "light_control","on", None,          "v97_일어나면불켜"),
    ("창문 쪽 바람이 세네",             "unknown",    None,  None,           "v97_창문바람관찰"),
    ("손님 오시는데 집이 좀 밝았으면", "light_control","on", None,           "v97_밝았으면소원"),
    ("방이 찜통 같아",                  "ac_control", "on",  None,           "v97_찜통비유"),
    ("자기 전에 난방 꺼",               "heat_control","off",None,           "v97_자기전난방꺼"),
    # ── v98: 귀가/취침/코드스위칭/잠금확인 ──────────────────────────────
    ("ventilation 켜줘",               "vent_control", "on",  None,          "v98_영어ventilation"),
    ("여행 다녀왔어 집이 꿉꿉해",       "vent_control", "on",  None,          "v98_꿉꿉"),
    ("오랜만에 집에 왔어 좀 환기시켜",  "vent_control", "on",  None,          "v98_귀가환기"),
    ("불 다 끄고 자자",                 "light_control","off", None,          "v98_끄고자자"),
    ("창문 잠겼어?",                    "door_control", None,  None,          "v98_창문잠겼어"),
    # ── v99: 습도불쾌/영어코드스위칭 확장 ───────────────────────────────
    ("집이 좀 텁텁한 것 같아",          "vent_control", "on",  None,          "v99_텁텁"),
    ("방이 눅눅해",                     "vent_control", "on",  None,          "v99_눅눅"),
    ("거실이 좀 뭉글뭉글해",            "vent_control", "on",  None,          "v99_뭉글"),
    # ── v100: 불만표현/수사적질문/존댓말꺼 ──────────────────────────────
    ("불 좀 꺼줘요, 부탁이에요",         "light_control","off", None,          "v100_꺼줘요부탁"),
    ("에어컨 계속 켜져있잖아",           "ac_control",  "off", None,          "v100_켜져있잖아"),
    ("에어컨이 왜 아직도 켜져 있어?",    "ac_control",  "off", None,          "v100_왜아직켜져"),
    ("왜 난방이 아직 돌아가?",           "heat_control","off", None,          "v100_왜아직돌아가"),
    # ── v101: 더위비유 확장/만족관찰 교정 ────────────────────────────────
    ("거실이 사막 같아",                 "ac_control",  "on",  None,          "v101_사막비유"),
    ("집이 가마솥 같아",                 "ac_control",  "on",  None,          "v101_가마솥비유"),
    ("방이 사우나가 따로 없어",           "ac_control",  "on",  None,          "v101_사우나따로없어"),
    ("온실 같이 따뜻하네",               "unknown",     None,  None,          "v101_온실만족관찰"),
    # ── v102: 완곡 제안형(어때/어떨까) dir 복구 ──────────────────────────
    ("난방 끄는 게 어때?",               "heat_control","off", None,          "v102_끄는게어때"),
    ("좀 시원하면 어때요?",              "ac_control",  "on",  None,          "v102_시원하면어때"),
    # ── v103: 미지원기기/긴급/긴박부사 ────────────────────────────────────
    ("냉장고 좀 열어줘",                 "unknown",     None,  None,          "v103_냉장고미지원"),
    ("가스 냄새나",                      "gas_control", "close",None,         "v103_가스냄새긴급"),
    ("빨리 불 켜줘",                     "light_control","on", None,          "v103_빨리켜줘"),
    ("얼른 에어컨 꺼줘",                 "ac_control",  "off", None,          "v103_얼른꺼줘"),
    # ── v104: 요리중 환기/밖이추운 날씨관찰 ─────────────────────────────
    ("삼겹살 굽는 중이야",               "vent_control","on",  None,          "v104_삼겹살굽기"),
    ("찌개 끓이고 있어",                 "vent_control","on",  None,          "v104_찌개끓이기"),
    ("밖이 추운 것 같은데",              "weather_query",None, None,          "v104_밖이추운것"),
    # ── STT+preprocess 보강 ──────────────────────────────────────────────
    ("거이 불 켜줘",                     "light_control","on",  "living",      "STT_거이거실"),
    ("조금씩 밝혀줘",                    "light_control","up",  None,          "STT_밝혀줘unknown"),
    # ── v105: 공기탁함/강약불만/허락형꺼도 ───────────────────────────────
    ("집이 너무 탁해",                   "vent_control", "on",  None,          "v105_탁해"),
    ("여기 에어컨이 너무 세다",           "ac_control",   "down",None,          "v105_너무세다"),
    ("난방이 너무 약해",                  "heat_control", "up",  None,          "v105_너무약해"),
    ("에어컨 이제 꺼도 되지?",            "ac_control",   "off", None,          "v105_꺼도되지"),
    # ── v106: v72모델 오예측 교정 + 쾌적 바람 ─────────────────────────────
    ("이제 좀 어둡게 해줘",              "light_control", "down",None,          "v106_어둡게_v72fix"),
    ("밥 먹으러 왔어",                   "unknown",       None,  None,          "v106_밥먹으러왔어"),
    ("집이 좀 쾌적했으면 좋겠어",        "ac_control",    "on",  None,          "v106_쾌적했으면"),
    ("퇴근하고 왔어",                    "security_mode", None,  None,          "v106_퇴근하고왔어_귀가"),
    ("불 좀 어둡게 해줄 수 있어?",       "light_control", "down",None,          "v106_어둡게_공손형"),
    # ── v108: hearsay확장/지시어/STT오인식 ────────────────────────────────
    ("남편이 춥다고 해",                 "heat_control",  "on",  None,          "v108_춥다고해_hearsay"),
    ("아내가 덥다고 하네",               "ac_control",    "on",  None,          "v108_덥다고하네_hearsay"),
    ("저것도 꺼줘",                      "unknown",       "off", None,          "v108_저것도꺼줘_dir"),
    ("불 켜조",                          "light_control", "on",  None,          "v108_켜조_STT"),
    ("에어컨 꺼조",                      "ac_control",    "off", None,          "v108_꺼조_STT"),
    ("외출 모드 해제해줘",               "security_mode", "off", None,          "v108_외출모드해제"),
    # ── v109: 창문닫았어 교정 / STT껴줘 / 의향형 / 습관형 ────────────────────
    ("창문 다 닫았어",                   "curtain_control","close",None,         "v109_창문닫았어_gas교정"),
    ("불 껴줘",                          "light_control",  "on",  None,          "v109_껴줘_STT불켜줘"),
    ("에어컨 켜야겠어",                  "ac_control",     "on",  None,          "v109_켜야겠어_의향"),
    ("보통 이 시간엔 에어컨 켜는데",     "ac_control",     "on",  None,          "v109_보통이시간에"),
    # ── v110: 간접화행 (Indirect Speech Acts) ─────────────────────────────────
    ("문 안 잠겼어",                     "door_control",   "close",None,         "v110_문안잠겼어_close"),
    ("현관 아직 열려있어",               "door_control",   "close",None,         "v110_현관열려있어_close"),
    ("에어컨에서 이상한 소리 나",         "unknown",        None,  None,          "v110_이상한소리_unknown"),
    ("화장실 불 켜져 있어",              "light_control",  "off", None,          "v110_불켜져있어_관찰형off"),
    # ── v110: 요리패턴 확장 (라면/전/요리중이야) ─────────────────────────────
    ("라면 끓이는 중이야",               "vent_control",   "on",  None,          "v110_라면끓이기_환기"),
    ("전 부치는 중이야",                 "vent_control",   "on",  None,          "v110_전부치기_환기"),
    ("요리 중이야",                      "vent_control",   "on",  None,          "v110_요리중이야"),
    # ── v111: 한증막/약한것같아/어두운것같아/훈훈포근 ─────────────────────────
    ("방이 한증막 같아",                 "ac_control",     "on",  None,          "v111_한증막_ac교정"),
    ("에어컨이 좀 약한 것 같아",         "ac_control",     "up",  None,          "v111_약한것같아_up"),
    ("불이 좀 어두운 것 같아",           "light_control",  "up",  None,          "v111_어두운것같아_up"),
    ("훈훈하네",                         "unknown",        None,  None,          "v111_훈훈_만족관찰"),
    ("포근하네요",                       "unknown",        None,  None,          "v111_포근_만족관찰"),
    # ── v112: 수사적반어/이중부정/더시원하게/월패드 ────────────────────────────
    ("조금 더 시원하게 해줘",            "ac_control",     "up",  None,          "v112_더시원하게_up"),
    ("좀 더 따뜻하게 해줘",             "heat_control",   "up",  None,          "v112_더따뜻하게_up"),
    ("에어컨 끄면 안 되나요?",           "ac_control",     "off", None,          "v112_끄면안되나_반어off"),
    ("불 좀 켜면 안 될까요?",           "light_control",  "on",  None,          "v112_켜면안될까_반어on"),
    ("에어컨 안 끄면 안 돼?",           "ac_control",     "off", None,          "v112_안끄면안돼_이중부정off"),
    ("불 안 켜도 안 돼?",               "light_control",  "on",  None,          "v112_안켜도안돼_이중부정on"),
    ("월패드 좀 꺼줘",                  "unknown",        None,  None,          "v112_월패드꺼줘_unknown"),

    # ── v113: DST 컨텍스트 오버라이드 (단독발화시 NLU예측 확인용) ─────────────
    # 온도 높여줘: heat_control/up → DST에서 AC컨텍스트면 ac_control/up 변환 (단독시 heat_control)
    ("온도 높여줘",                     "heat_control",   "up",  None,          "v113_온도높여줘_단독_heatcontrol"),
    ("온도 올려줘",                     "heat_control",   "up",  None,          "v113_온도올려줘_단독_heatcontrol"),

    # ── v114: 수사적 불평/완곡/STT오인식 확장 ────────────────────────────────
    ("조명이 좀 눈부셔요",              "light_control",  "down",None,          "v114_눈부셔요_down"),
    ("왜 이렇게 어둡냐",               "light_control",  "up",  None,          "v114_왜이렇게어둡냐_up"),
    ("에어컨 없이 어떻게 살아",          "ac_control",     "on",  None,          "v114_에어컨없이어떻게살아_on"),
    ("이 더위에 에어컨 안 틀면 누가 버티겠어", "ac_control", "on", None,        "v114_수사적_버티겠어_on"),
    ("저 좀 더운 것 같기도 하고",        "ac_control",     "on",  None,          "v114_더운것같기도_완곡_on"),
    ("실내가 좀 더운 것 같지 않아요",    "ac_control",     "on",  None,          "v114_더운것같지않아요_수사적on"),
    ("커트 열어줘",                     "curtain_control","open",None,          "v114_STT_커트_커튼"),
    ("벤틸 켜줘",                       "vent_control",   "on",  None,          "v114_STT_벤틸_환기"),
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
    {
        "name": "v82: 방 순차 전환 + AC 온도 낮춰줘",
        "turns": [
            ("거실 불 켜줘",       "light_control", "on",  "living",       "1_거실"),
            ("꺼줘",               "light_control", "off", "living",       "2_거실상속"),
            ("안방 에어컨 켜줘",   "ac_control",    "on",  "bedroom_main", "3_안방AC"),
            ("온도 낮춰줘",        "ac_control",    "down","bedroom_main", "4_온도낮춰"),
            ("꺼줘",               "ac_control",    "off", "bedroom_main", "5_AC꺼"),
        ]
    },
    {
        "name": "v82: unknown 후 기기 컨텍스트 복원",
        "turns": [
            ("방이 너무 더워",     "ac_control",    "on",  None,           "1_더워"),
            ("세게 틀어줘",        "ac_control",    "on",  None,           "2_세게"),
            ("이제 시원해졌어",    "unknown",       None,  None,           "3_만족표현"),
            ("이제 꺼줘",          "ac_control",    "off", None,           "4_꺼_컨텍스트복원"),
        ]
    },
    {
        "name": "v87: 그것도 꺼줘 (대명사 아나포라)",
        "turns": [
            ("거실 에어컨 켜줘",   "ac_control",    "on",  "living",       "1_AC켜기"),
            ("온도 조금 낮춰줘",   "ac_control",    "down","living",        "2_온도낮춰_regex수정"),
            ("그것도 꺼줘",        "ac_control",    "off", "living",       "3_그것도꺼_아나포라"),
        ]
    },
    {
        "name": "v87: 저녁 귀가 — 이제 그만 됐어",
        "turns": [
            ("다 왔어 불 켜줘",    "light_control", "on",  None,           "1_귀가_불켜"),
            ("에어컨도 틀어줘",    "ac_control",    "on",  None,           "2_에어컨켜"),
            ("온도 조금 낮춰줘",   "ac_control",    "down",None,           "3_온도낮춰"),
            ("이제 그만 됐어",     "ac_control",    "off", None,           "4_그만됐어_AC꺼"),
        ]
    },
    {
        "name": "v109: 방 전환 후 fn 상속 — 주방→거실",
        "turns": [
            ("주방 불 꺼줘",       "light_control", "off", "kitchen",      "1_주방명시"),
            ("거실은 켜줘",        "light_control", "on",  "living",       "2_거실전환_fn상속"),
            ("주방도 켜줘",        "light_control", "on",  "kitchen",      "3_주방복귀"),
        ]
    },
    {
        "name": "v109: 간접비유→기기제어→만족 흐름",
        "turns": [
            ("방이 찜통이야",      "ac_control",    "on",  None,           "1_찜통비유_AC켜기"),
            ("온도 좀 더 낮춰줘",  "ac_control",    "down",None,           "2_온도낮춰"),
            ("이제 좀 시원하네",   "unknown",       None,  None,           "3_만족관찰"),
            ("에어컨 꺼줘",        "ac_control",    "off", None,           "4_AC꺼기"),
        ]
    },
    {
        "name": "v109: 의향형 발화 → 제어 → 상태 확인",
        "turns": [
            ("에어컨 켜야겠어",    "ac_control",    "on",  None,           "1_의향형켜기"),
            ("보통 이 시간엔 온도 낮추는데", "ac_control", "down", None,  "2_습관형온도낮춤"),
            ("이제 잘 시간이야",   "unknown",       None,  None,           "3_잘시간_수면"),
        ]
    },
    {
        "name": "v110: 요리 중 환기 → 완료 후 꺼기",
        "turns": [
            ("라면 끓이는 중이야",  "vent_control",  "on",  None,           "1_라면요리중"),
            ("환기 좀 세게 해줘",   "vent_control",  "up",  None,           "2_환기강하게"),
            ("이제 그만 됐어",      "vent_control",  "off", None,           "3_요리완료_환기꺼"),
        ]
    },
    {
        "name": "v113: AC 컨텍스트에서 온도 높여/낮춰 → ac_control",
        "turns": [
            ("에어컨 켜줘",         "ac_control",    "on",  None,           "1_AC켜기"),
            ("온도 높여줘",         "ac_control",    "up",  None,           "2_온도높여_DST변환"),
            ("온도 좀 낮춰줘",      "ac_control",    "down",None,           "3_온도낮춰_DST변환"),
            ("온도 올려줘",         "ac_control",    "up",  None,           "4_온도올려_DST변환"),
        ]
    },
    {
        "name": "v113: 커튼 컨텍스트 후 완전히/끝까지 내려줘 → curtain",
        "turns": [
            ("거실 커튼 열어줘",    "curtain_control","open","living",       "1_커튼열기"),
            ("완전히 내려줘",       "curtain_control","close","living",      "2_완전히내려_DST변환"),
            ("다시 올려줘",         "curtain_control","up",  "living",       "3_올려주기"),
            ("끝까지 올려줘",       "curtain_control","up",  "living",       "4_끝까지올려_DST변환"),
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
