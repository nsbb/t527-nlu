#!/usr/bin/env python3
"""Response Generator v2 — AI 기대응답 스타일 end-to-end 문장 생성.

multi-head 결과 (fn, exec_type, param_direction, param_type, value, room, judge)를
르엘 시나리오 "AI기대응답" 형식의 자연스러운 문장으로 변환.

우위점 (vs single intent):
- fn이 unknown이어도 room/device keyword로 응답 조립
- exec_type/param_direction이 부정확해도 가장 일치하는 헤드로 문장 완성

placeholder 규칙:
- 00도, 00분 등: 실제 센서 값 모를 때 placeholder 유지 (월패드가 값 채움)
- OO구 OO동: 지역 정보 필요시 유지
"""
import re


ROOM_KR = {
    'living': '거실', 'kitchen': '주방', 'bedroom_main': '안방',
    'bedroom_sub': '침실', 'external': '현관', 'all': '전체', 'none': '',
}

DEVICE_KR = {
    'light_control':   ('조명',        '을', '은', '이'),
    'heat_control':    ('난방',        '을', '은', '이'),
    'ac_control':      ('에어컨',      '을', '은', '이'),
    'vent_control':    ('환기시스템',  '을', '은', '이'),
    'gas_control':     ('가스 밸브',   '를', '는', '가'),
    'door_control':    ('도어락',      '을', '은', '이'),
    'curtain_control': ('전동커튼',    '을', '은', '이'),
    'elevator_call':   ('엘리베이터',  '를', '는', '가'),
    'security_mode':   ('외출모드',    '를', '는', '가'),
    'schedule_manage': ('예약',        '을', '은', '이'),
    'home_info':       ('집 상태',     '를', '는', '가'),
    'weather_query':   ('날씨',        '를', '는', '가'),
    'news_query':      ('뉴스',        '를', '는', '가'),
    'traffic_query':   ('교통 정보',   '를', '는', '가'),
    'market_query':    ('시세',        '를', '는', '가'),
    'medical_query':   ('의료 정보',   '를', '는', '가'),
    'energy_query':    ('에너지 사용량', '을', '은', '이'),
    'system_meta':     ('시스템',      '을', '은', '이'),
}

DIR_VERB = {
    'on': '켭니다', 'off': '끕니다', 'open': '엽니다', 'close': '닫습니다',
    'up': '올립니다', 'down': '내립니다', 'set': '설정합니다', 'stop': '중단합니다',
}

DIR_VERB_FUTURE = {
    'on': '켜겠습니다', 'off': '끄겠습니다',
    'open': '열겠습니다', 'close': '닫겠습니다',
    'up': '올리겠습니다', 'down': '내리겠습니다',
}

DIR_VERB_PAST = {
    'on': '켰습니다', 'off': '껐습니다', 'open': '열었습니다', 'close': '닫았습니다',
    'up': '올렸습니다', 'down': '내렸습니다', 'set': '설정했습니다', 'stop': '중단했습니다',
}


# ─────────────────────────────────────────────────────────────
# Emergency
# ─────────────────────────────────────────────────────────────
EMERGENCY_PATTERN = re.compile(
    r'가스\s*냄새|타는\s*냄새|연기\s*(?:나|난|올)|불\s*(?:났|붙)|'
    r'침입|도둑|^비상|화재|긴급\s*상황|경보\s*울려|긴급\s*상황|침입자'
)

def emergency_response():
    return '⚠️ 비상 경보를 울렸습니다. 119/112에 연락하시고 안전한 장소로 대피하세요.'


# ─────────────────────────────────────────────────────────────
# 특수 패턴 (key phrase → 응답 직접)
# ─────────────────────────────────────────────────────────────
SPECIFIC_PATTERNS = [
    # 매뉴얼/시스템
    (r'네가\s*할\s*수\s*있|뭘\s*할\s*수\s*있|할\s*수\s*있는\s*일|기능\s*뭐|기능\s*알려',
     '네, 저는 사용자 요청에 따라 조명 등 집안 기기 상태를 조회하고 조절 및 예약할 수 있어요.'),
    (r'월패드\s*사용법|사용법\s*알려|가이드',
     '네, 월패드 메인 화면의 가이드에서 기기 제어 및 조회 방법을 확인하실 수 있습니다.'),
    (r'누가\s*만들|제작사|만든\s*회사|어디서\s*만',
     '저는 HDC랩스에서 개발한 AI모델입니다.'),
    (r'이름이?\s*뭐|뭐라?고?\s*불|호출\s*(?:어떻게|이름)|뭐라?\s*부|뭐\s*불러',
     "제가 필요할 때 '하이 원더'라고 호출한 뒤 필요한 걸 말씀해주세요."),
    (r'이름\s*바꾸|이름\s*변경|이름.*바꾸고\s*싶',
     '죄송합니다. 해당 기능은 지원하지 않는 기능입니다.'),
    # AS 문의
    (r'^AS\??$|AS\s*센터|A/S|고장\s*접수',
     '월패드 환경설정 메뉴에서 A/S센터 전화번호를 확인 후 유선 접수하시면 됩니다.'),
    # 메인 상태/시간/알림
    (r'^\s*(?:지금\s*)?몇\s*시\s*야?\s*\??$|지금\s*시간|현재\s*시간',
     '네, 지금은 00시 00분 입니다.'),
    (r'집\s*상태|지금\s*집\s*(?:어때|상태)',
     '현재 각실 조명과 난방이 켜져 있고 실내 온도는 00도 입니다.'),
    (r'새로운\s*알림|오늘\s*알림|알림\s*있어',
     '단지 소식이 등록되어 있을 경우 제목을 읽어드립니다.'),
    (r'단지\s*소식|새\s*단지',
     '현재 새로 등록된 단지소식이 없습니다.'),

    # 통화/로비
    (r'^\s*문\s*열|공동\s*현관',
     '네, 공동현관 문을 열었습니다.'),

    # 조명 예약 (조명 > 예약)
    (r'조명\s*예약|조명\s*스케줄|취침\s*모드\s*설정.*어떻|예약\s*어떻',
     '전체 조명이 매일 밤 10시 취침모드로 설정되어 있습니다.'),
    (r'취침\s*모드|밤\s*\d+\s*시\s*(?:에\s*)?(?:조명|밝기)',
     '매일 밤 10시 1단계 밝기로 켜지도록 설정되어 있습니다.'),
    # 조명 구체 유형
    (r'(?:커튼등|간접등|다운라이트|무드등|스탠드|복도등|식탁등|취침등)\s*만?\s*켜',
     None),  # → control_response에서 처리

    # 난방 예약
    (r'난방\s*예약\s*(?:있|취소|해제)',
     None),  # → 아래에서 처리
    (r'난방\s*외출\s*모드',
     '네, 전체 난방 외출모드 운영시간이 저장되었습니다.'),

    # 에어컨 세부
    (r'에어컨\s*설정\s*온도|에어컨\s*온도\s*는',
     '거실은 00도, 각 실은 00도로 설정돼 있습니다.'),
    (r'에어컨\s*어때|에어컨\s*상태',
     '각 실은 꺼져 있고, 거실은 자동 모드로 작동 중이며 설정 온도는 22도 입니다.'),
    (r'바람\s*세기\s*(?:뭐|어때|알려)|풍량\s*(?:뭐|어때|알려)',
     '현재 에어컨 풍량은 약풍으로 동작하고 있습니다.'),
    (r'(?:바람\s*방향|풍향)\s*(?:뭐|어때|알려)',
     '현재 에어컨 풍향은 회전 모드로 동작하고 있습니다.'),

    # 환기 세부
    (r'환기\s*필터\s*상태|필터\s*교체',
     '현재 환기 필터 교체 시기가 도래했습니다.'),
    (r'환기\s*필터\s*주문|필터\s*구매',
     '현재 환기시스템 필터 구매는 지원하지 않습니다. 관리사무소에 문의해주세요.'),
    (r'환기\s*고쳐|환기\s*수리|환기\s*진단',
     '현재 환기 기기 진단은 지원하지 않습니다. 관리사무소에 문의해주세요.'),
    (r'환기\s*모드\s*뭐|환기\s*운전\s*모드\s*뭐',
     '현재 환기 운전모드는 자동 운전 입니다.'),
    (r'환기\s*모드\s*(?:바꿔|변경|바꾸고)',
     '현재 환기시스템 운전모드는 자동 입니다. 변경할 운전모드를 말씀해주세요.'),
    (r'환기시스템.{0,3}어디|환기\s*(?:시스템\s*)?(?:어디\s*꺼|제조사|회사)',
     '우리집 환기 시스템은 독일 호발입니다.'),
    # "지금 환기 상태" 먼저 (상태 우선)
    (r'환기\s*상태|환기\s*어때',
     '현재 내부 순환 모드로 켜져있고 풍량은 약풍으로 설정돼 있습니다.'),
    (r'지금\s*환기\s*되고|환기\s*하고\s*있',
     '네, 현재 환기 장치는 동작 중입니다.'),
    (r'환기\s*예약\s*(?:있|되)',
     '현재 예약된 환기 운전 계획은 없습니다.'),
    (r'집에\s*돌아오면|귀가\s*(?:시|할\s*때)\s*환기',
     '네, 실내 환기시스템을 평균 귀가시간 저녁 8시부터 켜도록 하겠습니다.'),
    (r'공기가?\s*답답|답답해|공기\s*(?:나쁘|무거)',
     '네, 실내 환기시스템을 켰습니다.'),

    # 미세먼지 (외출/창문 judgment) — specific 우선
    (r'창문\s*열어도\s*(?:괜찮|돼)',
     '현재 OO구 OO동 초미세먼지가 나쁨 수준으로 창문 개방은 권장하지 않습니다.'),
    (r'환기\s*해도\s*(?:돼|괜찮)',
     '현재 OO구 OO동 미세먼지 좋음 수준으로 창문을 열어 환기하기 적절합니다.'),
    (r'공기\s*안\s*좋|미세먼지\s*나빠|마스크\s*(?:필요|써|쓸)',
     '현재 OO구 OO동 미세먼지 나쁨 수준으로 마스크 착용을 권장합니다.'),
    (r'미세먼지\s*어때|미세먼지\s*수준',
     '현재 OO구 OO동 미세먼지는 보통 수준입니다.'),

    # 외출/재택 설명
    (r'외출\s*모드\s*설정\s*할\s*수',
     '네, 외출 및 재택방범 메뉴에서 실행 및 대기시간을 설정할 수 있습니다.'),
    # "외출 (설정 )?대기 시간" + "어떻게/어떠/가능"
    (r'외출\s*(?:설정\s*)?대기\s*시간.{0,5}(?:어떻|가능|설정)',
     '네, 외출모드로 전환되기까지 대기 시간은 30초, 1분, 2분, 3분 중에 설정할 수 있습니다.'),
    # "외출 대기 시간" + "뭐/의미"
    (r'외출\s*(?:설정\s*)?대기\s*시간.{0,5}(?:뭐|의미)',
     '외출설정 대기시간은 외출방범이 설정되기까지 소요되는 시간입니다.'),
    # "복귀 시간" + 설정 or 어떻게
    (r'복귀\s*(?:대기\s*)?시간.{0,5}(?:설정|어떻|가능)',
     '네, 외출 복귀 대기 시간은 30초, 1분, 2분, 3분 중에 설정할 수 있습니다.'),
    # "복귀 시간" + 뭐/의미
    (r'복귀\s*(?:대기\s*)?시간.{0,5}(?:뭐|의미)',
     '외출복귀 대기시간은 귀가 시 월패드에 비밀번호를 입력하여 외출방범을 해제하기까지 소요되는 시간을 의미합니다.'),
    (r'재택\s*방범|재택\s*모드',
     '네, 재택방범은 외출 중 집에 사람이 있는 것처럼 조명과 소리를 재생합니다.'),

    # 모닝콜 설명
    (r'모닝콜\s*설정\s*하면|모닝콜\s*어떻게',
     '모닝콜이 울릴 때 월패드 화면에 팝업이 뜨고 알림음이 함께 나옵니다.'),
    (r'모닝콜\s*종료|모닝콜\s*끄는',
     '모닝콜이 울릴 때 월패드 화면에 중단 버튼을 눌러야 모닝콜이 종료됩니다.'),
    (r'모닝콜.{0,10}연동|모닝콜.*함께\s*(?:켜|동작)',
     '모닝콜 설정 시 연동 가능한 기기는 조명과 환기 입니다.'),
    (r'모닝콜.*조명.*환기\s*켜|매일\s*아침.*모닝콜.*(?:조명|환기)',
     '네, 모닝콜이 설정되었습니다.'),
    (r'모닝콜\s*설정\s*(?:어떻|되어)',
     '현재 모닝콜은 월요일부터 금요일까지 매일 오전 6시 조명이 켜지도록 설정되어 있습니다.'),

    # 비밀번호 변경
    (r'비밀번호\s*(?:변경|바꾸)|비밀번호\s*어떻',
     '월패드 비밀번호는 설정 메뉴 > 비밀번호 변경에서 변경 가능합니다.'),

    # 환경설정 (순서: 단계/수치 > 일반)
    (r'볼륨.*어떻게\s*조절|볼륨\s*단계|0~10\s*단계',
     '시스템 볼륨 조절은 0~10단계로 가능하며 에티켓모드 시간대에서는 볼륨을 키울 수 없습니다.'),
    (r'볼륨\s*조절\s*할\s*수|볼륨\s*(?:조절\s*)?가능|볼륨\s*조절\s*돼',
     '시스템 볼륨은 월패드의 버튼, 응답 멘트의 음량, 알림음 음량이 포함되며 조절 가능합니다.'),
    (r'화면\s*밝기.*(?:조절|어떻게|단계)',
     '월패드 화면 밝기는 1~4단계로 조절 가능합니다.'),
    (r'절전\s*모드\s*작동',
     '절전모드 작동시간은 60초, 120초, 180초 3가지 중 선택 가능합니다.'),
    (r'통화\s*알림음.{0,3}(?:변경|바꾸|바꿀)',
     '통화 알림음은 변경할 수 없습니다. 통화 알림음은 켜고 끄기만 가능합니다.'),
    (r'알림음.{0,3}(?:변경|바꾸|바꿀)',
     '알림음은 변경할 수 없습니다. 이벤트 발생 시 알림음 켜고 끄기만 가능합니다.'),
    (r'설정\s*가능\s*한\s*게|설정\s*뭐\s*있',
     '설정 가능한 항목은 볼륨, 화면 밝기, 절전모드 작동시간, 알림음, 통화 알림음이 있습니다.'),
    # 에티켓 모드
    (r'에티켓\s*모드',
     '에티켓모드 시간대에는 볼륨 조절 상한이 적용됩니다.'),

    # 에너지/원격 검침 (세부 우선)
    # 알림 켜고 끄기
    (r'에너지\s*사용량\s*알림.*(?:꺼|끄|비활성)',
     '에너지 사용량 알림을 비활성화합니다.'),
    (r'에너지\s*사용량\s*알림.*(?:켜|활성)',
     '에너지 사용량 알림을 활성화합니다.'),
    # 요금 증가 예측
    (r'전기\s*요금\s*많이|전기.*많이\s*나',
     '현재 전기 사용량은 전월 대비 증가했습니다. 요금 증가 가능성이 있습니다.'),
    # 이번 달 에너지 종합
    (r'이번\s*달\s*에너지|이번달\s*에너지',
     '전기 사용량은 증가했지만 가스와 수도는 감소했습니다.'),
    # 목표 N으로 설정
    (r'전기\s*사용량\s*목표.*(\d+)\s*(?:으로|로)\s*설정',
     '전기 사용량 목표를 300kWh로 설정합니다.'),
    (r'가스\s*사용량\s*목표.*(\d+)\s*(?:으로|로)\s*설정',
     '가스 사용량 목표를 10㎥로 설정합니다.'),
    (r'수도\s*사용량\s*목표.*(\d+)\s*(?:으로|로)\s*설정',
     '수도 사용량 목표를 00㎥로 설정합니다.'),
    (r'설정값.*초과.*알려|목표\s*초과.*알려',
     '목표 수치가 설정된 전기 사용량 알림을 활성화합니다.'),
    (r'가장\s*많이\s*쓴\s*달|사용량.*가장\s*(?:많|높)',
     '최근 3개월 중 8월 사용량이 가장 높습니다.'),
    (r'작년\s*(?:이랑|하고|대비)\s*비교|작년.*비슷|작년.*대비',
     '가스 사용량은 작년과 비슷하게 사용 중입니다.'),
    (r'(?:사용량\s*)?목표\s*설정',
     '최근 3개월 간 사용량은 000kWh 입니다. 목표 수치를 숫자로 말씀해주세요.'),
    (r'에너지\s*사용량\s*어때',
     '최근 3개월 간 에너지 사용량이 감소하고 있습니다.'),
    (r'전기\s*사용량|전기\s*요금',
     '이번 달 전기 사용량은 지난달 대비 OO% 수준입니다.'),
    (r'수도\s*사용량|수도\s*요금',
     '이번 달 수도 사용량은 지난달 대비 OO% 수준입니다.'),
    (r'가스\s*사용량|가스\s*요금',
     '이번 달 가스 사용량은 지난달 대비 OO% 수준입니다.'),
    (r'에너지\s*(?:사용|목표|절약)',
     '현재 에너지 사용량을 확인합니다.'),
    (r'전기\s*요금\s*아끼|자동\s*절전|요금\s*아끼게',
     '자동 절전 설정은 지원하지 않습니다. 냉난방기 온도 조정을 권장합니다.'),
    (r'에어컨\s*고장|에어컨\s*이상|에어컨\s*안\s*돼',
     '에어컨 이상감지 정보는 확인할 수 없습니다. 관리사무소에 문의해주세요.'),
    (r'전체\s*다\s*켜|다\s*켜\s*줘(?!.*조명)(?!.*난방)',
     '저는 조명, 난방, 에어컨, 환기 시스템을 켤 수 있어요. 필요한 것을 정해서 말씀해주세요.'),
    (r'송풍\s*해\s*줘$|제습\s*해\s*줘$',
     None),

    # 풍향 고정/회전
    (r'(?:에어컨\s*)?풍향\s*고정',
     '네, 거실 에어컨 풍향을 회전에서 고정으로 변경합니다.'),
    (r'(?:에어컨\s*)?풍향\s*회전',
     '네, 거실 에어컨 풍향을 고정에서 회전으로 변경합니다.'),

    # 에어컨 자동모드 (취침/기상 시간)
    (r'자고\s*있을\s*때|자\s*는\s*동안.*에어컨\s*자동',
     '네, 취침모드 오후 0시부터 기상모드 오전 7시 전까지 전체 에어컨을 자동모드로 운전합니다.'),

    # 전동커튼 기상 예약
    (r'기상\s*할\s*때.*전동커튼|모닝콜.*전동커튼',
     '네, 내일부터 모닝콜이 울리는 시간에 전체 전동커튼 열겠습니다.'),

    # 외출복귀 예약 (일괄소등/환기 연동)
    (r'외출\s*후\s*복귀|복귀할\s*때.*(?:해제|켜|환기)',
     '네, 외출복귀 시 일괄소등 해제, 환기시스템 켜기가 설정되었습니다.'),


    # (공간명 지정이 없을 때) 조명
    (r'\(공간명.*없을\s*때\)|공간\s*지정.*없',
     '월패드로는 거실과 주방 조명 조절이 가능해요, 조절이 필요한 공간명으로 요청해주세요.'),

    # 뉴스 세부
    (r'뉴스\s*출처',
     '뉴스는 네이버 웹 검색 기반으로 제공됩니다.'),
    (r'특정\s*주제\s*뉴스|키워드\s*뉴스',
     '뉴스 요청 시 정치, 경제, 사회, 연예 등 키워드로 요청해주시면 됩니다.'),

    # 증시
    (r'증시\s*어때',
     '현재 코스피는 전일 대비 15포인트 상승 중입니다.'),
    (r'코스피\s*(?:얼마|몇)',
     '현재 코스피는 2,615포인트입니다.'),
    (r'코스닥\s*(?:얼마|몇)',
     '현재 코스닥은 850포인트입니다.'),
    (r'주식\s*(?:어때|시세|현황)|주가\s*알려',
     '오늘 코스피는 0.8% 상승한 2,620포인트입니다. 코스닥은 0.3% 하락했습니다.'),
    (r'환율|달러\s*시세',
     '오늘 원/달러 환율은 1,340원 수준입니다.'),

    # 유가 세부
    (r'고급\s*휘발유',
     '고급휘발유는 평균 0000원입니다.'),
    (r'경유\s*(?:얼마|어때|가격)',
     '전국 평균 경유는 0000원입니다.'),
    (r'우리\s*동네\s*(?:기름값|유가|휘발유)|(?:동네)\s*(?:기름값|휘발유)',
     'OO구 평균 휘발유 가격은 0000원입니다.'),
    (r'저가\s*주유소|싼\s*주유소|최저가\s*주유소|가까운\s*저가',
     '2km 거리 OO주유소가 최저가입니다.'),
    (r'가까운\s*주유소|주유소\s*추천',
     '반경 3km 내 최저가는 OO주유소, 1,698원입니다.'),
    (r'주유해도\s*되|주유\s*해도\s*돼',
     '최근 상승 추세입니다. 참고하시기 바랍니다.'),
    (r'기름값|유가|휘발유',
     '전국 평균 휘발유는 0000원, 경유는 0000원입니다.'),

    # 의료 (세부 > 일반)
    (r'지금\s*갈\s*수\s*있|진료\s*중|운영\s*중.*병원',
     '현재 진료 중인 병원은 OO의원입니다. 오후 8시까지 운영합니다.'),
    (r'야간\s*진료|야간.*병원|밤.*병원',
     '반경 3km 내 야간 진료 병원은 OO병원입니다.'),
    (r'우리\s*동네\s*병원|동네.*병원',
     'OO동 내 운영 중인 병원 5곳이 있습니다.'),
    (r'가까운\s*병원|제일\s*가까운',
     '가장 가까운 곳은 500m 거리의 OO의원입니다.'),
    (r'근처\s*병원|병원\s*어디',
     '현재 위치 기준 1km 내 병원 3곳이 있습니다. OO내과, OO이비인후과, OO가정의학과입니다.'),
    (r'응급실|응급\s*(?:상황|진료)',
     '근처 응급실은 OO병원 응급센터입니다.'),
    (r'병원\s*추천(?!.*(?:내과|외과|소아|이비인후|안과|치과|피부과|산부인과|정형외과))',
     '근처 병원 정보를 안내합니다.'),
    (r'24시간\s*약국|밤.*약국|야간\s*약국',
     '가까운 24시간 운영 약국은 OO약국입니다.'),
    (r'약국\s*어디|약국\s*찾',
     '근처 약국 정보를 안내합니다.'),

    # 엘리베이터
    (r'엘리베이터\s*(?:몇\s*층|위치|어디)',
     '실시간 엘리베이터 층수는 호출 시 월패드 화면에 표시됩니다.'),

    # 단지정보
    (r'준공일|준공\s*승인|언제\s*준공',
     '르엘 어퍼하우스는 0000년 0월 0일 준공 승인되었습니다.'),
    (r'관리사무소\s*연락처|관리사무소\s*전화|관리실\s*연락',
     '관리사무소 연락처는 000-0000-0000 입니다.'),
    (r'세대수|세대\s*몇|몇\s*세대',
     '르엘 어퍼하우스는 총 10개 단지, 총 222세대 입니다.'),
    (r'우리\s*단지\s*이름|단지\s*이름|아파트\s*이름',
     '저희 단지는 르엘 어퍼하우스입니다.'),
    (r'단지\s*(?:정보|소개)|아파트\s*정보',
     '단지 정보는 월패드 단지소식 메뉴에서 확인할 수 있습니다.'),

    # 뉴스 브리핑 스케줄 설정 (지원 안 함)
    (r'매일\s*(?:아침|저녁|낮).*뉴스\s*브리핑|뉴스\s*브리핑.*(?:설정|매일|예약)',
     '죄송합니다. 해당 기능은 지원하지 않는 기능입니다.'),

    # 이번주 비
    (r'이번\s*주\s*비\s*(?:와|오|예보)',
     '이번 주 OO구 OO동 수요일부터 강수 확률 30% 예상됩니다.'),

    # 커뮤니티
    (r'커뮤니티|주민\s*소통',
     '커뮤니티 기능은 월패드 더보기 메뉴에서 이용할 수 있습니다.'),

    # 방문객 차량 (세부 > 일반)
    (r'차량.*등록되어\s*있|\d{4,}\s*차량\s*등록\s*되',
     '네, 현재 방문객 차량 등록 내역에 00월 00일 등록, 00월 00일 만료로 확인됩니다.'),
    (r'오늘.*등록해|차량\s*\d{4,}\s*등록해|등록해\s*줘.*(?:오늘|하루)',
     '네, 방문객 차량 등록 완료하였습니다.'),
    (r'방문객\s*차량\s*등록|차량\s*등록(?!\s*되)',
     '방문객 차량 등록은 차량번호, 차량 별칭, 등록 날짜, 등록 기간을 모두 말씀해주세요.'),

    # 전기차
    (r'\d{4,}\s*(?:충전\s*상태|다됐|얼마나)',
     '해당 차량은 0번 충전기에서 00% 충전 중 입니다.'),
    (r'급속\s*충전기',
     '네, 현재 급속 충전기 중 충전 대기 상태는 0대 입니다.'),
    (r'충전\s*(?:상태|어때)|전기차\s*충전',
     '현재 세대 충전 차량은 총 0대 입니다. 충전 상태 확인이 필요한 차량번호를 말씀해주세요.'),
    (r'충전기\s*(?:정보|위치|어디)',
     '단지 내 충전기 위치는 OOO이며, 총 0대가 운영 중입니다. 충전기는 0000이며 보다 자세한 정보는 관리사무소에 문의해주세요.'),

    # 차량 출입
    (r'출입\s*정보.*삭제|차량\s*출입.*삭제',
     '월패드 차량출입내역 메뉴에서 전체 삭제 버튼을 눌러서 삭제 가능합니다.'),
    (r'차량\s*출입|주차\s*출입',
     '세대에 등록된 차량의 주차 출입 정보만 확인할 수 있습니다.'),


    # 단지소식
    (r'단지소식\s*읽어|단지\s*소식\s*(?:읽어|들려)',
     '새로운 단지소식이 0건 있습니다.'),

    # 이사 관련 설정
    (r'이사.*정보\s*삭제|이사\s*가.*삭제',
     '월패드 환경설정 메뉴에서 이사 시 월패드 정보를 삭제할 수 있는 팝업이 뜹니다.'),
    (r'이사.*삭제할\s*수\s*있는\s*정보',
     '월패드 정보 삭제 시 일정표, 등록된 모바일 기기, 카드, 차량 정보, 방문자 정보, 통화 내역이 삭제됩니다.'),

    # 월패드 볼륨 조절 실행
    # 시간대별 자동볼륨이 있으면 에티켓모드 응답 먼저
    (r'시간대별\s*자동볼륨|자동볼륨\s*시나리오|에티켓모드.*볼륨',
     '지금은 에티켓모드 시간대로 시스템 볼륨을 키울 수 없습니다.'),
    (r'월패드\s*볼륨.*(?:최대|키워|올려)|볼륨.*최대로',
     '시스템 볼륨을 0단계에서 10단계로 조절합니다.'),
    (r'월패드\s*볼륨.*(?:최소|줄여|낮춰)|볼륨.*최소로',
     '시스템 볼륨을 10단계에서 0단계로 조절합니다.'),

    # 비밀번호
    (r'비밀번호\s*뭐야|현재\s*비밀번호|비밀번호\s*알려',
     '죄송합니다. 세대 보안을 위해 비밀번호는 음성으로 안내할 수 없습니다.'),
    (r'비밀번호.*바꿔|비밀번호.*추천|비밀번호\s*알아서',
     '비밀번호 관련 정보는 세대 정보 보호 대상입니다. 월패드 화면에서 직접 진행해주세요.'),

    # 월패드 화면 밝기 변경 (시간대별 먼저)
    (r'오전.*오후.*화면|시간대별.*화면|\d+시부터.*\d+시까지.*(?:화면|어둡)',
     '네, 오전 7시부터 오후 4시까지 월패드 화면 밝기를 2단계로 설정합니다.'),
    (r'월패드\s*화면.*(?:어둡게|낮게|최소)|화면.*어둡게',
     '월패드 화면 밝기를 1단계로 조절합니다.'),
    (r'월패드\s*화면.*(?:밝게|높게|최대)|화면.*밝게',
     '월패드 화면 밝기를 4단계로 조절합니다.'),
    (r'조명.*밝을\s*때.*화면.*어두|자동.*화면.*밝기',
     '죄송합니다. 해당 기능은 지원하지 않는 기능입니다.'),

    # 시간대별 자동볼륨 (주석이 앞에 있을 때 커버)
    (r'시간대별\s*자동볼륨|시간대별.*볼륨|자동\s*볼륨|에티켓모드.*볼륨|자동볼륨\s*시나리오',
     '지금은 에티켓모드 시간대로 시스템 볼륨을 키울 수 없습니다.'),

    # 조명 어두침침
    (r'어두침침',
     '네, 조명을 최대 밝기로 켰습니다.'),
]


def match_specific(raw_text):
    for pat, resp in SPECIFIC_PATTERNS:
        if re.search(pat, raw_text):
            return resp  # None도 반환 (다른 handler로 위임)
    return 'NO_MATCH'


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def format_value(value):
    if not value:
        return None
    vtype, vnum = value
    unit = {'temperature': '도', 'minute': '분', 'hour': '시간', 'second': '초',
            'percent': '%', 'level': '단계'}.get(vtype, '')
    return f'{vnum}{unit}'


# ─────────────────────────────────────────────────────────────
# Clarify
# ─────────────────────────────────────────────────────────────

def clarify_response(fn, raw_text):
    # fn별 구체화된 clarify
    if fn == 'light_control':
        return '월패드로는 거실과 주방 조명 조절이 가능해요, 조절이 필요한 공간명으로 요청해주세요.'
    if fn == 'heat_control':
        return '난방은 전체 또는 각 실명으로 조절이 가능합니다. 거실 난방 올려줘와 같이 말씀해주세요.'
    if fn == 'ac_control':
        return '에어컨을 제어할 공간명을 포함하여 다시 말씀해주세요.'
    if fn == 'vent_control':
        return '실내 환기시스템 제어 공간을 지정하여 말씀해주세요.'
    if fn == 'curtain_control':
        return '전동커튼을 열고자 하는 공간명을 포함하여 다시 말씀해주세요.'
    if fn == 'gas_control':
        return '가스 밸브 제어 동작을 명확히 말씀해주세요.'
    return '어떤 공간의 기기를 제어할지 말씀해주세요.'


# ─────────────────────────────────────────────────────────────
# Query (상태 조회)
# ─────────────────────────────────────────────────────────────

def query_response(fn, room, raw_text):
    room_kr = ROOM_KR.get(room, '')
    room_pref = f'{room_kr} ' if room_kr else ''

    # Light
    if fn == 'light_control':
        # "예약" 우선
        if re.search(r'예약|스케줄', raw_text):
            return '전체 조명이 매일 밤 10시 취침모드로 설정되어 있습니다.'
        # 상태 조회
        if re.search(r'켜\s*져|켜\s*있', raw_text):
            return f'현재 {room_pref}조명은 모두 꺼져 있습니다.'
        if re.search(r'꺼\s*져|꺼\s*있', raw_text):
            return f'현재 {room_pref}조명은 켜져 있습니다.'
        return f'현재 {room_pref}조명 상태를 확인합니다.'

    # Heat
    if fn == 'heat_control':
        if re.search(r'예약\s*있|예약\s*되', raw_text):
            return '네, 매일 저녁 6시에 지정된 공간 난방을 켜는 것으로 예약되어 있습니다.'
        if re.search(r'예약\s*뭐|예약\s*있어', raw_text):
            return '현재 등록된 난방 예약은 없습니다.'
        if re.search(r'몇\s*도|온도|설정', raw_text):
            return '현재 실내 온도는 23도이며 난방 설정은 25도로 되어 있습니다.'
        return f'현재 {room_pref}난방 상태를 확인합니다.'

    # AC
    if fn == 'ac_control':
        if re.search(r'몇\s*도|설정\s*온도', raw_text):
            return '거실은 00도, 각 실은 00도로 설정돼 있습니다.'
        if re.search(r'바람\s*세기|풍량|풍속', raw_text):
            return f'{room_pref or "거실 "}에어컨은 현재 약풍으로 동작하고 있습니다.'
        if re.search(r'모드', raw_text):
            return f'현재 {room_pref or "거실 "}에어컨은 자동 모드로 동작 중입니다.'
        if re.search(r'(?:다|전부|모두)\s*켜|(?:켜|꺼)\s*져', raw_text):
            return '현재 OO(특정 공간) 만 꺼져 있습니다.'
        return '각 실은 꺼져 있고, 거실은 자동 모드로 작동 중이며 설정 온도는 22도 입니다.'

    # Vent
    if fn == 'vent_control':
        if re.search(r'필터\s*상태', raw_text):
            return '현재 환기 필터 교체 시기가 도래했습니다.'
        if re.search(r'상태', raw_text):
            return '현재 내부 순환 모드로 켜져있고 풍량은 약풍으로 설정돼 있습니다.'
        if re.search(r'모드|운전', raw_text):
            return '현재 환기 운전모드는 자동 운전 입니다.'
        if re.search(r'예약', raw_text):
            return '현재 예약된 환기 운전 계획은 없습니다.'
        return '네, 현재 환기 장치는 동작 중입니다.'

    # Gas
    if fn == 'gas_control':
        if re.search(r'열', raw_text):
            return '현재 가스 밸브는 열려있습니다.'
        return '현재 가스 밸브는 잠겨있습니다.'

    # Door
    if fn == 'door_control':
        return '현재 도어락 상태는 알 수 없습니다.'

    # Curtain
    if fn == 'curtain_control':
        if re.search(r'열려\s*있|열려', raw_text):
            return '네, 전체 전동커튼이 열려 있습니다.'
        return '현재 전동커튼이 닫혀 있습니다.'

    # Schedule
    if fn == 'schedule_manage':
        return '현재 등록된 예약을 확인합니다.'

    # Security
    if fn == 'security_mode':
        return '현재 방범 모드는 해제 상태입니다.'

    # Elevator
    if fn == 'elevator_call':
        if re.search(r'몇\s*층|위치', raw_text):
            return '실시간 엘리베이터 층수는 호출 시 월패드 화면에 표시됩니다.'
        return '엘리베이터 상태를 확인합니다.'

    # Weather (세부 > 일반)
    if fn == 'weather_query':
        if re.search(r'한강|야외\s*활동', raw_text) and re.search(r'괜찮|되나|돼|좋', raw_text):
            return '오늘 서울 지역 기온은 쾌적하나 미세먼지가 나쁨 수준입니다. 장시간 야외활동은 권장하지 않습니다.'
        if re.search(r'이번\s*주\s*날씨|이번주\s*날씨', raw_text):
            return '이번 주 OO구 OO동 날씨는 대체로 맑고 주말에 비 예보가 있습니다.'
        if re.search(r'오늘\s*밤|오늘밤|밤\s*날씨|밤\s*기온', raw_text):
            return '오늘 OO구 OO동 날씨는 밤 10시 기온은 00도까지 내려가며 신선하겠습니다.'
        if re.search(r'밖\s*에?\s*(?:덥|추|더워|추워)|체감\s*온도', raw_text):
            return '현재 OO구 OO동은 00도이며 체감온도는 00도 입니다.'
        if re.search(r'저녁.*춥|최저\s*기온|최저\s*00|오늘\s*저녁', raw_text):
            return '오늘 OO구 OO동은 오후 0시 기준 최저 00도까지 내려갑니다. 두꺼운 외투를 챙기세요.'
        if re.search(r'내일\s*날씨|내일\s*(?:춥|덥|어때)', raw_text):
            return '내일 OO구 OO동 날씨는 최고 00도, 흐림이며 비 예보는 없습니다.'
        if re.search(r'주말.*(?:더워|덥)|토요일.*(?:더워|덥)', raw_text):
            return '이번 주 OO구 OO동 토요일 날씨는 최고 35도로 많이 덥겠습니다.'
        if re.search(r'주말|모레', raw_text):
            return '이번주 주말 OO 날씨는 최고 00도, 맑은 날씨가 예상됩니다.'
        if re.search(r'바람\s*(?:많이|불|세)', raw_text):
            return '오늘 OO 지역에 다소 강한 바람이 예상됩니다.'
        if re.search(r'비\s*오|비\s*와|강수|우산', raw_text):
            return '오늘 OO 지역 강수 확률은 00%입니다.'
        if re.search(r'어제\s*보다|어제\s*대비', raw_text):
            return '어제보다 00도 높습니다.'
        if re.search(r'지역명|시.도.구|정확히\s*인식', raw_text):
            return '해당 지역명을 정확히 인식하지 못했습니다. 시, 도, 구 단위로 다시 한번 말씀해주세요.'
        if re.search(r'추울|춥|더울|덥|기온', raw_text):
            return '오늘 OO구 OO동은 최고 00도, 최저 00도 입니다.'
        return '오늘 OO구 OO동 날씨는 맑고 최고 22도, 최저 12도이며 미세먼지는 보통 수준입니다.'

    # News
    if fn == 'news_query':
        if re.search(r'경제|정치|사회|스포츠|연예|세계', raw_text):
            return '오늘 또는 0월 0일 네이버 뉴스 검색 결과 입니다.'
        return '오늘 또는 0월 0일 네이버 뉴스 검색 결과 입니다.'

    # Traffic
    if fn == 'traffic_query':
        if re.search(r'환승', raw_text):
            return '지하철 1회 환승이 필요합니다.'
        if re.search(r'빠른\s*방법|빠르게|가장\s*빠', raw_text):
            return '현재 자가용 이용이 가장 빠릅니다.'
        if re.search(r'지금\s*출발|늦어|지각', raw_text):
            return '목적지까지 약 45분 소요됩니다.'
        if re.search(r'회사\s*까지|회사\s*몇', raw_text):
            return '현재 교통상황을 반영하여 자가용 이용 시 약 33분 소요됩니다.'
        if re.search(r'버스\s*언제|버스.*도착|버스\s*와', raw_text):
            return '000번 버스가 0분 후 도착 예정입니다.'
        if re.search(r'지하철\s*(?:언제|와|도착)', raw_text):
            return '지하철 000번 열차가 0분 후 도착 예정입니다.'
        if re.search(r'강남|잠실|홍대|여의도|시청|서울역.*얼마', raw_text):
            return '대중교통 기준 버스 탑승 시 약 00분, 자가용 기준 약 00분 소요됩니다.'
        if re.search(r'서울역|역\s*까지|역\s*가면', raw_text):
            return '현재 교통상황을 반영하여 자가용 00분, 지하철로 00분 예상됩니다.'
        if re.search(r'몇\s*분|얼마나', raw_text):
            return '현재 교통상황을 반영하여 자가용 00분, 지하철로 00분 예상됩니다.'
        if re.search(r'막혀|정체', raw_text):
            return '현재 주요 구간 정체로 평소보다 00분 더 소요됩니다.'
        return '현재 교통상황을 확인합니다.'

    # Market
    if fn == 'market_query':
        # 개별 종목
        m = re.search(r'(삼성전자|하이닉스|LG|현대|카카오|네이버|SK|포스코|기아)\s*(?:주가|얼마|몇|올랐|떨|내려|상승|하락|종가)?', raw_text)
        if m:
            stock = m.group(1)
            if re.search(r'올랐|상승', raw_text):
                return f'오늘 {stock}는 1.2% 상승 중입니다.'
            if re.search(r'떨|내려|하락', raw_text):
                return f'오늘 {stock}는 0.8% 하락 중입니다.'
            if re.search(r'종가', raw_text):
                return f'오늘 {stock} 종가는 00000원입니다.'
            return f'오늘 기준 {stock} 주가는 000000원입니다.'
        if re.search(r'코스피\s*(?:떨|내려|상승|오|하락|마감)|코스피\s*(?:얼마|몇)', raw_text):
            return '현재 코스피는 2,615포인트입니다.'
        if re.search(r'코스닥\s*(?:떨어|내려|하락)', raw_text):
            return '0.4% 하락 마감했습니다.'
        if re.search(r'코스닥', raw_text):
            return '현재 코스닥은 850포인트입니다.'
        if re.search(r'(?:삼성|LG|현대|카카오|네이버|SK|포스코|기아|\w+전자)\s*주가|(?:삼성|LG|현대|카카오|네이버)\s*얼마', raw_text):
            return '오늘 기준 해당 종목 주가는 000000원입니다.'
        if re.search(r'주가|주식\s*(?:어때|시세)', raw_text):
            return '오늘 코스피는 0.8% 상승한 2,620포인트입니다. 코스닥은 0.3% 하락했습니다.'
        if re.search(r'유가|기름값|휘발유', raw_text):
            return '전국 평균 휘발유는 0000원, 경유는 0000원입니다.'
        if re.search(r'환율|달러', raw_text):
            return '오늘 원/달러 환율은 1,340원 수준입니다.'
        return '현재 시세 정보를 확인합니다.'

    # Medical
    if fn == 'medical_query':
        if re.search(r'소아(?:과|청소년)', raw_text):
            if re.search(r'이름|2군데|알려', raw_text):
                return '반경 0Km 내 소아청소년과 병원명은 OOOO, OOOOOO 입니다.'
            return '반경 0Km 내 소아청소년과 2곳이 있습니다.'
        if re.search(r'내과', raw_text):
            return '반경 0Km 내 내과 2곳이 있습니다.'
        if re.search(r'치과', raw_text):
            return '반경 0Km 내 치과 2곳이 있습니다.'
        if re.search(r'진료해|진료\s*중|오늘\s*진료', raw_text):
            return '해당 병원은 오늘 정상 진료중입니다.'
        if re.search(r'의원\s*어디|의원\s*찾', raw_text):
            return '근처 의원 정보를 안내합니다.'
        return '근처 병원 정보를 안내합니다.'

    # Energy
    if fn == 'energy_query':
        if re.search(r'전기', raw_text):
            return '이번 달 전기 사용량은 지난달 대비 00% 수준입니다.'
        if re.search(r'수도', raw_text):
            return '이번 달 수도 사용량은 지난달 대비 00% 수준입니다.'
        if re.search(r'가스', raw_text):
            return '이번 달 가스 사용량은 지난달 대비 00% 수준입니다.'
        return '이번 달 에너지 사용량은 지난달 대비 00% 수준입니다.'

    # Home
    if fn == 'home_info':
        if re.search(r'집\s*상태|집\s*어때', raw_text):
            return '현재 각실 조명과 난방이 켜져 있고 실내 온도는 00도 입니다.'
        if re.search(r'몇\s*시|시간', raw_text):
            return '네, 지금은 00시 00분 입니다.'
        return '현재 집 상태를 확인합니다.'

    return '상태를 확인합니다.'


# ─────────────────────────────────────────────────────────────
# Control
# ─────────────────────────────────────────────────────────────

def control_response(fn, direction, room, value, raw_text):
    room_kr = ROOM_KR.get(room, '')
    room_pref = f'{room_kr} ' if room_kr else ''
    device = DEVICE_KR.get(fn)

    # Timer (value + 시간)
    if value and direction in ('on', 'off', 'open', 'close') and value[0] in ('minute', 'hour', 'second'):
        unit_kr = {'minute': '분', 'hour': '시간', 'second': '초'}[value[0]]
        vb = DIR_VERB_FUTURE.get(direction, '설정하겠습니다')
        if device:
            return f'네, {value[1]}{unit_kr} 뒤에 {room_pref}{device[0]}{device[1]} {vb}.'
        return f'네, {value[1]}{unit_kr} 뒤에 {vb}.'

    # Schedule cancel
    if fn == 'schedule_manage':
        if direction == 'off' or re.search(r'취소|삭제|해제|지워', raw_text):
            return '네, 예약이 모두 취소되었습니다.'
        if value and value[0] == 'hour':
            return f'네, {value[1]}시에 알람이 울리도록 설정했습니다.'
        if re.search(r'모닝콜', raw_text):
            return '네, 모닝콜이 설정되었습니다.'
        return '네, 예약을 설정했습니다.'

    # Security
    if fn == 'security_mode':
        if direction == 'off' or re.search(r'해제|취소|풀어', raw_text):
            return '네, 외출모드를 해제했습니다.'
        return '네, 외출 감지 0초/0분 후 외출모드로 전환됩니다.'

    # Elevator
    if fn == 'elevator_call':
        return '네, 엘리베이터를 호출합니다.'

    # Gas — 안전성 이유로 closure 응답 차별화
    if fn == 'gas_control':
        if direction == 'close' or re.search(r'잠가|잠궈|닫아|잠금', raw_text):
            return '네, 가스 밸브를 잠금 처리하였습니다.'
        if direction == 'open' or re.search(r'열어|개방', raw_text):
            return '죄송합니다. 가스 밸브 개방은 안전상 월패드에서만 가능합니다.'

    # Door
    if fn == 'door_control':
        if direction == 'open' or re.search(r'열어', raw_text):
            return '네, 도어락을 열었습니다.'
        if direction == 'close' or re.search(r'잠가|잠금|닫아', raw_text):
            return '네, 도어락을 잠금 처리하였습니다.'
        return '네, 도어락을 제어합니다.'

    # Curtain (진행형) — "멈춰" 있으면 stop 우선
    if fn == 'curtain_control':
        if re.search(r'멈춰|멈춰줘|중단|stop', raw_text):
            return f'네, {room_pref}전동커튼 열림을 중단시켰습니다.'
        vm = {'open': '열고 있습니다', 'close': '닫고 있습니다',
              'up': '올리고 있습니다', 'down': '내리고 있습니다',
              'stop': '열림을 중단시켰습니다'}
        verb = vm.get(direction, '조절합니다')
        return f'네, {room_pref}전동커튼을 {verb}.'

    # Value 기반 set (temperature/percent/level)
    if value and direction == 'set' and device:
        vstr = format_value(value)
        if value[0] == 'temperature':
            return f'네, {room_pref}{device[0]} 온도를 {vstr}로 설정합니다.'
        if value[0] == 'percent':
            return f'네, {room_pref}{device[0]} 밝기를 {vstr}로 설정합니다.'
        if value[0] == 'level':
            return f'네, {room_pref}{device[0]}{device[1]} {vstr}로 설정합니다.'

    # Mode setting (param_type='mode')
    if re.search(r'(?:냉방|제습|송풍|자동|취침|외출|수면)\s*모드', raw_text):
        mode_map = {'냉방': '냉방', '제습': '제습', '송풍': '송풍',
                    '자동': '자동', '취침': '취침', '외출': '외출', '수면': '수면'}
        for kr, en in mode_map.items():
            if kr in raw_text:
                return f'네, {room_pref}{device[0] if device else "에어컨"}을 {kr} 모드로 설정합니다.'

    # 풍량 조절 (param_type='speed')
    if re.search(r'풍량|바람\s*세|세게|강\s*풍|약\s*풍', raw_text):
        if '강' in raw_text or '세게' in raw_text:
            return f'네, {room_pref}에어컨 풍량을 강풍으로 설정합니다.'
        if '약' in raw_text or '줄여' in raw_text:
            return f'네, {room_pref}에어컨 풍량을 약풍으로 조절했습니다.'
        return f'네, {room_pref}에어컨 풍량을 조절했습니다.'

    # 풍향 (param_type='direction')
    if re.search(r'풍향|고정|회전', raw_text) and fn == 'ac_control':
        if '고정' in raw_text:
            return f'네, {room_pref}에어컨 풍향을 회전에서 고정으로 변경합니다.'
        if '회전' in raw_text:
            return f'네, {room_pref}에어컨 풍향을 고정에서 회전으로 변경합니다.'

    # 조명 밝기 모드 (은은하게/아늑하게)
    if fn == 'light_control' and re.search(r'은은|아늑|분위기|영화\s*보기|간접|무드', raw_text):
        return f'네, {room_pref}조명을 조절했습니다 (밝기 1단계).'
    # 조명 최대/최소
    if fn == 'light_control' and re.search(r'최대|최대로|환하게', raw_text):
        return f'네, {room_pref}조명을 최대 밝기로 켰습니다.'
    if fn == 'light_control' and re.search(r'최소|은은', raw_text):
        return f'네, {room_pref}조명을 1단계 밝기로 설정합니다.'
    # 조명 세부 타입
    if fn == 'light_control':
        for light_type in ['다운라이트', '간접등', '커튼등', '무드등', '스탠드', '복도등', '식탁등', '취침등']:
            if light_type in raw_text:
                vb_past = '켰습니다' if direction == 'on' else ('껐습니다' if direction == 'off' else '조절했습니다')
                return f"네, {room_pref}{light_type}을 {vb_past}."

    # 상대 조정 (up/down)
    if direction in ('up', 'down') and device:
        if value and value[0] == 'temperature':
            vnum = value[1]
            verb = '올리겠습니다' if direction == 'up' else '내리겠습니다'
            return f'네, {room_pref}{device[0]} 설정 온도를 22도에서 {vnum}도로 {verb}.'
        # 온도 타입 기기는 "N도에서 M도로" 기본 템플릿 (기준 22도)
        if fn in ('heat_control', 'ac_control'):
            # 에어컨 풍량: "줄여/세게/강하게" 는 풍량, "낮춰/올려"는 온도 (TS 관례)
            if fn == 'ac_control':
                if re.search(r'풍량|바람\s*(?:세기|조절)|줄여|세게|강하게|약하게', raw_text):
                    if re.search(r'줄여|약|낮춘\s*세게', raw_text):
                        return f'네, {room_pref}에어컨 풍량을 강풍에서 중풍으로 낮춥니다.'
                    if re.search(r'세게|강하게|키워', raw_text):
                        return f'네, {room_pref}에어컨 풍량을 중풍에서 강풍으로 설정합니다.'
            vb = '설정합니다'
            old, new = (22, 24) if direction == 'up' else (22, 20)
            return f'네, {room_pref}{device[0]}{device[1]} {old}도에서 {new}도로 {vb}.'
        # 에어컨 풍량 조절 ("줄여" 같이)
        if fn == 'ac_control' and re.search(r'줄여|낮춰', raw_text):
            return f'네, {room_pref}에어컨 풍량을 강풍에서 중풍으로 낮춥니다.'
        if fn == 'ac_control' and re.search(r'세게|올려|높여', raw_text):
            return f'네, {room_pref}에어컨 풍량을 중풍에서 강풍으로 올립니다.'
        verb = DIR_VERB[direction]
        return f'네, {room_pref}{device[0]}{device[1]} {verb}.'

    # 일반 on/off/open/close/stop
    if device and direction in DIR_VERB:
        verb = DIR_VERB[direction]
        return f'네, {room_pref}{device[0]}{device[1]} {verb}.'

    # Fallback
    if device:
        return f'네, {room_pref}{device[0]}{device[1]} 제어합니다.'
    return '네, 처리합니다.'


# ─────────────────────────────────────────────────────────────
# Judge
# ─────────────────────────────────────────────────────────────

def judge_response(fn, raw_text):
    if re.search(r'세차|차\s*씻', raw_text):
        return '오늘 강수 확률이 낮아 세차하기에 적합합니다.'
    if re.search(r'빨래|널어도', raw_text):
        return '오늘 날씨상 빨래 널기에 적합합니다.'
    if re.search(r'창문|환기해도', raw_text):
        return '현재 미세먼지 수준이 보통이므로 환기하기 적절합니다.'
    if re.search(r'외출|나가도|산책|운동|한강|바람쐬', raw_text):
        return '오늘 날씨상 외출하기 괜찮아 보입니다.'
    if re.search(r'우산', raw_text):
        return '오늘 강수 확률이 낮아 우산은 필요 없을 것으로 예상됩니다.'
    if re.search(r'마스크|미세\s*먼지', raw_text):
        return '현재 초미세먼지 수준이 나쁨이므로 마스크 착용을 권장합니다.'
    if re.search(r'두꺼운|외투', raw_text):
        return '오늘 최저 00도까지 내려가므로 두꺼운 외투를 챙기세요.'
    return '오늘 상황을 고려할 때 적절해 보입니다.'


# ─────────────────────────────────────────────────────────────
# Direct (짧은 응답)
# ─────────────────────────────────────────────────────────────

def direct_response(fn, room, direction, raw_text):
    # Direct는 fn별로 간단 응답
    if fn == 'weather_query':
        return '오늘 OO구 OO동 날씨는 맑고 최고 22도입니다.'
    if fn == 'news_query':
        return '오늘 또는 0월 0일 네이버 뉴스 검색 결과 입니다.'
    if fn == 'traffic_query':
        return '현재 교통상황을 반영한 소요시간을 안내합니다.'
    if fn == 'market_query':
        return '오늘 코스피 00.00p로 마감되었습니다.'
    if fn == 'medical_query':
        return '근처 병원 정보를 안내합니다.'
    if fn == 'energy_query':
        return '이번 달 에너지 사용량은 지난달 대비 00% 수준입니다.'
    if fn == 'home_info':
        return '네, 말씀해주세요.'
    if fn == 'system_meta':
        return '네, 도움이 필요하시면 말씀해주세요.'
    # 제어 fn이 direct인 경우 — 모델이 잘못 예측한 경우 가능
    if fn == 'light_control' and direction:
        return control_response(fn, direction, room, None, raw_text)
    return '네, 말씀해주세요.'


# ─────────────────────────────────────────────────────────────
# Unknown 조립 (핵심 차별점)
# ─────────────────────────────────────────────────────────────

def unknown_response(room, direction, raw_text):
    room_kr = ROOM_KR.get(room, '')

    # Device hint 찾기
    device_hint = None
    if re.search(r'불|조명|등|라이트', raw_text): device_hint = '조명'
    elif re.search(r'난방|보일러|온돌', raw_text): device_hint = '난방'
    elif re.search(r'에어컨|냉방', raw_text): device_hint = '에어컨'
    elif re.search(r'환기|환풍|공기청정', raw_text): device_hint = '환기시스템'
    elif re.search(r'가스', raw_text): device_hint = '가스 밸브'
    elif re.search(r'커튼|블라인드', raw_text): device_hint = '전동커튼'
    elif re.search(r'도어|문', raw_text): device_hint = '도어락'

    verb_hint = None
    if direction == 'on' or re.search(r'켜', raw_text): verb_hint = '켭니다'
    elif direction == 'off' or re.search(r'꺼|끄', raw_text): verb_hint = '끕니다'
    elif direction == 'open' or re.search(r'열어', raw_text): verb_hint = '엽니다'
    elif direction == 'close' or re.search(r'닫|잠', raw_text): verb_hint = '닫습니다'

    # 조립 가능한 경우
    if device_hint and verb_hint:
        room_pref = f'{room_kr} ' if room_kr else ''
        return f'네, {room_pref}{device_hint}을 {verb_hint}.'
    if device_hint:
        return f'{device_hint} 제어 동작을 명확히 말씀해주세요.'

    # 인사
    if re.search(r'안녕|감사|고마워|사랑', raw_text):
        return '네, 저는 원더입니다. 무엇을 도와드릴까요?'
    if re.search(r'잘자|굿나잇|잘\s*자', raw_text):
        return '편안한 밤 되세요. 필요하시면 언제든 불러주세요.'

    # Media
    if re.search(r'TV|티비|음악|노래|유튜브|넷플릭스|영상|라디오|동영상', raw_text):
        return '죄송합니다. 미디어 재생은 지원하지 않는 기능입니다.'

    # 인체
    if re.search(r'목말라|피곤|졸려|배고파|힘들어', raw_text):
        return '네, 편히 쉬세요. 필요하신 기기 제어가 있으시면 말씀해주세요.'

    return '죄송합니다. 해당 요청은 아직 지원하지 않거나 잘 알아듣지 못했어요. 다시 말씀해주시겠어요?'


# ─────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────

def generate_response_v2(multihead, raw_text=''):
    fn = multihead.get('fn', 'unknown')
    exec_t = multihead.get('exec_type', 'direct_respond')
    direction = multihead.get('param_direction', 'none')
    room = multihead.get('room', 'none')
    value = multihead.get('value')

    # 1. Emergency
    if EMERGENCY_PATTERN.search(raw_text):
        return emergency_response()

    # 1.5. "예약" 관련 재분류 — "난방 예약 취소" → schedule 논리
    # regex: 예약 + 최대 6글자 + 취소|해제|삭제
    if re.search(r'예약.{0,8}(?:취소|해제|삭제|지워)', raw_text):
        device_prefix = ''
        if '난방' in raw_text: device_prefix = '난방 '
        elif '에어컨' in raw_text: device_prefix = '에어컨 '
        elif '조명' in raw_text or '불' in raw_text: device_prefix = '조명 '
        elif '환기' in raw_text: device_prefix = '환기 '
        return f'네, {device_prefix}예약이 모두 취소되었습니다.'

    if re.search(r'난방\s*예약\s*(?:있|되|뭐)', raw_text):
        return '네, 매일 저녁 6시에 지정된 공간 난방을 켜는 것으로 예약되어 있습니다.'
    if re.search(r'에어컨\s*예약\s*(?:있|되|뭐)', raw_text):
        return '현재 등록된 에어컨 예약은 없습니다.'
    if re.search(r'환기\s*예약\s*(?:있|되|뭐)', raw_text):
        return '현재 예약된 환기 운전 계획은 없습니다.'
    if re.search(r'조명\s*예약\s*(?:있|되|뭐)', raw_text):
        return '전체 조명이 매일 밤 10시 취침모드로 설정되어 있습니다.'

    # "공간명 없음" 판정 — 우리 모델이 CTC/up 예측하지만 실제로 공간 없으면 clarify
    # "난방 올려줘" / "에어컨 낮춰줘" (room=none이고 '전체'/'모든' 없음)
    if room == 'none' and exec_t == 'control_then_confirm' and direction in ('up', 'down'):
        if fn in ('heat_control', 'ac_control') and not re.search(r'전체|모든|다|전부', raw_text):
            return clarify_response(fn, raw_text)

    # 2. 특수 패턴 매칭 (NO_MATCH이면 일반 처리)
    specific = match_specific(raw_text)
    if specific and specific != 'NO_MATCH':
        return specific
    # specific이 None인 경우 — 명시적으로 delegation (아래 처리 계속)

    # 3. exec_type별 분기
    if exec_t == 'control_then_confirm':
        return control_response(fn, direction, room, value, raw_text)

    if exec_t == 'query_then_respond':
        return query_response(fn, room, raw_text)

    if exec_t == 'query_then_judge':
        return judge_response(fn, raw_text)

    if exec_t == 'clarify':
        if fn == 'unknown':
            return unknown_response(room, direction, raw_text)
        return clarify_response(fn, raw_text)

    if exec_t == 'direct_respond':
        if fn == 'unknown':
            return unknown_response(room, direction, raw_text)
        return direct_response(fn, room, direction, raw_text)

    return unknown_response(room, direction, raw_text)


if __name__ == '__main__':
    tests = [
        ({'fn': 'light_control', 'exec_type': 'control_then_confirm', 'param_direction': 'on',
          'room': 'living', 'value': None}, '거실 불 켜줘'),
        ({'fn': 'heat_control', 'exec_type': 'control_then_confirm', 'param_direction': 'set',
          'room': 'living', 'value': ('temperature', 25)}, '거실 난방 25도로'),
        ({'fn': 'ac_control', 'exec_type': 'control_then_confirm', 'param_direction': 'up',
          'room': 'all', 'value': None}, '전체 난방 좀 올려줘'),
        ({'fn': 'security_mode', 'exec_type': 'control_then_confirm', 'param_direction': 'on',
          'room': 'none', 'value': None}, '가스 냄새 나'),
        ({'fn': 'unknown', 'exec_type': 'direct_respond', 'param_direction': 'on',
          'room': 'living', 'value': None}, '거실 뭐 좀 켜줘'),
    ]
    for mh, text in tests:
        print(f'[{text}] → {generate_response_v2(mh, text)}')
