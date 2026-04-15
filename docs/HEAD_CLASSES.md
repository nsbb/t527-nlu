# Multi-Head 클래스 정의

## 1. fn (20개) — 기능/디바이스 식별

| 클래스 | 의미 | 예시 발화 |
|--------|------|----------|
| light_control | 조명 제어/조회 | "거실 불 켜줘", "조명 밝기 올려" |
| heat_control | 난방 제어/조회 | "난방 올려줘", "온도 어때?" |
| ac_control | 에어컨 제어/조회 | "에어컨 23도", "제습 모드로" |
| vent_control | 환기 제어/조회 | "환기 켜줘", "환기 상태 어때?" |
| gas_control | 가스 밸브 제어/조회 | "가스 잠가줘", "가스 열려있어?" |
| door_control | 도어락/현관문 제어/조회 | "도어락 열어줘", "현관문 상태" |
| curtain_control | 전동커튼 제어/조회 | "커튼 열어줘", "커튼 닫아" |
| elevator_call | 엘리베이터 호출/조회 | "엘리베이터 불러줘", "몇 층이야?" |
| security_mode | 외출/재택 보안 모드 | "외출모드 실행", "나 나간다" |
| schedule_manage | 예약/모닝콜 관리 | "모닝콜 설정해줘", "예약 확인" |
| weather_query | 날씨/미세먼지 조회+판단 | "오늘 날씨 어때?", "세차해도 돼?" |
| news_query | 뉴스 조회 | "뉴스 틀어줘", "경제 뉴스" |
| traffic_query | 교통 정보 조회 | "강남역까지 얼마나 걸려?" |
| energy_query | 에너지 사용량 조회/설정 | "에너지 사용량 어때?", "목표 설정" |
| home_info | 집 상태/알림/시간/볼륨/화면 | "집 상태 어때?", "몇 시야?", "볼륨 올려" |
| system_meta | AI 기능/사용법/개발자 정보 | "뭐 할 수 있어?", "누가 만들었어?" |
| market_query | 주식/유가 조회 | "코스피 얼마야?", "유가 어때?" |
| medical_query | 병원/의료 정보 조회 | "근처 병원 어디?", "소아과 찾아줘" |
| vehicle_manage | 차량/주차/충전 관리 | "주차 정보 확인", "충전 상태 어때?" |
| unknown | 미지원/도메인 밖 | "피자 주문해줘", "노래 틀어줘" → 서버로 전송 |

## 2. exec_type (5개) — 실행 유형

| 클래스 | 의미 | 동작 | 예시 |
|--------|------|------|------|
| control_then_confirm | 기기 제어 → 확인 응답 | API 호출 후 "네, ~했습니다" | "거실 불 켜줘" → "네, 거실 조명을 켰습니다" |
| query_then_respond | 상태/정보 조회 → 정보 응답 | API 조회 후 결과 반환 | "온도 어때?" → "현재 23도입니다" |
| query_then_judge | 외부 정보 조회 → 판단 응답 | API 조회 + 규칙 판단 | "세차해도 돼?" → "비 예보 없어 괜찮습니다" |
| direct_respond | 바로 고정 응답 | 조회/제어 없이 응답 | "누가 만들었어?" → "HDC랩스입니다" |
| clarify | 정보 부족 → 재질문 | 사용자에게 추가 정보 요청 | "불 켜줘" → "어떤 공간의 조명을 켤까요?" |

## 3. param_direction (9개) — 동작 방향

| 클래스 | 의미 | 예시 |
|--------|------|------|
| on | 켜기/시작 | "켜줘", "틀어줘", "시작해" |
| off | 끄기/중단 | "꺼줘", "끄다", "중단해" |
| up | 올리기/높이기 | "올려줘", "높여줘", "세게" |
| down | 내리기/낮추기 | "낮춰줘", "줄여줘", "약하게" |
| open | 열기 | "열어줘", "열어봐" |
| close | 닫기/잠그기 | "닫아줘", "잠가줘" |
| set | 특정 값 설정 | "23도로 맞춰줘", "제습 모드로" |
| stop | 멈추기 | "멈춰줘", "중지" |
| none | 방향 없음 (조회 등) | "어때?", "상태 확인" |

## 4. param_type (5개) — 파라미터 종류

| 클래스 | 의미 | 예시 |
|--------|------|------|
| temperature | 온도 | "23도로", "온도 올려줘" |
| brightness | 밝기 | "밝기 올려", "어둡게 해줘", "은은하게" |
| mode | 모드 | "제습 모드", "송풍으로", "자동 모드" |
| speed | 풍량/볼륨/속도 | "세게 틀어", "볼륨 올려", "바람 세기" |
| none | 파라미터 없음 | "켜줘", "꺼줘", "상태 어때?" |

## 5. judge (5개) — 판단 유형

| 클래스 | 의미 | 트리거 키워드 | 예시 |
|--------|------|-------------|------|
| outdoor_activity | 야외활동 가능 판단 | 세차, 캠핑, 나가도, 소풍 | "세차해도 돼?" → 강수확률+미세먼지 확인 |
| clothing | 옷차림 추천 | 입고, 옷, 겉옷 | "뭐 입고 나가?" → 기온 확인 후 추천 |
| air_quality | 공기질 판단 | 창문, 환기해도, 미세먼지 | "창문 열어도 괜찮아?" → 미세먼지 등급 확인 |
| cost_trend | 가격 추세 판단 | 주유해도, 기름값 | "주유해도 되나?" → 유가 추세 확인 |
| none | 판단 없음 | - | 대부분의 제어/조회 발화 |

## 6. Rule 기반 추출 (모델 외부)

| 슬롯 | 방법 | 값 | 예시 |
|------|------|-----|------|
| room | 키워드 매칭 | living, kitchen, bedroom_main, bedroom_sub, all | "거실" → living |
| value | regex | 숫자+단위 | "23도" → 23, "5분" → 5 |
| 미지원 액션 | 키워드 매칭 | - | "택시 불러줘" → fn=traffic이지만 "택시" = 미지원 → 거부 |

## 조합 예시

| 발화 | fn | exec | direction | param | judge | room | value |
|------|-----|------|-----------|-------|-------|------|-------|
| 거실 불 켜줘 | light_control | control_then_confirm | on | none | none | living | - |
| 에어컨 23도 | ac_control | control_then_confirm | set | temperature | none | - | 23 |
| 찜통이야 | ac_control | control_then_confirm | on | none | none | - | - |
| 온도 어때? | heat_control | query_then_respond | none | temperature | none | - | - |
| 세차해도 돼? | weather_query | query_then_judge | none | none | outdoor_activity | - | - |
| 누가 만들었어? | system_meta | direct_respond | none | none | none | - | - |
| 나 나간다 | security_mode | control_then_confirm | on | none | none | - | - |
| 피자 주문해줘 | unknown | direct_respond | none | none | none | - | - |
