# 94-Intent 분류 체계 (Flat Intent CNN)

초기 NLU에서 사용한 94개 intent 목록. 총 16,879개 학습 데이터.
이후 Multi-Head 방식으로 전환됨.

## 조명 (6개, 932개)

| Intent | Action | 데이터 | 예시 |
|--------|--------|------:|------|
| light_on | power_on | 272 | "불 켜줘" |
| light_off | power_off | 225 | "불 꺼줘" |
| light_dim | control (brightness) | 256 | "조명 은은하게" |
| light_brighten | control (brightness) | 101 | "좀 밝게 해줘" |
| light_query | query | 35 | "불 켜져있어" |
| light_schedule | schedule | 43 | "조명 예약" |

## 난방 (8개, 698개)

| Intent | Action | 데이터 | 예시 |
|--------|--------|------:|------|
| heating_on | power_on | 271 | "난방 켜줘" |
| heating_off | power_off | 69 | "난방 꺼줘" |
| heating_up | control (temperature) | 155 | "난방 올려줘" |
| heating_down | control (temperature) | 64 | "난방 낮춰줘" |
| heating_query | query | 41 | "온도 어때" |
| heating_schedule_query | query | 29 | "난방 예약 있어" |
| heating_schedule_set | schedule | 39 | "난방 예약" |
| heating_schedule_cancel | schedule | 30 | "난방 예약 취소" |

## 에어컨 (9개, 591개)

| Intent | Action | 데이터 | 예시 |
|--------|--------|------:|------|
| ac_on | power_on | 145 | "에어컨 켜줘" |
| ac_off | power_off | 64 | "에어컨 꺼줘" |
| ac_temp | control (temperature) | 70 | "에어컨 23도" |
| ac_mode | control (mode) | 74 | "제습 모드" |
| ac_mode_noroom | control (mode) | 45 | "송풍 해줘" |
| ac_mode_schedule | schedule (mode) | 41 | "취침모드 에어컨" |
| ac_wind | control (wind) | 68 | "바람 세게" |
| ac_query | query | 46 | "에어컨 어때" |
| ac_exception | exception | 38 | "전체 다 켜?" |

## 환기 (7개, 354개)

| Intent | Action | 데이터 | 예시 |
|--------|--------|------:|------|
| ventilation_on | power_on | 118 | "환기 켜줘" |
| ventilation_off | power_off | 45 | "환기 꺼줘" |
| ventilation_mode | control (mode) | 47 | "환기 자동" |
| ventilation_query | query | 53 | "환기 상태" |
| ventilation_schedule_query | query | 29 | "환기 예약 있어?" |
| ventilation_schedule_set | schedule | 30 | "돌아오면 환기 틀어줘" |
| ventilation_exception | exception | 32 | "환기 필터 교체해줘" |

## 커튼 (5개, 171개)

| Intent | Action | 데이터 | 예시 |
|--------|--------|------:|------|
| curtain_open | power_on | 44 | "커튼 열어줘" |
| curtain_close | power_off | 34 | "커튼 닫아줘" |
| curtain_stop | control | 32 | "커튼 멈춰" |
| curtain_query | query | 30 | "전동커튼 열려있어" |
| curtain_schedule | schedule | 31 | "기상할 때 커튼 열어줘" |

## 도어/도어락 (3개, 154개)

| Intent | Action | 데이터 | 예시 |
|--------|--------|------:|------|
| door_open | power_on | 87 | "문 열어줘" |
| doorlock_open | power_on | 38 | "도어락 열어줘" |
| doorlock_query | query | 29 | "도어락 상태 어때" |

## 가스 (2개, 67개)

| Intent | Action | 데이터 | 예시 |
|--------|--------|------:|------|
| gas_close | power_off | 35 | "가스 잠궈줘" |
| gas_query | query | 32 | "가스 상태" |

## 보안 (3개, 118개)

| Intent | Action | 데이터 | 예시 |
|--------|--------|------:|------|
| security_activate | power_on | 44 | "외출 모드 켜줘" |
| security_query | query | 34 | "외출 설정" |
| security_return_set | schedule | 40 | "복귀할 때 소등 해제" |

## 엘리베이터 (2개, 62개)

| Intent | Action | 데이터 | 예시 |
|--------|--------|------:|------|
| elevator_call | call | 32 | "엘리베이터 불러줘" |
| elevator_query | query | 30 | "엘리베이터 몇 층" |

## 비상 (1개, 64개)

| Intent | Action | 데이터 | 예시 |
|--------|--------|------:|------|
| emergency | call | 64 | "비상" |

## 알람/모닝콜 (2개, 444개)

| Intent | Action | 데이터 | 예시 |
|--------|--------|------:|------|
| alarm_set | schedule | 289 | "알람 맞춰줘" |
| alarm_query | query | 155 | "모닝콜 설정" |

## 시스템/환경설정 (6개, 457개)

| Intent | Action | 데이터 | 예시 |
|--------|--------|------:|------|
| system_volume_set | control (volume) | 275 | "볼륨 키워줘" |
| system_faq | query | 50 | "볼륨 조절" |
| system_exception | exception | 38 | "에티켓모드" |
| system_brightness_set | control (brightness) | 30 | "화면 밝기 조절" |
| system_brightness_schedule | schedule (brightness) | 32 | "아침에 화면 밝게" |
| password_change | control | 32 | "비밀번호 바꾸고 싶어" |

## AI 어시스턴트 (5개, 200개)

| Intent | Action | 데이터 | 예시 |
|--------|--------|------:|------|
| manual_capability | query | 68 | "뭐 할 수 있어?" |
| manual_name | query | 36 | "뭐라고 불러?" |
| manual_creator | query | 33 | "누가 만들었어?" |
| manual_usage | query | 33 | "사용법 알려줘" |
| manual_unsupported | exception | 30 | "이름 바꾸고 싶어" |

## 홈/알림 (3개, 107개)

| Intent | Action | 데이터 | 예시 |
|--------|--------|------:|------|
| home_status_query | query | 39 | "집 상태 어때" |
| notification_query | query | 35 | "알림 있어" |
| notice_query | query | 33 | "단지소식 어때" |

## 날씨 (4개, 5,776개)

| Intent | Action | 데이터 | 예시 |
|--------|--------|------:|------|
| weather_info | query | 5,668 | "날씨 어때" |
| weather_activity | query (judge: outdoor) | 44 | "세차해도 돼?" |
| weather_exception | exception | 33 | "다음 달 날씨" |
| weather_clothing | query (judge: clothing) | 31 | "뭐 입고 나가" |

## 미세먼지 (1개, 4,229개)

| Intent | Action | 데이터 | 예시 |
|--------|--------|------:|------|
| dust_query | query | 4,229 | "미세먼지 어때" |

## 뉴스 (3개, 586개)

| Intent | Action | 데이터 | 예시 |
|--------|--------|------:|------|
| news_play | query | 525 | "오늘 뉴스" |
| news_info | query | 31 | "뉴스 출처가 어디" |
| news_exception | exception | 30 | "매일 뉴스 브리핑" |

## 교통 (3개, 504개)

| Intent | Action | 데이터 | 예시 |
|--------|--------|------:|------|
| traffic_route_query | query | 371 | "교통 정보" |
| traffic_exception | exception | 103 | "택시 불러" |
| traffic_bus_query | query | 30 | "버스 정보" |

## 에너지 (4개, 155개)

| Intent | Action | 데이터 | 예시 |
|--------|--------|------:|------|
| energy_usage_query | query | 63 | "전기요금" |
| energy_alert_on | power_on | 32 | "사용량 초과하면 알려줘" |
| energy_goal_set | control | 30 | "사용량 목표 설정" |
| energy_alert_off | power_off | 30 | "에너지 알림 꺼줘" |

## 주식 (3개, 267개)

| Intent | Action | 데이터 | 예시 |
|--------|--------|------:|------|
| stock_index_query | query | 166 | "주식 어때" |
| stock_price_query | query | 66 | "삼성전자 주가" |
| stock_exception | exception | 35 | "주식 추천해줘" |

## 유가 (3개, 95개)

| Intent | Action | 데이터 | 예시 |
|--------|--------|------:|------|
| fuel_price_query | query | 33 | "기름값 어때" |
| fuel_exception | exception | 32 | "기름값 예측해줘" |
| fuel_station_search | query | 30 | "주유소 가격" |

## 의료 (3개, 280개)

| Intent | Action | 데이터 | 예시 |
|--------|--------|------:|------|
| medical_search | query | 214 | "병원 찾아줘" |
| medical_hours | query | 34 | "병원 오늘 진료해?" |
| medical_exception | exception | 32 | "병원 예약해줘" |

## 차량/주차 (5개, 168개)

| Intent | Action | 데이터 | 예시 |
|--------|--------|------:|------|
| ev_charging | query | 42 | "충전 상태 어때" |
| visitor_parking_register | control | 36 | "차량 등록해줘" |
| car_history_query | query | 30 | "차량 출입 정보" |
| car_history_delete | control | 30 | "출입 내역 삭제" |
| visitor_parking_query | query | 30 | "차량 등록 되어있나" |

## 시간 (1개, 330개)

| Intent | Action | 데이터 | 예시 |
|--------|--------|------:|------|
| time_query | query | 330 | "지금 몇 시야" |

## 커뮤니티/단지 (2개, 70개)

| Intent | Action | 데이터 | 예시 |
|--------|--------|------:|------|
| complex_info | query | 35 | "단지 정보" |
| community_query | query | 35 | "커뮤니티" |

---

## 데이터 불균형 요약

| 비율 | 값 |
|------|:---:|
| 최대 | weather_info: 5,668 (33.6%) |
| 최소 | heating_schedule_query: 29 (0.2%) |
| 최대:최소 비율 | **195:1** |
| 상위 2개 합산 | weather_info + dust_query = 9,897 (**58.6%**) |

→ 극심한 불균형 때문에 Multi-Head 방식으로 전환함.
