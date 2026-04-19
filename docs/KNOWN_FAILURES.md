# v28 알려진 실패 패턴 (36개)

v28 모델이 틀리는 표현들. 학습 데이터에 없거나, 다른 fn과 혼동되는 패턴.
패치 추가 시 기존 패턴이 regression되므로 v28에서는 수정하지 않음.

## 카테고리별

### 학습 데이터 없음 (14개)
| 발화 | 기대 | 실제 | 원인 |
|------|------|------|------|
| 22도 | ac_control | market_query | 숫자만으로는 의도 파악 불가 |
| 온풍기 켜 | heat_control | ac_control | "온풍기" 학습 없음 |
| 이비인후과 | medical_query | unknown | 과 이름 학습 부족 |
| 에어컨 이십삼도 | ac_control | unknown | 한글 숫자 학습 없음 |
| 미세문지 | weather_query | door_control | STT 오류 학습 없음 |
| 손풍 해줘 | ac_control | weather_query | "손풍" STT 오류 학습 없음 |
| 거시 볼 커줘 | light_control | unknown | 심한 STT 오류 |
| 119 | unknown | traffic_query | 숫자만 |
| 에어컨 필터 언제 바꿔야 해 | ac_control | unknown | 유지보수 질문 학습 없음 |
| 비 올 확률 | weather_query | unknown | "확률" 학습 부족 |
| 준공일 언제 | home_info | unknown | 단지정보 학습 부족 |
| 하이 원더 | system_meta | market_query | 웨이크워드 학습 없음 |
| 빨래 해줘 | unknown | weather_query | "빨래" 학습 없음 |
| 30분 후에 꺼줘 | ac_control | energy_query | 시간조건 발화 학습 부족 |

### fn 경계 혼동 (14개)
| 발화 | 기대 | 실제 | 혼동 이유 |
|------|------|------|----------|
| 왔어 | security_mode | door_control | "왔어"=도착 → door 연상 |
| 실내 온도 너무 높아 | ac_control | heat_control | "온도"=난방 연상 |
| 안방이 좀 쌀쌀해 | heat_control | light_control | "안방이 좀"=light 연상 |
| 주방이 좀 텁텁해 | vent_control | light_control | "주방이 좀"=light 연상 |
| 폭염 주의보야 | weather_query | ac_control | "폭염"=에어컨 연상 |
| 단지 소식 | home_info | news_query | "소식"=뉴스 연상 |
| 오늘 장 어때 | market_query | weather_query | "어때"=날씨 연상 |
| 실내 온도 몇 도 | heat_control | weather_query | "몇 도"=날씨 연상 |
| 공기 좀 바꿔줘 | vent_control | weather_query | "공기"=미세먼지 연상 |
| 교통 정보 알려줘 | traffic_query | vehicle_manage | "정보"=차량 연상 |
| 기능 안내 | system_meta | home_info | "안내"=home_info 연상 |
| 화면 밝기 올려 | home_info | light_control | "밝기"=조명 연상 |
| 알람 꺼 | schedule_manage | energy_query | "꺼"=energy 연상 |
| 에어컨이랑 선풍기 뭐가 좋아 | unknown | ac_control | "에어컨" 키워드 |

### 극한 표현 (8개)
| 발화 | 기대 | 실제 | 원인 |
|------|------|------|------|
| 주식 장 마감했어? | market_query | unknown | 금융 용어 학습 부족 |
| 유가 전망 | market_query | unknown | "전망" 학습 부족 |
| 오늘 사건사고 | news_query | unknown | "사건사고" 학습 부족 |
| 헬스장 언제 열어 | home_info | unknown | 시설 운영시간 학습 부족 |
| 관리비 | energy_query | unknown | "관리비" 학습 없음 |
| 에어컨 좀 켜주시면 감사 | ac_control | unknown | 극존칭 학습 부족 |
| 불 왜 이래 | light_control | unknown | 불평형 학습 부족 |
| 길 어때 | traffic_query | unknown | "길"만으로 교통 연상 부족 |

## 향후 개선 방향

1. **학습 데이터 추가 시 regression 최소화**: 현재 v28 패치 접근법의 한계. 전체 데이터를 재설계해야 함.
2. **경계 혼동**: fn 간 의미 경계가 모호한 발화들은 모델만으로 해결 어려움. 규칙 보정 또는 멀티턴 DST 필요.
3. **STT 오류 내성**: "미세문지", "손풍" 같은 패턴은 STT 전처리 사전으로 해결 가능.
4. **한글 숫자**: "이십삼도" 같은 한글 숫자는 전처리에서 아라비아 숫자로 변환하면 해결.
