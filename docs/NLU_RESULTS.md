# T527 NLU 결과 요약

> 상세: [NLU_FINAL_REPORT.md](NLU_FINAL_REPORT.md)

## 모델

| 항목 | 값 |
|------|-----|
| 아키텍처 | ko-sbert Embedding(frozen,768) → Linear(768→256) → CNN 4L → FC(94) |
| Embedding | ko-sbert-sts 768dim (frozen, 24.58M params) |
| CNN | 4L Conv1d(k=3,5,7,3) + ReLU + BN + Dropout(0.1), d=256 |
| Total / Trainable | 25.98M / **1.40M** |
| Intent | **94개** |
| Tokenizer | BERT WordPiece 32000 vocab (ko-sbert와 동일) |

## 파이프라인

```
STT 텍스트 → Tokenizer(CPU) → Embedding(CPU, ko-sbert 768d) → Linear 768→256(CPU) → CNN body(NPU) → Intent(94개)
```

## 정확도 (르엘 시나리오 기준, 학습/테스트 겹침 0%)

| 테스트 소스 | 개수 | 정확도 | 설명 |
|------------|------|--------|------|
| **르엘 원문** | 217 | **94.0%** | 핵심 목표 |
| 간접 표현 | 76 | 98.7% | "너무 추워"→heating_on |
| STT 오탈자 | 47 | 93.6% | "남방 커줘"→heating_on |
| 비유/관용 | 43 | 51.2% | "냉장고야"→heating_on (학습에 없음) |
| 르엘 STT 변형 | 48 | 89.6% | 르엘 원문의 STT 오류 버전 |
| **전체** | **431** | **89.1%** | |

## Intent 94개

르엘 219개 시나리오를 전부 커버. intent 하나로 응답 액션 특정 가능.

상세: [intent_mapping_v2.csv](../data/intent_mapping_v2.csv)

| 카테고리 | Intent 수 | 예시 |
|---------|----------|------|
| 조명 | 6 | light_on, light_off, light_dim, light_brighten, light_query, light_schedule |
| 난방 | 8 | heating_on/off/up/down/query, heating_schedule_query/set/cancel |
| 에어컨 | 9 | ac_on/off/temp/mode/wind/query/exception, ac_mode_schedule/noroom |
| 환기 | 7 | ventilation_on/off/mode/query, ventilation_schedule_query/set, ventilation_exception |
| 커튼 | 5 | curtain_open/close/stop/query/schedule |
| 출입/가스 | 5 | door_open, doorlock_open/query, gas_close/query |
| 보안 | 4 | security_activate/query/return_set, emergency |
| 설정 | 8 | system_faq/volume_set/brightness_set/brightness_schedule/exception, alarm_query/set, password_change |
| 정보 | 9 | home_status_query, notification_query, notice_query, manual_capability/usage/creator/name/unsupported, complex_info |
| 에너지 | 5 | energy_usage_query/goal_set/alert_on/alert_off, ev_charging |
| 생활 | 28 | weather_info/clothing/activity/exception, dust_query, news_play/info/exception, traffic_route_query/bus_query/exception, stock_index/price_query/exception, fuel_price_query/station_search/exception, medical_search/hours/exception, time_query, car_history_query/delete, community_query, visitor_parking_query/register |
| **합계** | **94** | |

## 학습 데이터

| 소스 | 개수 | 비고 |
|------|------|------|
| Amazon MASSIVE 한국어 | ~3,300 | 15개 intent 매핑 |
| kochat | ~9,000 | weather_info + dust_query |
| 자체 생성 (LLM 증강 포함) | ~4,400 | 94개 intent 커버 |
| **총 학습** | **~16,700** | 테스트와 겹침 0% |

## 관련 문서

| 문서 | 내용 |
|------|------|
| [NLU_FINAL_REPORT.md](NLU_FINAL_REPORT.md) | 전체 진행 과정 + 상세 결과 |
| [PRETRAINED_EMBEDDING_TRANSFER.md](PRETRAINED_EMBEDDING_TRANSFER.md) | 사전학습 임베딩 전이학습 방법론 |
| [WHY_CNN_WORKS.md](WHY_CNN_WORKS.md) | CNN이 간접 표현 잡는 원리 |
| [NLU_LIMITATIONS_AND_PLAN.md](NLU_LIMITATIONS_AND_PLAN.md) | 구조적 한계 + 모델 비교 |
| [NPU_NLU_EXPERIMENTS.md](NPU_NLU_EXPERIMENTS.md) | NPU 양자화 실험 (BERT 9가지 실패, CNN 성공) |
