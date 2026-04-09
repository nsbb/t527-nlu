# Semantic Action Parser 결과 요약

## 모델

| 항목 | 값 |
|------|-----|
| 아키텍처 | TextConformer 3L (ko-sbert emb frozen + Conformer + 7-head FC) |
| Embedding | ko-sbert-sts 768dim → Linear 768→256 (frozen) |
| Encoder | Conformer Block × 3 (d=256, nh=4, ff=1024, k=31) |
| Head | 7개 독립 FC (exec/fn/room/param/api/judge/multi) |
| Total Params | 29.2M (trainable 4.6M) |
| 학습 데이터 | 7,053개 |

## 7-Head 구조

```
TextConformer Encoder
     ↓ CLS token
├── exec_head (5): query_then_respond / control_then_confirm / query_then_judge / direct_respond / clarify
├── fn_head (15): light/heat/ac/vent/gas/door/curtain/elevator/security/schedule/weather/news/traffic/energy/info
├── room_head (8): living / kitchen / bedroom_main/sub / all / external / none / ambiguous
├── param_head (8): none / temperature / brightness / mode / speed / direction / time / keyword
├── api_head (7): none / inbase_device / weather_api / news_api / traffic_api / energy_api / local_info_api
├── judge_head (5): none / outdoor_activity / clothing / air_quality / cost_trend
└── multi_head (2): single / composite
```

## 정확도 (르엘 219개 시나리오)

| Head | 정확도 |
|------|--------|
| exec_type | 79% |
| fn | **86%** |
| room | 77% |
| param_type | 78% |
| api | **80%** |
| judge | **99%** |
| multi_action | 99% |
| **fn + exec** | **69.4%** |
| **combo (all 7 heads)** | **38.8%** |

## 핵심 성과

| 발화 | fn | exec | judge | 비고 |
|------|-----|------|-------|------|
| 지금 온도 어떠니 | heat_control ✅ | query ✅ | | **어미 변형 해결** |
| 찜통이야 | ac_control ✅ | confirm ✅ | | **비유 표현 해결** |
| 거실 에어컨 제습 23도 | ac_control ✅ | confirm ✅ | | **안 본 조합 일반화** |
| 세차해도 돼? | weather ✅ | judge ✅ | outdoor ✅ | **판단형 해결** |
| 뭐 입고 나가? | weather ✅ | judge ✅ | clothing ✅ | **옷차림 판단** |
| 나 나간다 | security ✅ | confirm ✅ | | **간접 의미 이해** |
| 누가 만들었어 | info ✅ | direct ✅ | | **고정 응답** |

## Intent 분류 대비 장점

| | Intent 분류 (94개) | SAP (7-head) |
|---|---|---|
| 어미 변형 | ❌ "어떠니"→실패 | ✅ heat_control로 정확 |
| 비유 표현 | △ 학습한 것만 | ✅ 구조적 분해로 일반화 |
| 안 본 조합 | ❌ 새 intent 필요 | ✅ 각 축 독립 예측 |
| 판단형 발화 | ❌ intent 추가 필요 | ✅ judge_head로 통합 |
| 새 디바이스 | N개 intent 추가 | fn_head에 1개 추가 |

## 실행

```bash
# 대화형 모드
python3 scripts/sap_inference.py

# 커맨드라인
python3 scripts/sap_inference.py "거실 에어컨 23도로 맞춰줘"
```

## 진행 히스토리

| 버전 | combo | fn+exec | fn | api | 데이터 |
|------|-------|---------|-----|-----|--------|
| v1 | 5.0% | - | - | - | 1,211 |
| v3 | 22.4% | - | - | - | 13,495 |
| v5 | 22.8% | 62.1% | 71% | 57% | 6,120 |
| v7 | 24.2% | 63.5% | 82% | 68% | 6,228 |
| v9 | 29.7% | 66.7% | 84% | 66% | 6,878 |
| v10 | 31.5% | 64.8% | 82% | 77% | 6,878 |
| **v11** | **38.8%** | **67.6%** | **86%** | 76% | 6,951 |
| v12 | 37.0% | 69.4% | 86% | **80%** | 7,053 |
