# T527 NPU NLU 최종 결과

## 모델

| 항목 | 값 |
|------|-----|
| 아키텍처 | ko-sbert Embedding(frozen) + Linear(768→256) + CNN 4L |
| Layer | 4 (Conv1d + ReLU + BN + Dropout) |
| Embedding | **ko-sbert-sts 768dim 사전학습 (frozen)** |
| CNN dim | 256 |
| Conv kernels | 3, 5, 7, 3 |
| Dropout | 0.1 |
| Total Params | 25.98M (임베딩 24.58M frozen + CNN 1.40M trainable) |
| Intent 수 | **94개** |

## 파이프라인

```
STT 텍스트 → Tokenizer (CPU) → Embedding (CPU) → uint8 양자화 (CPU) → CNN body (NPU, ~300KB, ~1ms) → Intent
```

---

## 정확도

### 682개 검증 데이터 (CPU)

| 모델 버전 | Intent | 학습 데이터 | 정확도 | val_acc |
|----------|--------|-----------|--------|---------|
| v2 | 55개 | 1,908개 | 96.0% (655/682) | 99.5% |
| v4 | 54개 | 2,067개 | 99.4% (678/682) | 99.4% |
| v6 | 50개 | 2,636개 | 99.56% (679/682) | 98.4% |
| v8 | 50개 | 2,675개 | 100% (682/682) | 98.9% |
| v9 | 94개 | 2,997개 | 99.7% (680/682) | 97.2% |
| v10 | 94개 | 3,009개 | 100% (682/682)* | 96.7% |

*v10까지 학습/테스트 84.6% 겹침 → 수치 무의미

### 학습/테스트 분리 후 진짜 정확도 (겹침 1.0%)

| 모델 | Params | d_model | dropout | 학습 | 테스트 | 정확도 |
|------|--------|---------|---------|------|--------|--------|
| v11 4L | 4.40M | 128 | 0.0 | 13,595 | 2,452 | 94.62% |
| v13 4L | 4.40M | 128 | 0.0 | 13,780 | 2,452 | 95.31% |
| v13 8L | 6.51M | 128 | 0.0 | 13,780 | 2,452 | 93.64% |
| **v13 4L d256** | **9.40M** | **256** | **0.1** | **13,780** | **2,452** | **96.0%** |

### 르엘 시나리오 기준 최종 정확도 (테스트 겹침 0%)

**테스트셋:** 르엘 219개 + 간접 표현 + STT 오류 + 비유/관용 = 431개
**학습셋:** 르엘과 겹치지 않는 16,632개

| 모델 | 임베딩 | 르엘 219 | 간접 76 | STT 47 | 비유 43 | 전체 431 |
|------|--------|---------|--------|--------|--------|---------|
| CNN 4L d256 (랜덤) | 랜덤 128d | 92.2% | 93.4% | 76.6% | 37.2% | 80.7% |
| CNN 4L d256 (KLUE-RoBERTa) | KLUE 768d | 94.0% | - | - | - | - |
| **CNN 4L d256 (ko-sbert)** | **ko-sbert 768d** | **97.2%** | **96.1%** | **78.7%** | **51.2%** | **87.0%** |

**ko-sbert 사전학습 임베딩이 최종.** 랜덤 대비 르엘 +5%p, 비유 +14%p.

### Intent 정리 (v4→v8)

| 변경 | 내용 |
|------|------|
| 추가 (3개) | `doorlock_control`, `notice_query`, `password_change` |
| 통합 (6개) | `weather`→`weather_query`, `dust`→`dust_query`, `light_specific`→`light_on`, `heating_away`→`heating_schedule`, `temp_query`→`heating_query`, `ac_schedule`→`ac_mode` |
| 삭제 (2개) | `restaurant`, `travel` (kochat 전용, 르엘에 없음) |

### 모델 크기별 비교

레이어를 늘려도 정확도가 향상되지 않음. **데이터가 핵심, 모델 크기는 무관.**

| 모델 | Params | Val_Acc | NB 크기 | NPU 추론 |
|------|--------|---------|---------|---------|
| **cnn_4L** | **4.40M** | **0.9852** | **291KB** | **~1ms** |
| cnn_8L | 6.51M | 0.9847 | - | - |
| cnn_12L | 7.95M | 0.9847 | - | - |
| cnn_16L | 9.14M | 0.9837 | - | - |
| cnn_24L | 35.03M | 0.9837 | - | - |

### 21개 핵심 테스트 (NPU, v4 기준)

| 항목 | 값 |
|------|-----|
| NPU 정확도 | **100%** (21/21) |
| NPU vs CPU 일치 | **100%** |

### 간접 표현 (CPU)

| 발화 | 결과 | 정답 |
|------|------|------|
| 너무 추워 | heating_on | O |
| 얼어 죽겠다 | heating_on | O |
| 코가 시려 | heating_on | O |
| 어두워 | light_on | O |
| 캄캄해 | light_on | O |
| 눈이 침침해 | light_on | O |
| 너무 밝아 | light_off | O |
| 눈부셔 | light_off | O |
| 너무 더워 | ac_on | O |
| 찜통이야 | ac_on | O |
| 땀이 나 | ac_on | O |
| 공기가 탁해 | ventilation_on | O |
| 숨이 막혀 | ventilation_on | O |
| 냄새 나 | ventilation_on | O |
| 택배 왔어 | door_open | O |

### STT 오류 변형 (CPU)

| 발화 (STT 오류) | 결과 | 정답 |
|----------------|------|------|
| 남방 커줘 | heating_on | O |
| 보일라 커줘 | heating_on | O |
| 불커줘 | light_on | O |
| 부ㄹ켜줘 | light_on | O |
| 에에컨 켜줘 | ac_on | O |
| 완기 켜줘 | ventilation_on | O |
| 문열어 | door_open | O |
| 뭄열어줘 | door_open | O |

---

## 학습 데이터

| 데이터 | 크기 | 출처 |
|--------|------|------|
| 스마트홈 v8 | 2,675개 | 자체 생성 (50 intent) |
| kochat v6 | 9,996개 | github.com/hyunwoongko/kochat (weather_query, dust_query만) |
| **총** | **12,671개** | |

---

## Intent 목록 (50개)

| 카테고리 | Intent | 설명 |
|---------|--------|------|
| 조명 | light_on, light_off, light_dim, light_query, light_schedule | 조명 제어/조회/예약 |
| 난방 | heating_on, heating_off, heating_up, heating_down, heating_query, heating_schedule | 난방 제어/조회/예약 |
| 에어컨 | ac_on, ac_off, ac_temp, ac_mode, ac_wind, ac_query, ac_exception | 에어컨 제어/조회 |
| 환기 | ventilation_on, ventilation_off, ventilation_mode, ventilation_query | 환기 제어/조회 |
| 커튼 | curtain_control | 전동커튼 제어 |
| 출입 | door_open, doorlock_control | 공동현관/도어락 |
| 가스 | gas_off | 가스 밸브 |
| 보안 | security_mode, emergency | 외출모드/비상 |
| 엘리베이터 | elevator_control | 호출/조회 |
| 에너지 | energy_query | 사용량 조회 |
| 설정 | system_setting, alarm_setting, password_change | 시스템/알람/비밀번호 |
| 정보 | home_status_query, notification_query, notice_query, manual_query, complex_info | 상태/알림/단지정보 |
| 생활 | weather_query, dust_query, news_query, traffic_query, stock_query, fuel_query, medical_query, time_query, car_history, community_query, visitor_parking, ev_charging | 날씨/뉴스/교통 등 |

---

## NPU 변환 과정

1. PyTorch 학습 (GPU)
2. Embedding 분리 — Embedding은 CPU, Conv body만 ONNX export
3. Acuity import → uint8 양자화 → NB export
4. **입력을 uint8로 양자화해서 넣기** (float32 아님!)

### 핵심 발견

**이전 모든 NPU 실패의 원인은 입력 형식이었다.**
- float32를 그대로 vpm_run에 넣으면 → 결과 불일치
- uint8로 양자화해서 넣으면 → 완벽 일치
- Conformer STT에서 mel을 uint8로 넣는 것과 동일

---

## 파일 구조

```
t527-nlu/
├── checkpoints/
│   ├── cnn_4L_v8_best.pt             # 최종 모델 (100%)
│   └── label_map.json                # 50 intent 매핑
├── data/
│   ├── smarthome_intent_v8.csv       # 최종 학습 데이터 2,675개
│   ├── kochat_intent_v6.csv          # kochat 9,996개
│   ├── test_v8.csv                   # 검증 682개
│   ├── ruel_scenarios.csv            # 르엘 219개 시나리오
│   ├── indirect_expressions.csv      # 간접 표현 76개
│   └── stt_error_variants.csv        # STT 오류 47개
├── tokenizer/                        # BERT tokenizer
├── train_cnn_v6.py                   # PureCNN 학습 스크립트
├── docs/
│   ├── NLU_RESULTS.md                # 이 문서
│   ├── NPU_NLU_EXPERIMENTS.md        # NPU 실험 전수 조사
│   └── TRAINING_ROADMAP.md           # 학습 로드맵
└── README.md
```
