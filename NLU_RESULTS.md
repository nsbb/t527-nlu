# T527 NPU NLU 최종 결과

## 모델

| 항목 | 값 |
|------|-----|
| 아키텍처 | Pure CNN (Conv1d + ReLU + BatchNorm) |
| Layer | 4 |
| Params | 4.4M |
| Embedding | 128dim, CPU에서 처리 |
| Conv filters | 128 |
| Conv kernels | 3, 5, 7, 3 (반복) |
| NB 크기 | **291KB** |
| NPU 추론 | **~1ms** |
| Intent 수 | 55개 |

## 파이프라인

```
STT 텍스트 → Tokenizer (CPU) → Embedding (CPU) → uint8 양자화 (CPU) → CNN body (NPU, 291KB, ~1ms) → Intent
```

---

## 정확도

### 682개 검증 데이터 (CPU)

| 항목 | 값 |
|------|-----|
| 전체 정확도 | **96.0%** (655/682) |
| val_acc (학습) | 99.5% |

### 21개 핵심 테스트 (NPU)

| 항목 | 값 |
|------|-----|
| NPU 정확도 | **100%** (21/21) |
| NPU vs CPU 일치 | **100%** |

### 간접 표현 (CPU)

| 발화 | 결과 | 정답 |
|------|------|------|
| 너무 추워 | heating_on | O |
| 얼어 죽겠다 | heating_on | O |
| 집이 너무 춥다 | heating_on | O |
| 어두워 | light_on | O |
| 캄캄해 | light_on | O |
| 너무 밝아 | light_off | O |
| 너무 더워 | ac_on | O |
| 찜통이야 | ac_on | O |
| 시원하게 해줘 | ac_on | O |
| 공기가 탁해 | ventilation_on | O |
| 답답해 | ventilation_on | O |
| 숨이 막혀 | ventilation_on | O |
| 택배 왔어 | door_open | O |

### STT 오류 변형 (CPU)

| 발화 (STT 오류) | 결과 | 정답 |
|----------------|------|------|
| 남방 커줘 | heating_on | O |
| 보일라 커줘 | heating_on | O |
| 불커줘 | light_on | O |
| 에에컨 켜줘 | ac_on | O |
| 완기 켜줘 | ventilation_on | O |
| 문열어 | door_open | O |

---

## 학습 데이터

| 데이터 | 크기 | 출처 |
|--------|------|------|
| 스마트홈 증강 | 1,908개 | 자체 생성 (51 intent) |
| STT 오류 변형 | 47개 | 자체 생성 |
| kochat | 19,992개 | github.com/hyunwoongko/kochat |
| **총** | **21,900개** | |

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

## 남은 오류 (27/682)

대부분 **heating_on vs heating_up** 구분 문제:
- "보일러 올려줘" → heating_on (모델) vs heating_up (정답)
- 학습 데이터에 "보일러 올려줘"→heating_up 추가하면 해결

---

## 파일 구조

```
t527-nlu/
├── checkpoints/
│   ├── cnn_4L_v2_best.pt          # 학습된 모델
│   └── label_map.json             # 55 intent 매핑
├── data/
│   ├── smarthome_intent_v2.csv    # 학습 데이터 1,908개
│   ├── kochat_intent.csv          # kochat 19,992개
│   ├── test_1000_cases_fixed.csv  # 검증 682개
│   ├── ruel_scenarios.csv         # 르엘 219개 시나리오
│   ├── indirect_expressions.csv   # 간접 표현 76개
│   └── stt_error_variants.csv     # STT 오류 47개
├── tokenizer/                     # BERT tokenizer
├── train_textconformer.py         # 학습 스크립트
├── NPU_NLU_EXPERIMENTS.md         # NPU 실험 전수 조사
├── NLU_RESULTS.md                 # 이 문서
└── TRAINING_ROADMAP.md            # 학습 로드맵
```
