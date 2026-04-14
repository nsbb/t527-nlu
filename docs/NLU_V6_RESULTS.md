# NLU v6 결과 — CNN 5-Head Multi-head Classifier

## 아키텍처

```
ko-sbert Embedding (32000×768, frozen)
  → Linear 768→256
  → Conv1d(k=3) + BN + ReLU + Dropout (residual)
  → Conv1d(k=5) + BN + ReLU + Dropout (residual)
  → Conv1d(k=7) + BN + ReLU + Dropout (residual)
  → Conv1d(k=3) + BN + ReLU + Dropout (residual)
  → Global Mean Pool → [B, 256]
  → 5 Classification Heads:
     ├── fn_head (20): 19 devices + unknown
     ├── exec_head (5): control/query/judge/direct/clarify
     ├── direction_head (9): on/off/up/down/open/close/set/stop/none
     ├── param_head (5): none/temperature/brightness/mode/speed
     └── judge_head (5): none/outdoor/clothing/air_quality/cost
```

- Total: 26.1M params (1.5M trainable, embedding frozen)
- Input: 32 tokens (ko-sbert tokenizer)

## 학습 데이터

| 소스 | 수 |
|------|---:|
| GT 증강 (204 시나리오 × ~25 변형) | 4,999 |
| CNN 재매핑 보충 (fn당 500 목표) | 3,275 |
| Unknown (MASSIVE OOD + HA + hard neg) | 2,068 |
| **합계** | **10,342** |

fn별 분포: 118 (elevator) ~ 842 (ac_control), unknown 2,068

## 정확도

### 219개 원본 GT (260330 엑셀, 개발O)

```
fn 정확도: 213/219 = 97.3%
오류 6개 (전부 unknown→known 방향)
False rejection: 0건
```

### Validation Set

| Head | 정확도 |
|------|:------:|
| fn | 95.9% (known 97.2%, unknown 91.0%) |
| exec_type | 92.1% |
| param_direction | 97.5% |
| param_type | 97.2% |
| judge | 99.9% |
| **combo (5-head 전부)** | **86.5%** |

### Edge Cases

| 테스트 | 결과 |
|--------|:----:|
| 간접 표현 (찜통이야, 좀 춥다, 어두워, 나 나간다) | ✅ 전부 정확 |
| STT 오류 (남방 커줘, 에어콘 틀어) | ✅ 정확 |
| 판단형 (세차해도 돼?, 뭐 입고 나가?, 창문 열어도?) | ✅ 전부 정확 |
| Unknown 거부 (피자 주문, 노래 틀어, ㅋㅋㅋ) | ✅ 정확 |
| 미지원 액션 거부 (택시 불러, 병원 예약, 비번 바꿔) | ✅ response generator에서 처리 |
| False rejection (정상 발화→unknown 오분류) | ✅ 0건 |

## 진행 히스토리

| 버전 | 데이터 | 아키텍처 | fn | combo | 비고 |
|------|-----:|---------|:---:|:-----:|------|
| SAP v12 | 7,053 | Conformer 3L 8-head | 86% | 38.8% | 기존 |
| CNN v1 | 13,904 | CNN 4L 5-head | 96.5% | 90.8% | CNN 전환 |
| CNN v2 | 13,904 | CNN 4L 5-head | 95.1% | 89.1% | MASSIVE 추가 |
| CNN v3 | 10,342 | CNN 4L 5-head (20fn) | - | 89.5% | unknown 추가 |
| **CNN v6** | **10,342** | **CNN 4L 5-head (20fn)** | **97.3%*** | **86.5%** | **GT 기준 증강 + hard neg** |

*219개 원본 GT 기준

## 파일

| 파일 | 설명 |
|------|------|
| `scripts/model_cnn_multihead.py` | 5-head CNN 모델 정의 |
| `scripts/sap_inference_v2.py` | 대화형 추론 (rule slots + unknown + 미지원 처리) |
| `scripts/parse_gt_scenarios.py` | 219개 GT → known/unknown 분리 |
| `scripts/augment_per_scenario.py` | 시나리오당 100개 균등 증강 |
| `scripts/train_cnn_multihead.py` | 학습 스크립트 |
| `scripts/eval_ruel.py` | 평가 스크립트 |
| `checkpoints/cnn_multihead_v6.pt` | 최적 모델 |
| `data/train_final_v6.json` | 학습 데이터 |
| `data/gt_known_scenarios.json` | 204개 known GT |
| `data/gt_unknown_scenarios.json` | 15개 unknown GT |
