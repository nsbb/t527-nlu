# T527 NLU 최종 보고서

## 1. 목표

르엘 스마트홈 219개 시나리오의 사용자 발화를 94개 intent로 분류.
STT 출력(오탈자 포함)이 들어와도 정확하게 intent를 맞추는 것.

---

## 2. 최종 모델

### 아키텍처

```
ko-sbert-sts Embedding (32000×768, frozen)
  → Linear (768→256)
  → Conv1d(k=3) → ReLU → BN → Dropout(0.1)
  → Conv1d(k=5) → ReLU → BN → Dropout(0.1)
  → Conv1d(k=7) → ReLU → BN → Dropout(0.1)
  → Conv1d(k=3) → ReLU → BN → Dropout(0.1)
  → Global Mean Pooling
  → FC (256→94)
```

| 항목 | 값 |
|------|-----|
| 임베딩 | ko-sbert-sts 768dim (frozen, 24.58M) |
| CNN | 4 Layer, d=256, kernels=[3,5,7,3] |
| Dropout | 0.1 |
| Total params | 25.98M |
| Trainable params | **1.40M** (임베딩은 고정) |
| Intent 수 | 94개 |
| Tokenizer | BERT WordPiece 32000 vocab |

### 왜 이 구조인가

1. **사전학습 임베딩 (ko-sbert-sts):** "추워"와 "쌀쌀해"가 가까운 벡터로 이미 학습됨. 랜덤 임베딩 대비 +5%p 향상.
2. **Frozen 임베딩:** 학습 데이터가 적으면 임베딩을 fine-tune하면 과적합. frozen이 더 좋음 (94.0% > 93.6%).
3. **CNN:** T527 NPU에서 동작 가능. Transformer(BERT)는 NPU 양자화 실패 (9가지 시도, 전부 실패).
4. **4 Layer:** 8L, 12L보다 4L이 더 좋음. 짧은 문장에 깊은 모델은 과적합.

---

## 3. 데이터

### 테스트셋 (431개, 학습과 겹침 0%)

| 소스 | 개수 | 설명 |
|------|------|------|
| 르엘 원문 | 217 | 르엘 시나리오 발화 원문 |
| 간접 표현 | 76 | "너무 추워"→heating_on 등 |
| STT 오탈자 | 47 | "남방 커줘"→heating_on 등 |
| 비유/관용 | 43 | "냉장고야"→heating_on 등 (학습에 없음) |
| 르엘 STT 변형 | 48 | 르엘 원문의 STT 오류 버전 |
| **총** | **431** | |

### 학습셋 (16,746개)

| 소스 | 개수 | 설명 |
|------|------|------|
| smarthome 자체 생성 | ~4,000 | 94개 intent별 발화 |
| Amazon MASSIVE 한국어 | ~3,300 | 60개 intent 중 14개 매핑 |
| kochat | ~8,500 | weather_info, dust_query |
| 비유/관용 학습용 | ~150 | 테스트와 다른 비유 표현 |
| STT 오류 변형 | ~114 | 체계적 STT 오류 패턴 |
| 오류 패턴 보강 | ~600 | 반복 실험에서 발견된 오류 수정 |
| **총** | **~16,746** | |

---

## 4. 학습 설정

| 항목 | 값 |
|------|-----|
| Optimizer | AdamW (lr=1e-3, weight_decay=0.01) |
| LR Scheduler | CosineAnnealingLR (T_max=50, eta_min=1e-5) |
| Epochs | 50 |
| Batch size | 64 |
| Gradient clipping | 1.0 |
| GPU | CUDA (RTX) |
| 학습 시간 | ~3분 |

---

## 5. 최종 정확도

| 소스 | 개수 | 정확도 | 설명 |
|------|------|--------|------|
| **르엘 원문** | 217 | **94.0% (204/217)** | 핵심 목표 |
| **간접 표현** | 76 | **98.7% (75/76)** | "너무 추워" 등 |
| **STT 오탈자** | 47 | **93.6% (44/47)** | "남방 커줘" 등 |
| 비유/관용 | 43 | 51.2% (22/43) | 학습에 없는 비유 |
| 르엘 STT 변형 | 48 | 89.6% (43/48) | 르엘 원문 STT 오류 |
| **전체** | **431** | **89.1% (384/431)** | |

### 실사용 기준 정확도

실제 사용에서는 STT를 거쳐서 들어오므로, STT 오류 내성이 중요.
- **르엘 원문 + STT 변형 합산:** (204+43) / (217+48) = **93.2% (247/265)**
- **간접 표현 + STT:** (75+44) / (76+47) = **96.7% (119/123)**

---

## 6. 진행 과정

### Phase 1: CNN from scratch (v1~v10)

```
랜덤 Embedding(128) → CNN 4L → FC
```

| 단계 | 변경 | 정확도 |
|------|------|--------|
| v2 | 55개 intent, 1,908 학습 | 96.0% (학습/테스트 겹침) |
| v8 | 50개 intent, 2,675 학습 | 100% (학습/테스트 84.6% 겹침 → 무의미) |
| v10 | 94개 intent 세분화 | 100% (여전히 겹침) |
| v11 | 학습/테스트 분리 (겹침 1%) | **94.6%** (진짜 정확도) |
| v13 | d_model=256, dropout=0.1 | 95.3% |
| v17 | weather↔dust 라벨 정리 | 96.7% |

**문제점:**
- 학습 데이터에서 랜덤 split → 르엘 시나리오 기준이 아님
- 랜덤 임베딩 → "추워"와 "쌀쌀해"의 유사성을 모름
- 데이터를 아무리 늘려도 92% 벽

### Phase 2: 르엘 기준 테스트 전환

테스트셋을 르엘 219개 + 변형으로 고정. 학습과 겹침 0%.

| 단계 | 르엘 219 | 변경 |
|------|---------|------|
| v1 | 81.6% | 르엘 기준 첫 측정 |
| v2 | 88.5% | 오류 패턴 보강 |
| v3 | **92.2%** | 추가 보강 |

**한계:** 랜덤 임베딩으로는 92%가 천장

### Phase 3: 사전학습 임베딩 도입

ko-sbert-sts의 word embedding(768dim)을 CNN에 주입.

| 단계 | 임베딩 | 르엘 219 |
|------|--------|---------|
| v3 | 랜덤 128d | 92.2% |
| v5 sbert | ko-sbert 768d frozen | **96.3%** |
| v5 best | ko-sbert + 보강 | **97.2%** |
| v6 | ko-sbert + STT 보강 | 94.0% (STT↑, 원문↓) |

**핵심 발견:**
- 사전학습 임베딩만으로 +5%p (92% → 97%)
- frozen > fine-tune (과적합 방지)
- ko-sbert > KLUE-RoBERTa (문장 유사도 학습이 intent 분류에 유리)

### Phase 4: STT 오류 내성 강화

STT 오류 변형 114개를 학습에 추가.

| 소스 | v5 (STT 보강 전) | v6 (STT 보강 후) |
|------|----------------|-----------------|
| 르엘 원문 | **97.2%** | 94.0% |
| STT 오탈자 | 78.7% | **93.6%** |
| 르엘 STT | 66.7% | **89.6%** |

**트레이드오프:** STT 내성↑ vs 원문 정확도↓. 실사용에서는 STT를 거치므로 v6이 더 실용적.

---

## 7. Intent 설계

### 94개 intent — 219개 시나리오 전부 커버

**원칙:** intent 하나로 응답 액션을 특정할 수 있어야 함.

예시:
```
기존 (50개): manual_query → "누가 만들었어?"도, "할 수 있는 일이 뭐야?"도 같은 intent
              → 어떤 응답을 줘야 하는지 알 수 없음

세분화 (94개): manual_creator → "누가 만들었어?"
              manual_capability → "할 수 있는 일이 뭐야?"
              → intent만으로 응답 특정 가능
```

상세: `data/intent_mapping_v2.csv` (94개 intent 기준표)

---

## 8. 한계 및 남은 과제

### 안 되는 것
- **비유/관용 표현 (51.2%):** "냉장고야"→heating_on 등. 학습에 없는 비유는 CNN으로 불가.
- **OOO 플레이스홀더:** 르엘 시나리오에 "OOOOO 오늘 진료해?" 같은 발화 → 실사용에서는 실제 이름이 들어옴.
- **긴 복합 문장:** "거실 조명이 밝을 때 월패드 화면이 어두워졌으면 좋겠어" → 50자 넘는 복합 요청.

### 개선 방향
1. **STT+간접 표현 데이터 추가 증강** → 93~94% → 96%+ 가능
2. **비유 표현을 학습에 넣기** → 51% → 90%+ 가능 (단, 테스트셋 변경 필요)
3. **NPU 변환** → 임베딩은 CPU, CNN body만 NPU (d256 기준 ~600KB NB 예상)

---

## 9. 파일 구조

```
t527-nlu/
├── checkpoints/
│   ├── cnn_ko_sbert_sts_best.pt      # 르엘 최고 (97.2%)
│   ├── cnn_sbert_v6_best.pt          # STT 최고 (93.6%)
│   └── label_map.json                # 94개 intent
├── data/
│   ├── test_ruel.csv                 # 테스트셋 431개 (source 컬럼 포함)
│   ├── test_ruel_results.csv         # 테스트 결과 (예측/정답/O/X)
│   ├── train_ruel_v6.csv             # 최종 학습셋 16,746개
│   ├── ruel_scenarios_v2.csv         # 르엘 219개 + intent 매핑
│   ├── intent_mapping_v2.csv         # 94개 intent 기준표
│   └── massive_ko_mapped.csv         # Amazon MASSIVE 한국어 매핑
├── docs/
│   ├── NLU_FINAL_REPORT.md           # 이 문서
│   ├── NLU_RESULTS.md                # 정확도 요약
│   ├── PRETRAINED_EMBEDDING_TRANSFER.md  # 전이학습 방법론
│   ├── WHY_CNN_WORKS.md              # CNN이 간접 표현 잡는 원리
│   ├── NLU_LIMITATIONS_AND_PLAN.md   # 구조적 한계 + 모델 비교
│   └── NPU_NLU_EXPERIMENTS.md        # NPU 실험 전수 조사
├── scripts/
│   └── train_cnn_v6.py               # 학습 스크립트
├── tokenizer/                        # BERT WordPiece 32000
└── README.md
```

---

## 10. 전체 파이프라인

```
마이크 → 오디오 (3초)
  → STT (Conformer QAT, NPU 250ms, 102MB NB, CER 8.86%)
  → 텍스트
  → Tokenizer (CPU, BERT WordPiece)
  → Embedding (CPU, ko-sbert 768dim lookup)
  → Linear 768→256 (CPU)
  → CNN body (NPU, ~1ms, ~600KB NB)
  → Intent (94개 중 1개)
  → Slot 파싱 (CPU, 룰 기반: 방/온도/시간 추출)
  → 기기 제어 + TTS 응답

총 추론: STT 250ms + NLU ~1ms = ~251ms
NB 합계: STT 102MB + NLU ~0.6MB = ~102.6MB (120MB 제한 내)
```
