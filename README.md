# T527 온디바이스 NLU — Sentence Embedding + Cosine Similarity

월패드 STT 결과 텍스트 → **사용자 의도(intent) 파악** → 기기 제어.
간접 표현("너무 추워" → 난방 켜기) 포함 처리.

---

## 방식

```
사용자: "너무 추워"
    ↓
1. ko-sbert-sts로 문장 embedding (768차원 벡터)
    ↓
2. 미리 계산된 intent별 대표 문장 embedding과 cosine similarity 비교
    ↓
3. 가장 유사한 intent 선택 → heating_on (유사도 1.000)
    ↓
4. 기기 제어 명령 실행
```

기존 intent classification(KoELECTRA)과 달리, **의미적 유사도 기반**이라 간접 표현도 자연스럽게 처리.

---

## 모델

| 항목 | 값 |
|------|-----|
| 모델 | [jhgan/ko-sbert-sts](https://huggingface.co/jhgan/ko-sbert-sts) |
| 기반 | KLUE-RoBERTa-base |
| 파라미터 | 110.6M |
| Embedding 차원 | 768 |
| ONNX FP32 | 422MB |
| **ONNX INT8** | **106MB** |
| 추론 (WSL CPU) | **12ms/query** |
| 한국어 | 최적 (34GB 한국어 사전학습 + STS fine-tune) |

---

## 테스트 결과

### 간접 표현 테스트 (23개)

**정확도: 22/23 (95.7%)**

| 발화문 | 매칭 intent | 유사도 | 정답 |
|--------|-----------|-------|------|
| 너무 추워 | heating_on | 1.000 | O |
| 집이 너무 춥다 | heating_on | 0.836 | O |
| 얼어 죽겠다 | heating_on | 1.000 | O |
| 너무 더워 | ac_on | 1.000 | O |
| 어두워 | light_on | 1.000 | O |
| 안 보여 | light_on | 1.000 | O |
| 공기가 탁해 | ventilation_on | 1.000 | O |
| 답답해 | ventilation_on | 1.000 | O |
| 보일러 좀 켜줄래 | heating_on | 0.941 | O |
| 에어컨 틀어줘 | ac_on | 0.953 | O |
| 문 열어 | door_open | 0.928 | O |
| 지금 몇 시야 | time_query | 1.000 | O |

유일한 오류: "거실이 좀 어두침침한데" → ventilation_on (X, light_on이어야)

### 르엘 시나리오 테스트 (219개)

| 유사도 범위 | 개수 | 비율 |
|-----------|------|------|
| ≥ 0.7 (높은 확신) | 95개 | 43% |
| 0.5 ~ 0.7 (보통) | 45개 | 21% |
| < 0.5 (낮은 확신) | 79개 | 36% |

**낮은 확신 원인:** 대표 문장 DB에 해당 도메인(전기차, 차량등록, 환경설정 등)이 없음.
대표 문장 추가하면 정확도 올라감.

---

## vs KoELECTRA (기존 방식)

| | KoELECTRA | **ko-sbert-sts** |
|---|---|---|
| 방식 | Intent Classification | **Sentence Embedding + Cosine Similarity** |
| 간접 표현 | X (안 됨) | **O (95.7%)** |
| 새 intent 추가 | 재학습 필요 | **대표 문장만 추가** |
| 모델 크기 | 36MB (FP32) | 106MB (INT8) |
| 추론 | 80ms CPU | **12ms CPU** |
| 한국어 | 최적 | 최적 |

---

## 작동 원리

### 1. Intent별 대표 문장 정의

```python
intents = {
    "heating_on": ["난방 켜줘", "보일러 켜줘", "너무 추워", "춥다", "얼어 죽겠다"],
    "light_on": ["불 켜줘", "조명 켜줘", "어두워", "안 보여", "밝게 해줘"],
    "ac_on": ["에어컨 켜줘", "너무 더워", "시원하게 해줘"],
    ...
}
```

### 2. 대표 문장 embedding 사전 계산 (서비스 시작 시 1회)

```python
ref_embeddings = {}
for intent, sentences in intents.items():
    ref_embeddings[intent] = [encode(s) for s in sentences]
```

### 3. 사용자 발화 → intent 매칭 (실시간)

```python
query_emb = encode("너무 추워")  # 12ms
for intent, embs in ref_embeddings.items():
    similarity = max(cosine_sim(query_emb, e) for e in embs)
# → heating_on (1.000)
```

---

## 현재 정의된 Intent (39개)

| 도메인 | Intent 수 | 예시 |
|--------|----------|------|
| 조명 | 6 | light_on, light_off, light_dim, light_query, light_schedule |
| 난방 | 7 | heating_on, heating_off, heating_up, heating_down, heating_query, heating_schedule, heating_away |
| 에어컨 | 8 | ac_on, ac_off, ac_temp, ac_mode, ac_wind, ac_query, ac_schedule, ac_exception |
| 환기 | 3 | ventilation_on, ventilation_off, ventilation_query |
| 보안/도어 | 3 | door_open, gas_off, security_mode |
| 커튼 | 1 | curtain_control |
| 정보 | 7 | weather_query, dust_query, news_query, traffic_query, energy_query, temp_query, time_query |
| 시스템 | 3 | manual_query, home_status_query, notification_query |
| 기타 | 1 | elevator_control, alarm_setting |

---

## T527 배포 계획

### NPU 배포 (권장)
```
STT 추론 → NPU에서 STT NB 내림 → NLU NB 올림 → NPU 추론 → 결과
```
- STT(102MB)와 NLU를 시간차로 NPU 공유
- NLU NB 크기: ~50-60MB (INT8 양자화 후 추정)
- Acuity import 시도 필요

### CPU 배포 (대안)
```
STT (NPU) → NLU (ONNX Runtime CPU) → 12ms
```
- ONNX INT8 106MB → CPU에서 12ms
- NPU 변환 실패 시 이 방식

---

## 파일 구조

```
t527-nlu/
├── README.md
├── .gitignore              # *.onnx 제외
├── tokenizer/              # ko-sbert-sts tokenizer
│   ├── vocab.txt
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
├── ko_sbert_sts.onnx       # FP32 422MB (gitignore)
└── ko_sbert_sts_int8.onnx  # INT8 106MB (gitignore)
```

---

## 다음 할 일

1. [ ] 219개 시나리오 전체에 대표 문장 확장 (특히 전기차, 차량등록, 환경설정)
2. [ ] T527 디바이스 ONNX Runtime 추론 속도 측정
3. [ ] Acuity import → NPU NB 변환 시도
4. [ ] Slot 추출 (공간명, 온도, 기기 등) — Regex 기반
5. [ ] t527_vad_service에 NLU 모듈 통합

---

## NPU 변환 결과 (실패)

| 항목 | 값 |
|------|-----|
| NB 크기 | 87MB |
| NPU 추론 | 89ms |
| **NPU vs CPU cosine similarity** | **0.33 (심각한 왜곡)** |

**NPU uint8 양자화가 embedding 정밀도를 파괴.** Pooler output(Tanh → [-1,1])이 uint8(256단계)로는 정밀도 부족.

wav2vec2 Transformer 양자화 실패와 동일 원인: **Transformer self-attention의 양자화 에러 누적.**

### 결론

- **NPU 불가** — embedding 왜곡으로 cosine similarity 무의미
- **CPU ONNX Runtime 사용** — INT8 106MB, WSL 12ms, T527 예상 50-100ms
- Conformer STT가 NPU에서 되는 이유: CNN이 Attention 양자화 에러를 보정 (Macaron 구조)
- 순수 Transformer(BERT)는 T527 NPU에서 양자화 실패

---

## NPU 양자화 전수 조사 결과

MiniLM-L6 (6 layer, 22.7M params) 기준. pooler 포함/제거, 양자화 방식 전부 시도.

| # | 모델 | 양자화 | NB | NPU 추론 | cos_sim |
|---|------|--------|-----|---------|---------|
| 1 | ko-sbert (12L) | uint8 AA KL | 87MB | 89ms | 0.33 |
| 2 | MiniLM-L6 pooler (6L) | uint8 AA KL | 21MB | 22ms | -0.06 |
| 3 | multi-MiniLM-L12 (12L) | uint8 AA KL | 110MB | 44ms | 0.07 |
| 4 | MiniLM-L6 pooler (6L) | **INT16 DFP** | 40MB | 41ms | **-0.07** |
| 5 | MiniLM-L6 pooler (6L) | BF16 | - | - | 변환 실패 |
| 6 | MiniLM-L6 meanpool (6L) | uint8 AA KL | 21MB | 22ms | NaN (출력 0) |
| 7 | MiniLM-L6 meanpool (6L) | int8 AA KL | 21MB | 22ms | 0.04 |
| 8 | MiniLM-L6 meanpool (6L) | uint8 AA MA | 21MB | 22ms | NaN |
| 9 | MiniLM-L6 meanpool (6L) | INT16 DFP | 40MB | 41ms | NaN |

**9가지 조합 전부 실패.** cos_sim 최대 0.33 (의미 없음).

### 시도한 변수

- **모델**: ko-sbert 12L, MiniLM-L6 6L, multi-MiniLM 12L
- **Layer 수**: 6, 12
- **Pooler**: 포함(Tanh), 제거(mean pooling)
- **양자화 방식**: asymmetric_affine, dynamic_fixed_point
- **비트**: uint8, int8, int16, bf16
- **알고리즘**: kl_divergence, moving_average

### 결론

**T527 Vivante NPU에서 BERT/Transformer 계열은 어떤 양자화로도 동작하지 않음.**

원인: Self-Attention의 QKV matmul → softmax → weighted sum 과정에서 양자화 에러가 누적.
Layer 수를 줄여도(12→6), pooler를 제거해도, INT16으로 올려도 해결 안 됨.

Conformer STT가 되는 이유: Conv1D가 Attention 에러를 보정하는 "방파제" 역할.
순수 Transformer에는 이 보정 메커니즘이 없음.

### 확정 방식

**CPU ONNX Runtime** — ko-sbert-sts INT8 106MB, 12ms/query (WSL), T527 예상 50-100ms
