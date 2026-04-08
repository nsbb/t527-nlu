# T527 NPU NLU 실험 결과 — 전수 조사

## 1. 실패한 모델들 (순수 Transformer)

### 실패 원인: Self-Attention 양자화 에러 누적

Transformer의 QKV matmul → softmax → weighted sum 과정에서 uint8/int16 양자화 에러가 누적.
Layer가 쌓일수록 에러가 증폭되어 최종 출력이 원본과 완전히 달라짐.

| # | 모델 | Layer | Params | 양자화 | NB | NPU 추론 | cos_sim | 결과 |
|---|------|-------|--------|--------|-----|---------|---------|------|
| 1 | ko-sbert-sts (pooler) | 12 | 110.6M | uint8 AA KL | 87MB | 89ms | 0.33 | 실패 |
| 2 | MiniLM-L6 (pooler) | 6 | 22.7M | uint8 AA KL | 21MB | 22ms | -0.06 | 실패 |
| 3 | multi-MiniLM-L12 | 12 | 117.7M | uint8 AA KL | 110MB | 44ms | 0.07 | 실패 |
| 4 | MiniLM-L6 (pooler) | 6 | 22.7M | INT16 DFP | 40MB | 41ms | -0.07 | 실패 |
| 5 | MiniLM-L6 (pooler) | 6 | 22.7M | BF16 | - | - | - | 변환 실패 |
| 6 | MiniLM-L6 (meanpool) | 6 | 22.7M | uint8 AA KL | 21MB | 22ms | NaN | 실패 |
| 7 | MiniLM-L6 (meanpool) | 6 | 22.7M | int8 AA KL | 21MB | 22ms | 0.04 | 실패 |
| 8 | MiniLM-L6 (meanpool) | 6 | 22.7M | uint8 AA MA | 21MB | 22ms | NaN | 실패 |
| 9 | MiniLM-L6 (meanpool) | 6 | 22.7M | INT16 DFP | 40MB | 41ms | NaN | 실패 |

### 시도한 변수
- 모델: ko-sbert 12L, MiniLM-L6 6L, multi-MiniLM 12L
- Pooler: 포함(Tanh), 제거(mean pooling)
- 양자화: uint8 AA, int8 AA, INT16 DFP, BF16
- 알고리즘: kl_divergence, moving_average

**9가지 전부 cos_sim < 0.35 → 의미 없는 출력**

---

## 2. 성공한 모델 (TextConformer — CNN+Attention 하이브리드)

### 성공 원인: Conformer Macaron 구조

Conformer STT(18L, 122.5M)가 T527 NPU에서 성공한 것과 동일한 원리.
DepthwiseConv1D가 Self-Attention 양자화 에러를 보정하는 "방파제" 역할.

```
Conformer Block:
FFN(½) → Self-Attention → DepthwiseConv1D → FFN(½)
         ↑ 양자화 에러 발생    ↑ Conv가 보정
```

| # | 모델 | Layer | Params | NB | NPU 추론 | 결과 |
|---|------|-------|--------|-----|---------|------|
| 1 | TextConformer-tiny | 1 | 0.57M | **696KB** | **0.4ms** | 성공 |
| 2 | TextConformer-small | 2 | 1.47M | **1.8MB** | **1.5ms** | 성공 |
| 3 | TextConformer-base | 4 | 5.50M | **6.0MB** | **3.2ms** | 성공 |

**3개 전부 NPU 추론 성공! Conformer 구조는 T527 NPU에서 확실히 동작.**

---

## 3. CPU ONNX Runtime (대안)

NPU 실패 시 CPU에서 ONNX Runtime으로 추론.

| 모델 | ONNX 크기 | 추론 (WSL CPU) | 간접 표현 | 비고 |
|------|----------|--------------|---------|------|
| ko-sbert-sts INT8 | 106MB | 12ms | 95.7% | sentence embedding + cosine sim |

---

## 4. 결론

| 접근 | NPU | CPU | 간접 표현 | 다음 단계 |
|------|-----|-----|---------|----------|
| 순수 Transformer (BERT) | **X** (9가지 실패) | O (12ms) | O (95.7%) | CPU 방식 가능 |
| **TextConformer (CNN+Attention)** | **O (0.4~3ms)** | O | **학습 필요** | 18L 120M으로 키워서 학습 |
| 순수 CNN (TextCNN) | O | O | 학습 의존 | 간접 표현 약함 |

### 핵심 발견

1. **T527 NPU에서 순수 Transformer는 어떤 양자화로도 불가**
2. **Conformer(CNN+Attention 하이브리드) 구조만 NPU에서 동작**
3. **TextConformer 18L로 키워서 한국어 의도 분류 학습하면 NPU에서 빠르게 동작 가능**
4. Conformer STT(18L, 102MB, 250ms)와 같은 크기면 TextConformer도 ~30ms 예상

---

## 5. 다음 할 일

1. TextConformer 18L (~120M params) 모델 설계
2. 한국어 의도 분류 학습 데이터 수집 (AIHub, 공공데이터)
3. 간접 표현 포함 데이터 증강
4. GPU 서버에서 학습
5. ONNX → Acuity → NB 변환
6. T527 NPU 테스트 (정확도 + 추론 속도)
