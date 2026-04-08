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

## 2. 성공한 모델들

### 2-1. TextConformer (CNN+Attention 하이브리드)

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

### 2-2. Pure CNN (Conv1d + ReLU + BatchNorm) ★ 최종 채택

Self-Attention이 전혀 없는 순수 CNN. NPU에서 완벽 동작.

```
Embedding (CPU) → Conv1d(k=3) → ReLU → BN → Conv1d(k=5) → ReLU → BN
→ Conv1d(k=7) → ReLU → BN → Conv1d(k=3) → ReLU → BN → MeanPool → FC(55)
```

| # | 모델 | Layer | Params | NB | NPU 추론 | CPU 정확도 | NPU 정확도 |
|---|------|-------|--------|-----|---------|-----------|-----------|
| 1 | cnn_4L_v4 | 4 | 4.40M | **291KB** | **~1ms** | **99.4%** | **100% (21/21)** |
| 2 | cnn_8L | 8 | 6.51M | - | - | 98.5% | - |
| 3 | cnn_12L | 12 | 7.95M | - | - | 98.5% | - |
| 4 | cnn_16L | 16 | 9.14M | - | - | 98.4% | - |
| 5 | cnn_24L | 24 | 35.03M | - | - | 98.4% | - |

**핵심:** 레이어를 늘려도 정확도가 향상되지 않음. 4L이 최적.
데이터 증강(v2→v3→v4)이 정확도 향상의 핵심 (96.0% → 98.2% → 99.4%).

### 2-3. 큰 모델 NPU 테스트

TextConformer 18L과 Pure CNN body를 NPU에서 직접 테스트.

| 모델 | NB 크기 | NPU 추론 |
|------|---------|---------|
| Pure CNN body (4L sim) | 2.1MB | 870us |
| TextConformer 18L body (sim) | 111MB | 135ms |

---

## 3. CPU ONNX Runtime (대안)

NPU 실패 시 CPU에서 ONNX Runtime으로 추론.

| 모델 | ONNX 크기 | 추론 (WSL CPU) | 간접 표현 | 비고 |
|------|----------|--------------|---------|------|
| ko-sbert-sts INT8 | 106MB | 12ms | 95.7% | sentence embedding + cosine sim |

---

## 4. 결론

| 접근 | NPU | CPU | 간접 표현 | 결과 |
|------|-----|-----|---------|------|
| 순수 Transformer (BERT) | **X** (9가지 실패) | O (12ms) | O (95.7%) | 입력 형식과 무관하게 embedding 품질 저하 |
| TextConformer (CNN+Attention) | **O (0.4~3ms)** | O | O (학습) | NPU 동작 확인 |
| **Pure CNN 4L** | **O (~1ms)** | **O** | **O (99.4%)** | **★ 최종 채택** |

### 핵심 발견

1. **T527 NPU에서 순수 Transformer는 어떤 양자화로도 불가**
2. **Conformer(CNN+Attention 하이브리드), Pure CNN 모두 NPU에서 동작**
3. **Pure CNN 4L이 최적: 291KB NB, ~1ms, 99.4% 정확도**
4. **모든 NPU 실패의 진짜 원인은 입력 형식(float32→uint8 필요)**
5. **모델 크기보다 학습 데이터 증강이 정확도에 결정적**
