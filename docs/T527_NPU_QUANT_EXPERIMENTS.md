# T527 NPU 양자화 실험 매트릭스 (2026-05-07)

## 핵심 발견

**v28과 v46의 양자화 민감도가 매우 다름** — v46이 양자화에 훨씬 취약.

| 모델 | ONNX (no quant) | NPU uint8 | 손실 |
|---|---|---|---|
| **v28 단독** | 92.9% | **71.7%** | 21%p |
| **v46 단독** | 97.0% | **54.5%** | 42%p ⚠️ |
| Ensemble | 96.0% | 57.6% | 38%p |

v46이 generalization mixup 학습으로 dynamic range가 넓어 uint8 정밀도 부족. **ensemble이 v28을 끌어내림** — v46 망가짐 → 평균에서 v46의 잘못된 logit이 v28을 오버라이드.

## 실험 결과

### 시도 1: kl_divergence + 100 calib (이전)
- combo: 34.3%
- fn 정확도: 57.6%

### 시도 2: moving_average + 300 diverse calib (학습/골든 다양화)
- combo: 33.3% (변화 없음)
- fn 정확도: 57.6% (변화 없음)
- → **calib/algorithm 변경으로 안 풀림**

### 시도 3: v46 int16 NB 변환 (예정 평가)
- NB 빌드 성공 (`wksp_int16_nbg_unify/network_binary.nb` 2.7MB)
- 입력 형식: int16 fixed point (fl=15, 24576→49152 bytes)
- JNI 수정 필요 (현재 uint8 ByteArray 인터페이스만 지원)

## 단기 해결책 — v28 단독 + PostRules

v28 단독이 **71.7%**로 ensemble보다 나음. 후처리 룰(64개) 적용하면 80%+ 도달 가능.

```
v28 NB (1.36MB, ~1ms)
  → fn 71.7% (raw)
  → PostRules → 추정 80%+ (heat_control 보정 룰 다수)
```

vs ensemble (57.6% raw, 추정 65~70% with rules).

## 중기 해결책 — v46 int16 NB

int16은 65536단계 → uint8(256단계)의 256배 정밀도. v46의 wide dynamic range도 손실 없이 표현 가능. 다음 단계:

1. JNI에 `nativeRunInt16` 추가 또는 일반 byte 인터페이스로 변경
2. 입력 .bin int16 형식으로 재생성
3. v46 int16 + v28 uint8 mixed ensemble 평가

## 장기 — Acuity 옵션 더 탐색

- **pcq (per-channel quantization)** — Conv layer에 효과적
- **하이브리드 양자화** — 중요 layer만 int16, 나머지 uint8 → 크기와 정확도 균형
- **QAT (quantization-aware training)** — 모델 재학습 필요

## 결론

uint8 + asymmetric_affine으로 작은 CNN 모델 (특히 generalization 학습) 양자화는 한계. **v28만 단독 사용 + 후처리** 또는 **v46 int16** 둘 중 선택.
