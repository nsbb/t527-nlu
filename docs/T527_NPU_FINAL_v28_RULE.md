# T527 NPU 최종 권장: v28 단독 + PostRules (2026-05-07)

## 결론

**현재 NPU에서 가장 정확한 구성: v28 단독 NB + PostRulesV4** — fn 78.8%, combo 57.6%.

ensemble은 v46 양자화 망가짐으로 인해 v28 단독보다 못함.

## 골든셋 99 전체 비교

| 모드 | fn 정확도 | combo |
|---|---|---|
| v28-raw | 71.7% | 51.5% |
| **v28-rule** ⭐ | **78.8%** | **57.6%** |
| v46-raw | 54.5% | 31.3% |
| v46-rule | 59.6% | 37.4% |
| ens-raw | 57.6% | 33.3% |
| ens-rule | 64.6% | 39.4% |
| (참고) ONNX ensemble CPU + rules | 97% | 88.9% |

**Latency: 2.13 ms/sample** (NPU 2-NB) → v28만 쓰면 1ms

## 분석

### v28-rule이 ensemble보다 나은 이유
- v46이 mixup generalization 학습 → dynamic range 넓음 → uint8 양자화로 -42%p
- v28은 패턴 매칭 위주 학습 → dynamic range 좁음 → uint8에서 -21%p
- **logit 평균이 v46의 잘못된 강한 unknown 신호를 v28에 섞어 떨어뜨림**

### PostRules 효과
- v28: 71.7 → 78.8% (+7.1%p)
- v46: 54.5 → 59.6% (+5.1%p)
- ens: 57.6 → 64.6% (+7.0%p)

PostRules는 일관되게 +5~7%p 보완하지만 양자화 손실 절반밖에 못 메움.

## 권장 디바이스 구성

```
Token IDs
  → KoELECTRA token_emb (CPU, embedding lookup table)
  → uint8 quantize
  → cnn_body_v28.nb (NPU, 1ms)
  → 5-head logits
  → PostRulesV4 (Kotlin)
  → fn/exec/dir/param/judge
```

## 미해결 — v46 양자화 문제

v46의 -42%p 손실 회복 옵션:
1. **v46 int16 NB** 변환 완료 (`wksp_int16_nbg_unify/`, 2.7MB) — JNI int16 입력 인터페이스 필요
2. **v46 PCQ (per-channel quantize)** — 미시도
3. **v46 재학습** (mixup 빼고) — 모델 변경 필요

## 다음 작업

1. JNI에 int16 byte input 지원 (현재 `nativeRunUint8` → `nativeRunBytes`로 일반화)
2. v46 int16 NB로 v28-uint8 + v46-int16 mixed ensemble 평가
3. 만약 v46-int16이 90%+ 회복하면 ensemble 다시 의미 있어짐
