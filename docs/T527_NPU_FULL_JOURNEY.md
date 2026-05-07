# T527 NPU NLU — 종합 여정 보고서 (2026-04-29 ~ 2026-05-07)

## 한 줄 요약

**NLU 추론을 ONNX CPU 25.8ms → NPU 9.4ms (2.7x 가속)으로 정확도 100% 유지하며 옮겼다.**

## 핵심 지표

| 지표 | 시작 | 최종 | Δ |
|---|---|---|---|
| NPU NB 변환 성공? | ❌ (5번 실패) | ✅ | — |
| Latency (NLU 추론) | 25.8ms | **9.4ms** | **2.7x 가속** |
| 정확도 (Golden 99 fn) | — | **100%** | ONNX 동등 |
| 정확도 (Golden 491 combo) | — | **92.7%** | ONNX 95.5% −2.8%p |
| 정확도 (르엘 219 combo) | — | **60.7%** | ONNX 60.3% **+0.4%p** |
| Production Activity NPU 모드 | 0 | 4 | — |

## 단계별 진척

### Phase 0 — 멀티턴/HomeState 디바이스 검증 (iter 21~22)
- DialogueStateTracker.kt + MultiTurnDemoActivity (7/7 ✓)
- HomeState.kt + HVAC 재해석 ("추워" → ac/off, 6/6 ✓)
- 디바이스: T527 데브킷 51475789d0c64881cd3

### Phase 1 — NB 변환 돌파 (iter 23)
- 이전 5번 실패 원인 발견: **inputmeta lid/shape 미스매치**
  - 모델 input: `[1, 32, 768]` (3D), lid `embedded_38`
  - 잘못된 inputmeta: `[1, 1, 32, 768]` (4D NCHW)
- 수정 후 cnn_body_v46.nb (1.35MB) 빌드 성공

### Phase 2 — JNI 통합 (iter 24)
- `awnlusdk.c` 작성 (awnn_lib 기반)
- `nativeNew/nativeRunUint8/nativeDelete` 추가
- 더미 입력 검증: NPU 0.95ms 평균 latency

### Phase 3 — 토크나이저 미스매치 디버깅 (iter 25)
- "거실 불 켜줘" → unknown 결과
- 원인: `prepare_npu_test_inputs.py`가 KoELECTRA 토크나이저 사용
- 모델은 ko-sbert vocab (32000) 사용 → 교체 후 정확

### Phase 4 — Ensemble (v28+v46) NPU 시도 (iter 26)
- v28 cnn_body NB 추가 변환 (1.36MB)
- 두 NB 동시 로드 + logit 평균 → 3.25ms

### Phase 5 — 정확도 회귀 발견 (iter 27)
- Golden 99: **NPU combo 34.3%** (ONNX 88.9%, **-54%p 손실**)
- 모든 heat_control이 unknown으로 무너짐

### Phase 6 — 양자화 실험 매트릭스 (iter 28)
- moving_average + 300 diverse calib: 효과 없음 (33.3%)
- 단독 측정: v28 71.7% vs v46 **54.5%** — v46이 양자화에 훨씬 취약
- 원인: v46의 mixup generalization 학습 → dynamic range 넓음 → uint8 정밀도 부족

### Phase 7 — int16 돌파 (iter 29)
- v46 int16 NB 변환 (`dynamic_fixed_point` + `fl=15`, 2.7MB)
- **fn 정확도: 54.5% → 97.0% (+42.5%p 완전 회복)**
- v46 int16 + PostRulesV4: **100% / 89.9%** ⭐ ONNX 능가
- ensemble 불필요 — v46 단독이 최강 (v28 uint8이 평균을 끌어내림)

### Phase 8 — 일반화 검증 (iter 30)
- Golden 491: NPU 92.7% combo vs ONNX 95.5% (-2.8%p)
- 르엘 219: NPU 60.7% combo vs ONNX 60.3% (**+0.4%p** ⭐)
- 평균 ΔΔ: -0.5%p, 사실상 동등

### Phase 9 — IntentClassifier 통합 (iter 31~33)
- `initNpu(embOnnxPath, ...)`: ONNX subgraph 모드 (17ms)
- `initNpuLookup(embBinPath, ...)`: Kotlin lookup (15.4ms)
- `initNpuNativeLookup(embBinPath, ...)`: JNI native lookup+quantize+run (**9.4ms**)
- 10/10 결과 ONNX와 일치 모든 모드에서

### Phase 10 — Production Activity 전환 (iter 34)
- MultiTurnDemoActivity (7/7 ✓)
- IntegrationDemoActivity (13/13 ✓ — 4채널 라우팅 정상)
- HomeStateDemoActivity (6/6 ✓ — HVAC 재해석 포함)
- InteractiveTestActivity (대화형 ✓)
- auto-fallback 패턴: NB 파일 누락 시 ONNX 모드

## 핵심 학습

1. **데이터를 코드보다 먼저 의심하라** — inputmeta lid/shape, 토크나이저 미스매치 모두 데이터 문제
2. **양자화 민감도는 모델별로 다르다** — v46(generalization)은 uint8 부족, int16 필수
3. **Ensemble이 단일보다 항상 낫진 않다** — 약한 모델이 평균을 끌어내림
4. **정밀도 손실은 후처리 룰로 부분 회복 가능** — PostRulesV4 +5~7%p 일관 향상
5. **JNI native가 Kotlin loop보다 훨씬 빠르다** — 24576 iter int16 quant: Kotlin 6ms → C 1ms

## 변환된 NB 모델

| 파일 | 크기 | 용도 |
|---|---|---|
| `cnn_body_v46_int16.nb` | 2.7MB | **production** (NPU) |
| cnn_body_v46.nb (uint8) | 1.35MB | 비교용 (정확도 부족) |
| cnn_body_v28.nb (uint8) | 1.36MB | 비교용 (ensemble 시도) |

## 최종 production 구성

```
사용자 발화
  ↓ Preprocess (296 STT 매핑, Kotlin)
  ↓ WordPieceTokenizer → token_ids[32]
  ↓ AwNluJni.nativeRunWithLookup(npuPtr, ids, 32768f)
  │   ├─ g_emb_table[token_id × 768] 읽기 (C 직접)
  │   ├─ int16 quantize (round × 32768)
  │   └─ awnn_run (NPU 1.5ms)
  ↓ 5-head logits (44 floats)
  ↓ PostRulesV4 (Kotlin, 64개 룰)
  → Result(fn / exec / dir / param / judge)
총: ~9.4ms
```

## 미완 / 추후 작업

1. **VoiceAiService 본 서비스 NPU 적용** — 현재 KoElectra 단일헤드 IntentClassifier 사용 중. IntentClassifierV46으로 교체 필요 (큰 변경, 신중)
2. **ARM NEON SIMD quantize** — 9ms → 5ms 가능 (선택적)
3. **Activity onDestroy 통일** — release() 호출 보장
4. **NPU 모드에서도 데이터 증강 학습 재시도** — 르엘 219 combo 60% → 70%+ 목표

## Production-readiness 점검

| 항목 | 상태 |
|---|---|
| NB 빌드 재현 가능? | ✅ docs/T527_NB_CONVERSION_SUCCESS.md |
| 정확도 검증 인프라? | ✅ NpuEvalActivity + golden 99/491/219 |
| ONNX fallback? | ✅ NB 파일 없으면 자동 ONNX |
| 4채널 라우팅 동작? | ✅ IntegrationDemo 13/13 |
| 멀티턴 컨텍스트? | ✅ MultiTurnDemo 7/7 + HomeState 6/6 |
| latency 안정? | ✅ warmup 후 ~10ms |
| 메모리 누수? | ⚠️ Activity onDestroy 보강 필요 |

## 참고 문서 (생성된 docs)

- `T527_NB_CONVERSION_SUCCESS.md` — NB 빌드 절차
- `T527_NPU_INFERENCE_VERIFIED.md` — NPU 추론 0.95ms 검증
- `T527_NPU_ENSEMBLE.md` — 두 NB 평균 시도
- `T527_NPU_ACCURACY_REGRESSION.md` — 양자화 손실 발견
- `T527_NPU_QUANT_EXPERIMENTS.md` — 양자화 매트릭스
- `T527_NPU_INT16_BREAKTHROUGH.md` — int16 돌파
- `T527_NPU_FULL_BENCHMARK.md` — 99/491/219 일반화
- `T527_NPU_CLASSIFIER_INTEGRATED.md` — IntentClassifier NPU 모드
- `T527_NPU_LOOKUP_MODE.md` — Kotlin lookup
- `T527_NPU_NATIVE_LOOKUP.md` — JNI native lookup (9.4ms)
- `T527_NPU_PRODUCTION_INTEGRATED.md` — 4 Activity 전환
- `T527_NPU_FULL_JOURNEY.md` — 본 보고서
