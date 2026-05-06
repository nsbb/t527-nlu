# T527 디바이스 NLU 벤치마크 결과

작성: 2026-05-07

## 환경

- **디바이스**: T527 데브킷 (51475789d0c64881cd3, ARM64)
- **모델**: nlu_v28_v46_ensemble.onnx (105MB, KoELECTRA + CNN 5-head)
- **런타임**: ONNX Runtime Android 1.17.1 (CPU)
- **빌드**: t527_smart_v2 / NluBenchmarkActivity (Kotlin)
- **모델 위치**: /data/data/com.t527.smart_v2/files/ (run-as 복사)

## 결과

| 지표 | 1차 (raw 모델) | 2차 (+ PostRules 25개) |
|------|---------|---------|
| 정확도 | **10/15 (66%)** | **15/15 (100%)** |
| 단일 추론 평균 | 17ms | 20ms |
| 단일 추론 max | 22ms | 30ms |
| 100회 반복 평균 | 17ms | 20ms |
| 초기화 시간 | 952ms | 1017ms |

## 의미

- **모델 자체로는 66%** (서버 raw와 일치, 르엘 패턴은 잘 잡지만 비유/완곡/조사 등 약함)
- **132개 후처리 규칙 중 25개만 포팅해도 100%** (벤치마크 범위 한정)
- **추론 속도**: 서버 CPU 0.64ms vs T527 20ms ≈ 31배 차이 — 임베디드 ARM CPU 정상 범위
- **실시간 응답**: STT(200ms) + NLU(20ms) + AIDL(100ms) ≈ 320ms 응답 가능

## 다음 작업

1. **PostRules 132개 전체 포팅** (현재 25개 → 132개)
2. **NB 변환 시도** (CNN body만 NPU, 임베딩은 CPU lookup)
   - 예상: 20ms → 3ms (학계 Pure CNN 4L 사례 1ms)
3. **t527_smart_v2 본 서비스에 IntentClassifierV46 통합** (STT → NLU → AIDL chain)
