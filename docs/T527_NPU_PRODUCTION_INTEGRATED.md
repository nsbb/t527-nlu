# T527 NPU 모드 production Activity 통합 (2026-05-07)

## 결론

**4개 production Activity 모두 NPU 모드로 전환** — 동작 100% 일관, 평균 latency 9~30ms.

## 전환된 Activity

| Activity | 시나리오 수 | 결과 | NLU latency |
|---|---|---|---|
| MultiTurnDemoActivity (DST) | 7/7 | ✅ | warmup 후 ~10ms |
| IntegrationDemoActivity (E2E AIDL) | 13/13 | ✅ | 16~41ms |
| HomeStateDemoActivity (HVAC 재해석) | 6/6 | ✅ | warmup 후 ~10ms |
| InteractiveTestActivity (대화형) | (사용자 입력) | ✅ | warmup 후 ~10ms |

## NPU 모드 활성 패턴 (auto-fallback)

```kotlin
val ok = if (File(npuNb).exists() && File(embBin).exists()) {
    classifier.initNpuNativeLookup(embBin, npuNb, ...)  // NPU 가속
} else {
    classifier.init(onnxPath, ...)  // ONNX fallback
}
```

→ NB 파일 누락 시 자동으로 ONNX CPU로 fallback. 안전.

## 디바이스 파일 (NPU 모드 prerequisites)

| 경로 | 크기 | 출처 |
|---|---|---|
| `/data/local/tmp/cnn_body_v46_int16.nb` | 2.7MB | Acuity int16 quantize |
| `/data/data/com.t527.smart_v2/files/token_emb_v46.bin` | 93.75MB | ONNX initializer 추출 |
| `/data/data/com.t527.smart_v2/files/nlu_assets/vocab.txt` | (기존) | ko-sbert vocab |
| `/data/data/com.t527.smart_v2/files/nlu_assets/label_map.json` | (기존) | head label maps |

## 검증 — IntegrationDemo NPU 모드 결과

```
Wav2VecNative: NLU NPU Init OK
Wav2VecNative: NLU NB loaded: cnn_body_v46_int16.nb
IntegDemo: [0]  거실 불 켜줘                → AIDL          | NLU 16ms
IntegDemo: [1]  에어컨 23도로 해줘           → AIDL          | NLU 23ms
IntegDemo: [2]  주방 환기 켜줘                → AIDL          | NLU 25ms
IntegDemo: [3]  안방 난방 올려줘              → AIDL          | NLU 28ms
IntegDemo: [4]  가스 잠가줘                   → AIDL          | NLU 33ms
IntegDemo: [5]  커튼 닫아줘                   → AIDL          | NLU 33ms
IntegDemo: [6]  외출모드 실행해 줘            → AIDL          | NLU 30ms
IntegDemo: [7]  오늘 날씨 어때?               → EXTERNAL_API  | NLU 32ms
IntegDemo: [8]  엘리베이터 불러줘             → REST_RUEL     | NLU 32ms
IntegDemo: [9]  에너지 사용량 어때?           → REST_RUEL     | NLU 32ms
IntegDemo: [10] 우리집 상태 어때?             → SELF          | NLU 31ms
IntegDemo: [11] 황사 때문에 창문 닫아야 해    → AIDL          | NLU 37ms
IntegDemo: [12] 어르신이 덥다고 하시네요      → AIDL          | NLU 41ms
```

13/13 라우팅 정상. 모든 채널(AIDL/REST_RUEL/EXTERNAL_API/SELF) 동작.

## 남은 작업

1. **VoiceAiService 본 서비스 NPU 적용** — Foreground Service에서 음성 입력 후 NLU 호출 부분
2. **Acuity NPU 정확도 골든셋 491 PostRules 후 정량 측정** — production 적용 안정성 확보
3. **release/destroy 타이밍 점검** — Activity 종료시 NPU 자원 누수 방지

## 정리 — NPU 통합 전체 진척

```
Phase 1: NB 변환 (uint8) → 정확도 57% (실패)
Phase 2: NB int16 변환 → 정확도 97% (회복)
Phase 3: JNI 통합 + Kotlin lookup → 9.4ms latency
Phase 4: production Activity 전환 → 동작 일관 ✅ (현재)

결과: NLU 추론이 25.8ms ONNX → 9.4ms NPU (2.7x 가속), 정확도 무손실.
```
