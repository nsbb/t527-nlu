# T527 IntentClassifierV46 NPU 모드 통합 (2026-05-07)

## 결론

**ONNX CPU 28ms → NPU 모드 17ms (1.6x 가속), 10/10 결과 일치.**

## 통합 파이프라인

```
사용자 발화
  ↓ Preprocess (296 STT 매핑, Kotlin)
  ↓ WordPieceTokenizer (Kotlin, maxLen=32)
  ↓ token_emb_v46.onnx (ONNX CPU, 94MB embedding lookup)  ← ~14ms 병목
  ↓ float32 → int16 quantize (Kotlin: round(value * 32768))  ← ~1ms
  ↓ cnn_body_v46_int16.nb (T527 NPU)                        ← ~1.5ms
  ↓ 5-head logits (auto-dequantized by awnn)
  ↓ PostRulesV4 (Kotlin, 64개 룰)
  → Result(fn, exec, dir, param, judge)
```

## 검증 결과 (10개 시나리오)

| 발화 | NPU 결과 | ONNX 결과 | 일치 |
|---|---|---|---|
| 거실 불 켜줘 | light_control/control_then_confirm/on | (동일) | ✓ |
| 에어컨 켜줘 | ac_control/control_then_confirm/on | (동일) | ✓ |
| 주방 가스 잠가줘 | gas_control/control_then_confirm/close | (동일) | ✓ |
| 환기 켜줘 | vent_control/control_then_confirm/on | (동일) | ✓ |
| 난방 올려줘 | heat_control/control_then_confirm/up | (동일) | ✓ |
| 안방 에어컨 19도로 맞춰줘 | ac_control/control_then_confirm/set | (동일) | ✓ |
| 거실 커튼 닫아줘 | curtain_control/control_then_confirm/close | (동일) | ✓ |
| 엘리베이터 좀 불러줘 | elevator_call/control_then_confirm/on | (동일) | ✓ |
| 지금 집 상태 어때? | home_info/query_then_respond/none | (동일) | ✓ |
| 현관문 열어줘 | door_control/control_then_confirm/open | (동일) | ✓ |

**10/10 일치. PostRulesV4가 양자화 잔여 차이 보정.**

## Latency 분석

- NPU 17.1ms 평균 = token_emb ONNX(~14ms) + int16 quant(~1ms) + NPU(~1.5ms) + PostRules(~0.5ms)
- ONNX 28.1ms 평균 = full ensemble 추론

**병목: token_emb ONNX subgraph (94MB).** 단순 lookup인데 ONNX Runtime overhead가 큼.

## 다음 단계 — token_emb 직접 lookup으로 진짜 NPU 성능 (1.5ms) 회복

token_emb는 단순 `embedding_table[token_id]` 룩업:
- 32 tokens × 768 dim 복사
- 현재: ONNX Runtime로 ~14ms
- 직접 Kotlin 루프 + raw float array 메모리 접근: 추정 ~1ms
- 또는 JNI로 native lookup: ~0.5ms

→ **token_emb_v46.bin (raw 32000×768 float32) lookup table 추출 + Kotlin 직접 lookup**으로 변경하면
**전체 latency 17ms → ~3ms** 가능 (10x 가속).

## API 변경

### 기존 ONNX 모드 (호환 유지)
```kotlin
classifier.init(onnxPath, vocabPath, labelMapPath, preprocessPath)
```

### NPU 모드 (신규)
```kotlin
classifier.initNpu(
    embOnnxPath = "$dir/token_emb_v46.onnx",
    npuNbPath = "/data/local/tmp/cnn_body_v46_int16.nb",
    vocabPath, labelMapPath, preprocessPath
)
classifier.classify("거실 불 켜줘")  // 동일 인터페이스
```

## 다음 작업

1. **MultiTurnDemoActivity / IntegrationDemoActivity NPU 모드 전환**
2. **token_emb 직접 lookup 구현** (latency 17ms → 3ms 목표)
3. **VoiceAiService 본 서비스 NPU 통합**
