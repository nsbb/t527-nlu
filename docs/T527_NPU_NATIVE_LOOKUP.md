# T527 NPU Native Lookup — 9.4ms / 2.7x 가속 (2026-05-07)

## 결론

**JNI에서 lookup + int16 quantize + NPU run을 한 번에 처리 → ONNX 25.8ms → 9.4ms (2.7x)**, 정확도 10/10 유지.

## 진화

| 모드 | latency | speedup |
|---|---|---|
| ONNX CPU full | 25.8 ms | 1.0x |
| NPU + ONNX token_emb | 17.1 ms | 1.5x |
| NPU + Kotlin lookup | 15.4 ms | 1.7x |
| **NPU + Native lookup** | **9.4 ms** | **2.7x** ⭐ |

## 추가된 JNI 함수

```c
// awnlusdk.c
static float *g_emb_table = NULL;  // 캐시된 32000×768 lookup table

JNI nativeLoadEmbTable(path, vocab, dim) → bool
JNI nativeFreeEmbTable()
JNI nativeRunWithLookup(ptr, tokenIds[32], scale) → float[44]
```

`nativeRunWithLookup` 내부:
1. tokenIds × 768 offset으로 g_emb_table에서 row 직접 읽기
2. round(value × scale) clip → int16 quantize
3. awnn_set_input_buffers + awnn_run
4. 5-head dequantized logits 반환

## API (Kotlin)

```kotlin
classifier.initNpuNativeLookup(
    embBinPath = "$dir/token_emb_v46.bin",
    npuNbPath  = "/data/local/tmp/cnn_body_v46_int16.nb",
    vocabPath, labelMapPath, preprocessPath
)
val r = classifier.classify("거실 불 켜줘")
// → light_control / control_then_confirm / on  (9ms)
```

## Latency breakdown (9.4ms 추정)

- Preprocess + Tokenize (Kotlin): ~1.5ms
- JNI nativeRunWithLookup: ~6ms
  - 768 × 32 = 24576 lookup + quantize (C scalar): ~3ms
  - awnn_set_input + awnn_run: ~1.5ms
  - dequant 5-head copy: ~1ms
  - JNI marshal overhead: ~0.5ms
- PostRulesV4: ~1ms
- Other (Result alloc): ~0.5ms

추가 최적화 가능: ARM NEON SIMD로 quantize → ~1ms 절감 가능. 그러나 9ms도 production 충분.

## 프로덕션 권장 구성 (final)

```
사용자 발화
  ↓ Preprocess (Kotlin)
  ↓ WordPieceTokenizer → token_ids[32]
  ↓ AwNluJni.nativeRunWithLookup(npuPtr, ids, 32768f)  ← 6ms (lookup + int16 quant + NPU)
  ↓ 5-head logits (44 floats)
  ↓ PostRulesV4 (Kotlin, 64 룰)
  → Result (fn / exec / dir / param / judge)
총: ~9.4ms
```

## 검증

10 시나리오 모두 ONNX CPU와 결과 동일:
```
✓ '거실 불 켜줘'         → light_control/control_then_confirm/on (9ms)
✓ '에어컨 켜줘'          → ac_control/control_then_confirm/on (9ms)
✓ '주방 가스 잠가줘'      → gas_control/control_then_confirm/close (10ms)
✓ '환기 켜줘'            → vent_control/control_then_confirm/on (9ms)
✓ '난방 올려줘'          → heat_control/control_then_confirm/up (10ms)
✓ '안방 에어컨 19도로'   → ac_control/control_then_confirm/set (9ms)
✓ '거실 커튼 닫아줘'      → curtain_control/control_then_confirm/close (10ms)
✓ '엘리베이터 좀 불러줘' → elevator_call/control_then_confirm/on (10ms)
✓ '지금 집 상태 어때?'   → home_info/query_then_respond/none (9ms)
✓ '현관문 열어줘'        → door_control/control_then_confirm/open (9ms)
```

## 다음 단계

1. **MultiTurnDemoActivity / IntegrationDemoActivity NPU 모드 전환**
2. **VoiceAiService 본 서비스 NPU 통합**
3. (선택) ARM NEON SIMD quantize → 5ms 미만
