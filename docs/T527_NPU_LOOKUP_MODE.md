# T527 NPU + Lookup Table 모드 (2026-05-07)

## 결론

`token_emb_v46.bin` 직접 lookup table로 **ONNX 27.3ms → NPU+lookup 15.4ms (1.8x)**, 10/10 정확도 일치.

## 진화

| 모드 | latency | breakdown |
|---|---|---|
| ONNX CPU full ensemble | 27.3 ms | 모든 단계 ONNX |
| NPU + ONNX token_emb | 17.1 ms | ONNX emb(~14ms) + Kotlin quant(~1ms) + NPU(~1.5ms) + rules |
| **NPU + lookup table** | **15.4 ms** | lookup(~12ms) + Kotlin quant(~1ms) + NPU(~1.5ms) + rules |

## 남은 병목 분석

`classifyNpu(lookup mode)` 내부 추정:
- token_id → lookup row 복사 (32 × 768 floats): ~5ms
- float→int16 quantize (24576 iter): ~7ms
- ByteBuffer putShort 반복: ~5ms
- AwNluJni.nativeRunUint8 (NB 추론): ~1.5ms
- 합계: ~18ms (실측 15.4ms와 근사)

**Kotlin 루프 24576 iter가 주범.** JNI native로 lookup+quantize 합치면 ~3ms 가능.

## 다음 단계 — JNI native lookup+quantize

```c
// 추가할 JNI 함수
jfloatArray nativeRunWithLookup(
    JNIEnv*, jobject, jlong nativePtr,
    jfloatArray embTable,    // [32000 * 768] flat
    jintArray tokenIds,      // [32]
    jfloat scale             // 32768
)
```

C에서:
1. tokenIds[i] × 768 offset으로 embTable에서 row 읽기
2. round(value * scale).clip().putShort
3. nativeRunUint8 호출

→ 예상 latency: 3~5ms (lookup 1ms + quant native 0.5ms + NPU 1.5ms + JNI overhead 1ms)
→ ONNX 27ms 대비 **5~9x 가속** 가능

## API

```kotlin
classifier.initNpuLookup(
    embBinPath = "$dir/token_emb_v46.bin",   // 93.75 MB raw float32
    npuNbPath = "$dir/cnn_body_v46_int16.nb",
    vocabPath, labelMapPath, preprocessPath
)
```

### 디바이스 파일

| 파일 | 위치 | 크기 | 용도 |
|---|---|---|---|
| `cnn_body_v46_int16.nb` | `/data/local/tmp/` | 2.7MB | NPU NB |
| `token_emb_v46.bin` | `/data/data/com.t527.smart_v2/files/` | 93.75MB | lookup table |

## 정확도 (재확인)

10 시나리오 × 3 모드 모두 **10/10 결과 일치** (ONNX vs NPU+ONNX vs NPU+lookup).
PostRulesV4가 양자화 잔여 차이를 보정해줘서 차이 없음.
