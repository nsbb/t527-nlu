# T527 NPU NLU 추론 검증 완료 (2026-05-07)

## 결론

**NB가 T527 NPU에서 실제로 돌아간다.** 평균 0.95ms.

## 검증 환경

- 디바이스: T527 데브킷 `51475789d0c64881cd3`
- NPU 드라이버: VIPlite v1.13 (`/vendor/lib64/libVIPlite.so`)
- NB: `cnn_body_v46.nb` 1.35MB (uint8 asymmetric_affine)
- 입력: dummy 24576 bytes (모두 zero_point 155)

## 결과 (logcat)

```
NpuNbTest: NB 경로: /data/local/tmp/cnn_body_v46.nb (1356608 bytes)
NpuNbTest: ✅ NPU init + NB load OK (ptr=...)
AWNN: awnn_run total: 0.59 ms
AWNN: awnn_run total: 0.60 ms
... (20 iter)
NpuNbTest: === Latency (20 iter, dummy) ===
NpuNbTest: 평균: 0.95 ms
NpuNbTest: min : 0.76 ms
NpuNbTest: max : 1.70 ms
NpuNbTest: fn argmax: 19, exec argmax: 3, dir argmax: 0
NpuNbTest: ✅ NPU release OK
```

## ONNX CPU vs NPU 비교

| 모드 | 평균 latency | 비고 |
|---|---|---|
| ONNX Runtime CPU (v46 full) | 21~120ms | 현재 t527_smart_v2 사용 중 |
| **NPU NB (cnn_body_v46)** | **0.95ms** | 본 검증 |

20~120배 가속. 단, NPU는 CNN body만 처리 — KoELECTRA embedding(token lookup + projection) 부분은 별도 처리 필요.

## 추가된 컴포넌트

### JNI 모듈 (C)
- `jni/nlu/awnlusdk.c` — awnn_lib 기반 NB 로드 + 추론 래퍼
- `jni/Android.mk`: `awnlu` 모듈 추가

### Kotlin
- `nlu/AwNluJni.kt` — JNI binding (System.loadLibrary("awnlu"))
- `NpuNbTestActivity.kt` — 더미 입력 검증 Activity

### NB 출력 → Java 변환
awnn은 dequantize된 float 배열로 5-head 출력 반환:
- fn[20], exec[5], dir[9], param[5], judge[5]
- 합계 44 floats를 단일 FloatArray로 merge하여 반환

## 미완 — 다음 단계

1. **embedding ONNX 분리**: 현재 `cnn_body_v46.onnx` 입력은 `[1, 32, 768]` float32 embedded이지만,
   t527_smart_v2의 IntentClassifierV46은 token_ids 입력만 지원.
   → KoELECTRA embedding-only ONNX 분리 또는 JNI에서 lookup table 사용
2. **uint8 quantize**: float32 embedded → scale 0.003370, zp 155 적용
3. **IntentClassifierV46 NPU 모드**: ONNX CPU 결과와 NPU 결과 정확도 비교
4. **end-to-end latency**: 토큰화 + embedding + NPU + 후처리 합계

## 실행

```bash
adb -s 51475789d0c64881cd3 push checkpoints/cnn_body_acuity/wksp_nbg_nbg_unify/network_binary.nb \
    /data/local/tmp/cnn_body_v46.nb
adb -s 51475789d0c64881cd3 shell am start \
    -n com.t527.smart_v2/com.t527.smart_service.NpuNbTestActivity
adb -s 51475789d0c64881cd3 logcat -s NpuNbTest:I AWNN:D
```
