# Source Note — v72 cnn_body NB 변환 + 디바이스 검증 (2026-05-15)

## Source

- 디바이스: T527 데브킷 `51475789d0c64881cd3`
- 모델: `checkpoints/cnn_multihead_v72.pt` (epoch 30, combo 88.43, ke_fn 97.33, "v72 — indirect_v2 3794개 추가")

## 변환 절차

```
cnn_multihead_v72.pt
  → scripts/export_v72_standalone_onnx.py
  → checkpoints/nlu_v72_generalization.onnx (42 ops)
  → onnx.utils.Extractor (token_emb gather output 부터)
  → checkpoints/cnn_body_v72.onnx (31 nodes)
  → onnxsim
  → checkpoints/cnn_body_v72_sim.onnx (5.9MiB)
  → shape fix [1, 32, 768]
  → Acuity import (Error 0)
  → Acuity quantize int16 dynamic_fixed_point fl=15 (Error 0)
  → Acuity export NB VIP9000NANOSI_PLUS_PID0X10000016 (Error 0)
  → checkpoints/cnn_body_v72_acuity/wksp_int16_nbg_unify/network_binary.nb (2,744,448 bytes)
```

## 디바이스 검증 (10 시나리오, NpuClassifierTestActivity)

```
Classifier: IntentClassifierV46 initNpuNativeLookup
  npuNbPath = /data/local/tmp/cnn_body_v72_int16.nb (2.74 MB)
  embBinPath = /data/data/com.t527.smart_v2/files/token_emb_v46.bin (94 MB)
  (token_emb는 v28/v46/v71/v72 모두 ko-sbert frozen 동일 → v46 bin 재사용)

결과:
- ONNX CPU avg: 26.5 ms
- NPU v72 NB avg: 9.3 ms (2.8x speedup)
- ONNX↔NPU agreement: 10/10
- 발화 모두 정답 (raw, post-rules 없이)
```

## 큰 셋 평가 — ART JIT crash 발생

`NpuClassifierFullEvalActivity` 로 99 골든셋 batch 평가 시도 시 ART JIT thread SIGSEGV 재발. 이전 v46 NB로 491에서 봤던 crash와 동일 패턴.

```
F libc: Fatal signal 11 (SIGSEGV) in tid Jit thread pool
backtrace: art::jit::JitCodeCache::Reserve / GarbageCollectCache
```

- v46 NB로는 99셋 OK였는데 v72에서 처음부터 crash
- 추정: v72 NB가 메모리 약간 더 사용 (Acuity export 시 추가 layer/buffer)
- 또는 embedding 94MB + JIT cache 누적 압박

## 해결 방향 (미완)

1. batch 분할 (50개씩 + System.gc())
2. embedding lookup table을 native mmap으로 (Kotlin heap 회피)
3. ONNX session 강제 release 후 NPU 평가만

## Linked wiki

- `wiki/models/version-matrix.md`
- `wiki/projects/t527-npu-integration.md`
- `wiki/issues/server-device-version-gap.md`

## 추후 액션

- v72 NB 정량 정확도 측정 (ART crash 해결 후 99/491/219)
- v46 NB 대비 -2.8%p 차이 (이전 측정) 회복 여부 검증
- 정확도 검증 후 production Activity 5개 (Multi/Integration/HomeState/Interactive/등) NB path 일괄 v72로 교체
