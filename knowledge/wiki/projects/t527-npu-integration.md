# t527 NPU NLU 통합

## Current state

T527 NPU에서 CNN body 추론. NB 파일은 `cnn_body_v46_int16.nb` (2.7MB, int16 fixed point fl=15). 디바이스 t527_smart_v2 앱에 `initNpuNativeLookup` API 통합 완료.

**중요: NPU에 올라가 있는 건 v46 단독 cnn_body. 서버 production v72 ensemble과 다름.**

### 디바이스 파일

| 경로 | 크기 | 용도 |
|---|---|---|
| `/data/local/tmp/cnn_body_v46.nb` | 1.36MB | uint8 (정확도 약함 — 폐기 권장) |
| `/data/local/tmp/cnn_body_v46_int16.nb` | 2.74MB | int16 v46 (이전 default, fn 97%) |
| **`/data/local/tmp/cnn_body_v72_int16.nb`** | **2.74MB** | **int16 v72 (2026-05-15 추가, 10/10 ONNX 일치)** |
| `/data/local/tmp/cnn_body_v28.nb` | 1.36MB | uint8 (ensemble 시도용 — 폐기) |
| `/data/data/com.t527.smart_v2/files/token_emb_v46.bin` | 93.75MB | embedding lookup table (v28/v46/v71/v72 모두 동일 ko-sbert frozen — 재사용) |

### 파이프라인

```
사용자 발화
  → WordPieceTokenizer (Kotlin)
  → token_emb lookup (JNI native, g_emb_table)
  → int16 quantize (round * 32768)
  → cnn_body_v46_int16.nb (NPU 1.5ms)
  → 5-head logits dequantized
  → PostRulesV4 (Kotlin)
  → Result (fn / exec / dir / param / judge)
총: ~9.4ms (warmup 후) / ONNX CPU 대비 2.7x 가속
```

## Known-good settings

- 양자화: int16 + dynamic_fixed_point + fl=15 + kl_divergence + 300 diverse calib
- 입력 shape `[1, 32, 768]` float32 → int16 변환 후 49152 bytes 전송
- JNI: `nativeLoadEmbTable(path, vocab, dim)` 한 번 + `nativeRunWithLookup(ptr, ids, 32768f)` 매 추론
- uint8은 v46 mixup 모델의 dynamic range 표현 부족 — 사용 금지
- ensemble 시도 결과 v28 단독이 평균을 끌어내림 → v46 int16 단독 + PostRules가 최강

## Open issues

- ✅ v72 cnn_body NB 변환 완료 (2026-05-15, 10/10 ONNX 일치 검증)
- v72 NB 정량 정확도 측정 미완 — golden 99/491/219 batch 평가 시 ART JIT crash 재발 (이전 v46 NB 491에서 본 것과 동일 패턴)
- ART crash 해결책 후보: batch 분할 + GC, embedding mmap, ONNX session 해제
- 정확도 측정 결과:
  - Golden 99: fn 100% / combo 89.9%
  - Golden 491: fn 98.6% / combo 92.7% (ONNX 95.5% 대비 -2.8%p)
  - 르엘 219: fn 82.2% / combo 60.7% (ONNX 60.3% 대비 +0.4%p)
- 491 셋 batch 추론에서 ART JIT thread SIGSEGV crash 발생 (메모리 압박 추정, 미해결)
- ARM NEON SIMD quantize 적용 시 5ms 미만 가능 (미적용)

## Related sources

- `raw/source-notes/src-npu-int16-breakthrough-20260507.md`
- 변환 절차: `checkpoints/cnn_body_acuity/inputmeta_3d.yml` + Acuity 6.12 + Docker
- 양자화 파라미터: `wksp_int16_nbg_unify/nbg_meta.json` (fl=15)
- JNI: `t527_smart_v2/app/src/main/jni/nlu/awnlusdk.c`
- Kotlin: `t527_smart_v2/.../nlu/IntentClassifierV46.kt` (initNpuNativeLookup)

## Last updated

- `2026-05-13`
