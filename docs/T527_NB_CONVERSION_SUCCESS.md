# T527 NLU NB 변환 성공 (2026-05-07)

## 결론

**변환 성공.** 이전 5번 실패 원인은 **inputmeta lid 미스매치**였음.

## 증거

```
$ ls -la checkpoints/cnn_body_acuity/wksp_nbg_nbg_unify/
network_binary.nb     1,356,608 bytes  (1.35 MB)
nbg_meta.json         3,251 bytes
```

```
$ adb shell ls -la /data/local/tmp/cnn_body_v46.nb
-rw-rw-rw- shell shell 1356608 2026-05-07 09:25
```

## 핵심 수정

이전 `inputmeta.yml`:
```yaml
- lid: embedded_38
  shape: [1, 1, 32, 768]   # 4D (잘못)
  layout: nchw
```

import된 `cnn3d.json` 확인 결과 모델 입력은:
- **lid**: `embedded_38`
- **shape**: `[1, 32, 768]` (3D — Conv1D 기반 CNN)

수정 후 `inputmeta_3d.yml`:
```yaml
- lid: embedded_38
  shape: [1, 32, 768]      # 3D (정답)
  layout: nhc
```

calib npy도 4D `(1,1,32,768)` → 3D `(1,32,768)` 으로 squeeze 후 `calib3d/`에 저장.

## 파이프라인 전체 통과

```
import:    Error(0) Warning(1)  ✓
quantize:  Error(0) Warning(0)  ✓ (uint8 asymmetric_affine, kl_divergence)
export:    Error(0) Warning(0)  ✓ (VIP9000NANOSI_PLUS_PID0X10000016)
```

## 모델 구조 (CNN Multi-Head)

- 입력: `embedded` [1, 32, 768] float32 (KoELECTRA embedding 결과)
- 출력 5-head: fn[20], exec[5], dir[9], param[5], judge[5]
- Ops: Conv 4, Gemm 10, MatMul 1, Relu 9, ReduceMean 1
- Transformer 없음 — 순수 CNN body

## 양자화 파라미터

- **입력 embedded**: scale=0.003370, zero_point=155 (uint8)
- **fn 출력**: scale=0.0987, zero_point=172 (uint8, [-16.93, 8.24])
- **exec 출력**: scale=0.0383, zero_point=144 (uint8, [-5.51, 4.26])

## 실패한 이전 시도들 (과거 기록 정정)

이전 docs `T527_NB_CONVERSION_FINAL.md`에 기록한 5번 실패는 **모두 같은 원인** (inputmeta 4D vs 모델 3D + lid 추정 오류). 1번 정확히 짚으니 quantize/export 모두 통과.

## 다음 단계 — JNI 통합 (미완)

NB 파일은 만들어졌지만 t527_smart_v2 앱은 아직 ONNX CPU로 돌고 있음. NPU 추론을 실제 구동하려면:

1. **awnn_lib JNI 래퍼 추가** (`awaiasr_2`/`android_stt_bundle_app` 참조)
2. KoELECTRA embedding 부분은 ONNX CPU로 돌리고
3. CNN body만 NB로 추론 (`awnn_run`)
4. 5-head logits 후처리는 기존 PostRules 그대로

다음 iteration에서 JNI 통합 진행 예정.
