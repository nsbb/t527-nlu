# T527 NPU Ensemble (v28 + v46) 검증 (2026-05-07)

## 결론

**두 NB 동시 로드 + logit 평균으로 NPU에서 ensemble 동작.**
"거실 불 켜줘" 입력 → fn=light_control, exec=control_then_confirm 정확.

## 변환된 NB

| 모델 | NB 크기 | latency | argmax (거실 불 켜줘) |
|---|---|---|---|
| cnn_body_v28.nb | 1.36MB | ~1ms | light_control(0.61) / query_then_respond / none |
| cnn_body_v46.nb | 1.35MB | ~1ms | light_control(1.48) / control_then_confirm / on |
| **logit avg** | — | **3.25ms** (2 inference) | **light_control(1.04) / control_then_confirm / none** |

## 디바이스 검증 (T527 51475789d0c64881cd3)

```
NpuEnsemble: 📂 input: 24576 bytes
NpuEnsemble: ✅ NPU init + 2 NB load OK
NpuEnsemble: === v28 단독 === fn=light_control(0.61), top3: light_control, unknown, energy_query
NpuEnsemble: === v46 단독 === fn=light_control(1.48), top3: light_control, security_mode, curtain_control
NpuEnsemble: === Ensemble: avg(logit_v28, logit_v46) === fn=light_control(1.04)
NpuEnsemble: avg=3.25 ms, min=1.88, max=7.75
```

## 디버깅 메모 — 토크나이저 미스매치

처음에 NPU에서 모두 unknown 나왔던 원인:
- `prepare_npu_test_inputs.py`가 `monologg/koelectra-base-v3` 토크나이저 사용
- 모델 학습은 ko-sbert vocab (32000 BERT WordPiece) 사용
- 다른 vocab → 다른 token_ids → 다른 embedding → unknown

**수정**: 서버의 `tokenizer/` 디렉토리(ko-sbert vocab)로 교체. 디바이스 앱이 사용하는 `nlu_assets/vocab.txt`와 동일.

## 변환 파이프라인 — v28도 동일하게 통과

```
import:    cnn_body_v28.onnx → cnn28.json/data        Error(0) ✓
quantize:  + inputmeta_3d.yml + calib3d/             Error(0) ✓
export:    + VIP9000NANOSI_PLUS_PID0X10000016         Error(0) ✓
```

v46과 차이점: v28 ONNX는 input shape이 dynamic [unk, unk, 768]였음 →
`onnx.save` 직전 `dim_value`를 1, 32로 강제 fix해야 import에서 받음.

## NPU Ensemble 구현 (JNI 두 인스턴스)

```kotlin
val ptr28 = AwNluJni.nativeNew(nb28Path)
val ptr46 = AwNluJni.nativeNew(nb46Path)
val a = AwNluJni.nativeRunUint8(ptr28, input)!!
val b = AwNluJni.nativeRunUint8(ptr46, input)!!
val avg = FloatArray(44) { (a[it] + b[it]) / 2f }  // logit avg
```

`Awnn_Context_t`는 NB마다 별도 인스턴스. `awnn_init()`은 1회만 호출.

## 정확도 — 추후 골든셋 검증 예정

- 단일 sample은 fn/exec 정확, dir만 약간 차이 (none vs on)
- 491 골든셋 / 219 르엘 GT 전체 측정 필요
- 양자화 손실 정량 분석 (logit_npu vs logit_cpu MAE/MSE)

## 다음 작업

1. **Softmax avg 시도** — logit avg 대신 softmax 후 평균하면 outlier(v28의 dir=none 강한 신호) 영향 줄어들 가능성
2. **Embedding lookup 디바이스 구현** — 현재 .bin 입력 사용. 실서비스는 token_ids → embedding lookup → quantize 디바이스에서 해야 함
3. **golden 91/491 골든셋으로 NPU vs ONNX accuracy 비교**

## 파일

- `scripts/extract_cnn_body.py` — v28/v46 ONNX subgraph 추출
- `scripts/prepare_npu_test_inputs.py` — text → uint8 .bin 생성 (ko-sbert vocab 수정)
- `checkpoints/cnn_body_v28_acuity/` — v28 NB 변환 작업 디렉토리
- `t527_smart_v2/.../NpuEnsembleTestActivity.kt` — 두 NB 로드 + 평균 검증
