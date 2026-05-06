# T527 NPU NB 변환 진행 로그 (cnn_body_v46)

## 진전

✅ **단계 1: CNN body 추출 성공**
- v46 PyTorch checkpoint → CNN body only ONNX
- 105MB (전체) → 5.91MB (CNN body, 임베딩 분리)
- Input: `[1, 32, 768]` pre-embedded vector
- Output: 5 logits (fn/exec/dir/type/judge)
- 31 nodes, 30 initializers

✅ **단계 2: Acuity import 성공**
- pegasus import onnx → cnn_body_v46.json + cnn_body_v46.data (6MB)
- Error 0, Warning 1
- 모든 layer 정상 변환 (Conv 4개, Gemm 10개, ReLU/Add/Mean)

🟡 **단계 3: Quantize — 진행 중 (inputmeta 호환성 문제)**

시도한 것들:
1. NPY type → "Cannot load file containing pickled data when allow_pickle=False"
2. BIN type → "Unsupport database type"
3. H5FS type → "TypeError: expected bytes, NoneType found"
4. TEXT type + 4D layout (성공한 ko_citrinet 형식 차용) → "Network doesn't have a valid input meta"

원인 추정:
- Acuity 6.12의 NPY loader가 자체적으로 allow_pickle=False 사용
- 우리 캘리브레이션 데이터는 [1, 32, 768] 3D 형태인데 Acuity는 image-style 4D를 기대
- TEXT 타입 + dataset.txt + 4D 변환 매핑이 정확히 어떻게 되는지 더 조사 필요

⏸ **단계 4~6: NB export, vpm_run 검증, Android awnn 통합** (3단계 통과 후)

## 다음 시도 방향

1. **GENERATOR 타입** 사용 (Python 함수가 데이터 yield)
2. **inputmeta.yml에 redirect_to_output: true** 시도
3. **shape을 [768, 32]로 transpose해서** Conv1d 친화적 형태로 export
4. **NPY 파일을 NHWC 4D로 직접 (컨테이너 numpy 1.19.5로) 다시 저장**

## 기대 성능 (NB 성공 시)

```
현재 (CPU ONNX):  20ms / 추론 (T527 CPU)
NB 변환 후 예상:  3~5ms / 추론 (T527 NPU)  — 4~7배 가속
```

기존 Pure CNN 4L NPU 사례 (`docs/NPU_NLU_EXPERIMENTS.md`):
- TextConformer-base 4L: NB 6MB, 3.2ms
- Pure CNN 4L (4M params): NB 291KB, ~1ms

## 산출물

```
checkpoints/cnn_body_v46.onnx              5.91MB (CNN body only ONNX)
checkpoints/cnn_body_acuity/
  ├─ cnn_body_v46.json                     15KB  (Acuity import 성공)
  ├─ cnn_body_v46.data                     6MB
  ├─ inputmeta.yml                         (시도 중)
  ├─ dataset.txt                           (100개 캘리브 경로)
  └─ calib/                                100개 .bin + .npy
```
