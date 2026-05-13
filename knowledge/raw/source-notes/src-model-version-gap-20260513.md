# Source Note — 서버↔디바이스 모델 버전 gap 검증 (2026-05-13)

## Source

- 디바이스: T527 데브킷 `51475789d0c64881cd3`
- 검증 방법: adb run-as md5sum + 서버 md5sum 비교
- 검증 명령:
  ```bash
  adb shell run-as com.t527.smart_v2 md5sum /data/data/com.t527.smart_v2/files/nlu_v46.onnx
  md5sum checkpoints/nlu_v28_v{46,71,72}_ensemble.onnx
  ```

## Observation

```
디바이스 nlu_v46.onnx                   md5: d33228b64544ff09deb13916e2494370
서버 nlu_v28_v46_ensemble.onnx          md5: d33228b64544ff09deb13916e2494370  ← 동일
서버 nlu_v28_v71_ensemble.onnx          md5: 8f059fcdb63e85199aa347c0bdae5a45
서버 nlu_v28_v72_ensemble.onnx          md5: 8bba94b5bec9d35820b5cc8f66fce764

서버 deployment_pipeline_v2.py:206:
  onnx_path='checkpoints/nlu_v28_v72_ensemble.onnx'   ← v72
```

## Conclusion

- 디바이스 production: **v28+v46 ensemble** (서버 4월 20일 시점)
- 서버 production: **v28+v72 ensemble** (4월 28일 시점)
- gap: 한 세대 (v46 → v71 → v72)

## NPU NB 현황 (디바이스 `/data/local/tmp/`)

```
cnn_body_v46.nb              1,356,608 bytes  uint8
cnn_body_v46_int16.nb        2,742,656 bytes  int16 (production NPU 모드 default)
cnn_body_v28.nb              1,357,440 bytes  uint8 (ensemble 시도용, 폐기 권장)
```

→ NPU에는 v46 단독 cnn_body만. v72 cnn_body NB 없음.

## Linked wiki

- `wiki/overview.md`
- `wiki/issues/server-device-version-gap.md`
- `wiki/models/cnn-multihead-v72.md`
- `wiki/projects/t527-npu-integration.md`
