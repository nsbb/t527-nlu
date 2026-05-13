# 서버↔디바이스 NLU 모델 버전 gap

## Current state

서버 production과 디바이스 production이 **다른 모델 세대**를 사용 중.

| 위치 | 모델 | ONNX md5 |
|---|---|---|
| 서버 (`deployment_pipeline_v2.py`) | **v28+v72 ensemble** | 8bba94b5bec9d35820b5cc8f66fce764 |
| 디바이스 (`t527_smart_v2` 앱) | **v28+v46 ensemble** | d33228b64544ff09deb13916e2494370 |
| NPU NB | **v46 단독 cnn_body int16** | (NB는 md5 무의미) |

세 군데 모두 ground truth가 다르다.

## Known-good settings

- 서버는 v72까지 진척 (간접표현 20/21, GT-219 94.06% combo with rule 주장)
- 디바이스는 4월 20일 v46 ensemble 그대로 멈춤
- NPU는 v46 단독 → 가장 뒤처짐

## Open issues

- 디바이스에 v72 ensemble ONNX 배포 미완 (`nlu_v28_v72_ensemble.onnx`를 `/data/data/com.t527.smart_v2/files/nlu_v46.onnx`로 교체)
- v72 cnn_body 단독 추출 + int16 NB 변환 미완
- v72 ensemble은 fn=v72/exec=v28/dir=v72 head별 다른 모델 → NPU 변환 시 두 cnn_body NB 둘 다 만들고 JNI에서 head별로 선택 필요 (구현 복잡)

## 영향

- 사용자 분석 결과 d혼동: "지금 v46이 맞아 v72가 맞아?" 매 작업마다 다른 답 나옴
- 정확도 측정 시 어느 위치 기준인지 매번 다름 (NPU 60.7% vs 서버 94.06% 주장)

## 해결 방향

1. **즉시 (정확도 회복):**
   - `cp checkpoints/nlu_v28_v72_ensemble.onnx → 디바이스 /data/data/.../nlu_v46.onnx`
   - 파일명은 nlu_v46.onnx 유지하고 내용만 v72로 (또는 코드 path 업데이트)
   - CPU 모드 정확도 즉시 v72 수준으로

2. **중기 (NPU 정확도 회복):**
   - `cnn_body_v72.onnx` 추출 (extract_cnn_body.py 활용)
   - Acuity int16 변환 (`inputmeta_3d.yml` 그대로)
   - JNI는 변경 없음 (입력/출력 shape 동일)
   - v72 head 매핑: 만약 v72 cnn_body가 5-head 모두 v72 출력이면 직접 사용. 단일이 안 되면 v28+v72 둘 다 변환 + head 선택

3. **장기:**
   - 서버↔디바이스 자동 sync workflow
   - 모든 변경 사항은 knowledge wiki에 기록 (이 페이지 갱신)

## Related sources

- `raw/source-notes/src-model-version-gap-20260513.md`
- `wiki/models/cnn-multihead-v72.md`
- `wiki/projects/t527-npu-integration.md`

## Last updated

- `2026-05-13`
