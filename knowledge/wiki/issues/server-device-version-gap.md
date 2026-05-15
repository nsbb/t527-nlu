# 서버↔디바이스 NLU 모델 버전 gap

## Current state

서버 production과 디바이스 production이 **다른 모델 세대**를 사용 중. 세 위치 ground truth가 다름.

| 위치 | 모델 | ONNX md5 |
|---|---|---|
| 서버 (`deployment_pipeline_v2.py`) | **v28+v72 ensemble** | 8bba94b5bec9d35820b5cc8f66fce764 |
| 디바이스 (`t527_smart_v2` 앱) | **v28+v46 ensemble** | d33228b64544ff09deb13916e2494370 |
| NPU NB (`/data/local/tmp/cnn_body_v46_int16.nb`) | **v46 단독 cnn_body int16** | (NB는 md5 무의미) |

- 서버는 v72까지 진척, 진짜 GT 219 실측 **combo 93.2% / fn 96.3%** (2026-05-15 ci_quick_check)
- 디바이스 CPU는 4월 20일 v46 ensemble 그대로 멈춤 (진짜 GT 미측정)
- 디바이스 NPU에 v72 NB 추가됨 → 진짜 GT **combo 89.0% / fn 93.6%** (2026-05-15 NpuClassifierFullEval)
- 서버 ↔ NPU gap: -4.2%p (int16 양자화 + 단일 NB ensemble 효과 손실)

영향: 매 작업마다 "v46 맞아 v72 맞아?" 혼동 발생했음. 이전 NPU 60.7% 측정값은 **자동매핑 GT** (삭제됨) 기준이었던 게 2026-05-15 확정. 진짜 GT (gt_known_v2 + gt_unknown 219) 로 통일.

## Known-good settings

- `deployment_pipeline_v2.py` 가 single source of truth — `nlu_v28_v72_ensemble.onnx`
- ensemble head 선택: fn=v72, exec=v28, dir=v72, param=v28, judge=v72
- 디바이스 ONNX 파일명은 `nlu_v46.onnx` 이지만 실제로는 ensemble (v28+v46) 파일 — 파일명만 v46 표기

## Open issues

- 디바이스 v72 ONNX 배포 미완 — `nlu_v28_v72_ensemble.onnx`를 디바이스 `/data/data/com.t527.smart_v2/files/nlu_v46.onnx`로 교체 필요 (파일명 유지 또는 코드 path 업데이트)
- v72 cnn_body 단독 추출 + int16 NB 변환 미완 — `scripts/extract_cnn_body.py` 활용 + Acuity 6.12 (inputmeta_3d.yml 재사용)
- v72 head 선택 매핑 검토 — v72 cnn_body가 5-head 모두 v72 출력이면 단일 NB 가능. 안 되면 v28+v72 둘 다 변환 후 JNI에서 head별 선택 필요
- 서버↔디바이스 자동 sync workflow 없음 → 매번 수동 push, drift 위험

## Related sources

- `raw/source-notes/src-model-version-gap-20260513.md`
- `wiki/models/cnn-multihead-v72.md`
- `wiki/projects/t527-npu-integration.md`

## Last updated

- `2026-05-13`
