# t527-nlu Knowledge Log

## 2026-05-13

- **Knowledge wiki 초기화** — 팀 wiki (`wewonnim/t527_llm_wiki`) schema 그대로 따름
- 검증된 fact 정리:
  - 서버 production: `nlu_v28_v72_ensemble.onnx` (md5 8bba94b5...)
  - 디바이스 production: `nlu_v46.onnx` = `nlu_v28_v46_ensemble.onnx` (md5 d33228...)
  - NPU NB: v46 단독 cnn_body int16 (`cnn_body_v46_int16.nb`)
  - → 세 위치 ground truth가 다름
- 페이지 작성:
  - `wiki/overview.md`
  - `wiki/models/cnn-multihead-v72.md`
  - `wiki/projects/t527-npu-integration.md`
  - `wiki/issues/server-device-version-gap.md`
- Source note:
  - `raw/source-notes/src-model-version-gap-20260513.md`
- Catalog:
  - `raw/catalog.tsv` 1행 추가
- Open work:
  - 디바이스 v46 → v72 ONNX 교체
  - v72 cnn_body NB 변환
  - 491 NPU eval JIT crash 원인 분석
  - 르엘 GT-219 "뉴스 dir=on/exec=control" 13건 라벨 재검토
