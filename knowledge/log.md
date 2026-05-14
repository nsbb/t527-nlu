# t527-nlu Knowledge Log

## 2026-05-14

- **전체 버전 매트릭스 정리** — `wiki/models/version-matrix.md`
  - 4종 버전 시리즈 분리 (학습 모델 / 앙상블 / 룰 코드 / NPU NB)
  - 학습 모델 최종: v72 (4/28)
  - 룰/DST 코드 최종: v78 (4/28)
  - 후처리 룰 v100~v133 별도 추적
  - 위치별 현재 사용 모델 표 (서버 v72 / 디바이스 CPU v46 / 디바이스 NPU v46 int16) — gap 명시
- v73~v78이 모델이 아니라 룰/코드 revision임을 확정 (cnn_multihead_v73.pt 등 없음)
- docs/VERSION_LOG.md (416줄) + docs/CHANGELOG.md (2294줄) 가 한 줄짜리 commit 메시지의 빈약함을 보완하는 진짜 source임을 명시

## 2026-05-13 (오후)

- **회귀 테스트 하네스 도입** — pre-commit hook + ci_quick_check.py
  - 트리거: NLU 영향 있는 파일 변경 시 자동 실행
  - 평가: golden_99 + 르엘 219 (~0.4초)
  - Baseline: `data/ci_baseline_quick.json`
  - Threshold: -2%p 이상 떨어지면 commit BLOCK
- 초기 baseline 측정값:
  - golden_99: fn 94.9% / combo 83.8%
  - 르엘 219: fn 82.2% / combo 59.4%
- 새 페이지: `wiki/decisions/regression-harness.md`

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
