# t527-nlu Knowledge Log

## 2026-05-15 (오후)

- **v72 NB 정확도 측정 — v46 NB와 거의 동등 또는 약간 낮음**
  - Golden 99: combo 89.9% (v46과 동등), fn 99% (v46 100% 대비 -1)
  - 르엘 219: combo 58.0% (v46 59.8% 대비 -1.8)
  - 원인: NPU에 올린 v72는 단일 cnn_body. 서버 production v72는 ensemble (fn/dir=v72, exec/param=v28)
  - 즉 ensemble 효과 NPU에서 못 살림
- 25개 batch + System.gc() + 120ms sleep으로 99/219 평가 가능 (491은 여전히 crash)
- 다음 작업: v28 cnn_body int16 NB 추가 변환 + JNI에서 head별 선택 → ensemble 재현
- 추가: `raw/source-notes/src-v72-nb-eval-20260515.md`

## 2026-05-15

- **v72 cnn_body NB 변환 성공**
  - `cnn_multihead_v72.pt` → `nlu_v72_generalization.onnx` (export_v72_standalone_onnx.py)
  - extract_cnn_body.py + shape fix [1,32,768] + onnxsim
  - Acuity 6.12 import → int16 quantize (fl=15) → export NB
  - 결과: `/data/local/tmp/cnn_body_v72_int16.nb` (2.74MB)
- **NpuClassifierTestActivity v72 NB 검증 (10 시나리오)**
  - ONNX 26.5ms vs NPU 9.3ms (2.8x speedup)
  - ONNX↔NPU agreement: 10/10
  - 모든 발화 정답 (raw, post-rules 없이)
- **미해결: 큰 셋 (99/491/219) batch 평가시 ART JIT crash 재발**
  - libart.so JitCodeCache::Reserve / GarbageCollectCache
  - 정량 정확도 측정 보류
- 추가: `raw/source-notes/src-v72-nb-conversion-20260515.md`
- index.md, wiki/projects/t527-npu-integration.md 갱신

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
