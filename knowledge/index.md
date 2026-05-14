# t527-nlu Knowledge Index

## Overview

- [wiki/overview.md](wiki/overview.md) — t527-nlu 현재 상태 한눈에

## Projects

- [wiki/projects/t527-npu-integration.md](wiki/projects/t527-npu-integration.md) — T527 NPU NLU 통합 상태 (NB 변환, JNI, latency)

## Models

- [wiki/models/cnn-multihead-v72.md](wiki/models/cnn-multihead-v72.md) — Production NLU 모델 (v28+v72 ensemble, 서버)
- [wiki/models/version-matrix.md](wiki/models/version-matrix.md) — **전체 버전 매트릭스** (학습 모델 v1~v72 + 앙상블 + 룰 v100~v133/v73~v78 + NPU NB + 위치별 사용)

## Issues

- [wiki/issues/server-device-version-gap.md](wiki/issues/server-device-version-gap.md) — 서버 v72 vs 디바이스 v46 vs NPU v46 단독 — 세 위치 ground truth 다름

## Decisions

- [wiki/decisions/regression-harness.md](wiki/decisions/regression-harness.md) — pre-commit 회귀 체크 hook (0.4초, 99셋+219셋, -2%p 임계값)

## Evaluations

- (TBD — 르엘 219 / 골든 99 / 491 통합 측정 페이지 필요)

## Raw Sources

- [raw/catalog.tsv](raw/catalog.tsv) — Source 인덱스
- [raw/source-notes/src-model-version-gap-20260513.md](raw/source-notes/src-model-version-gap-20260513.md) — 서버↔디바이스 ONNX md5 비교

## Schema

- [schema/AGENTS.md](schema/AGENTS.md) — 운영 규칙 (팀 wiki와 동일)
- [schema/page-template.md](schema/page-template.md) — 페이지 템플릿

## 팀 wiki

- `/home/nsbb/travail/claude/T527/wewonnim/t527_llm_wiki/` — 팀장님이 만든 상위 wiki. 우리 t527-nlu는 추후 그쪽 `wiki/projects/t527-nlu.md` 등으로 흡수 가능.
