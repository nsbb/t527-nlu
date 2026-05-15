# data/ 폴더 구조

## Current state

`data/` 안 88개 파일을 6개 하위 폴더로 분류 (2026-05-15).

```
data/
├── golden/    평가 셋 (5) — 르엘 진짜 GT + TS 3043 + 학습시드 + 회사원본
│   ├── gt_known_scenarios_v2.json   (204) ★ production GT
│   ├── gt_unknown_scenarios.json    (15)  ★ production GT
│   ├── test_suite.json              (3043) 학습 회귀 체크용
│   ├── gt_seeds_integrated.json     학습 데이터 시드
│   └── test_ruel.csv                회사 원본 csv (응답 평가)
├── train/     학습 데이터 (33) — train_*.json, val_*.json
├── raw/       회사 원본 + 외부 데이터 (16) — 르엘 csv, AIDL xlsx, ko-electra/massive/ha
├── augment/   증강 소스 (4) — indirect, paraphrase, STT 변형
├── analysis/  오답/감사 결과 (16) — *_errors_*, label_audit_*
├── ci/        CI baseline (2)
└── archive/   레거시 (79)
```

2026-05-15 추가: data/golden/ 안의 자동 생성 시험용 파일 10개 삭제.
- 삭제: `golden_test_100/500.json`, `golden_indirect_55.json`, `gt_known_scenarios.json` (v1), `gt_unknown_scenarios_v2.json` (중복), `test_final_v9.json`, `test_suite_before_fix*.json`, `test_suite_suspects.json`, `test_suite_v67.json`
- 사유: 이름이 "golden_*"이라 진짜 production GT처럼 보였지만 자동 생성 시험용 → 측정값이 매번 헷갈렸음.
- ci_quick_check.py / ci_regression_check.py 도 219만 평가하도록 갱신.

## Known-good settings

- 새 파일 추가 시 해당 폴더에 직접 저장
- 스크립트 path는 `data/<subdir>/<file>` 형식 사용
- 옛 path (`data/<file>` 직접)는 grep으로 발견 시 갱신

## Open issues

- archive/ 안 79개 파일 — 필요 시 더 정리하거나 삭제
- raw/ Zone.Identifier 파일들 — Windows 잔여물, 제거 가능
- 옛 스크립트 중 path 갱신 누락된 것 있을 수 있음 (grep으로 점검 필요)

## Related sources

- `wiki/decisions/ruel-gt-set.md` — 르엘 GT 진짜 정의
- `wiki/decisions/regression-harness.md` — pre-commit 체크

## Last updated

- `2026-05-15`
