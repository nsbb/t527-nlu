# 회귀 테스트 하네스 (pre-commit)

## Current state

NLU 관련 파일 commit 시 자동으로 빠른 회귀 체크 실행. `data/ci_baseline_quick.json` 대비 fn/combo 정확도 비교, -2%p 이상 떨어지면 commit BLOCK.

- 트리거: `scripts/preprocess.py`, `ensemble_inference_with_rules.py`, `model_*.py`, `checkpoints/*.onnx`, `data/train_*`, `data/golden_*`, `data/test_suite*` 변경
- 평가 셋: golden_99 + 르엘 219 (총 318 발화)
- Latency: **0.4초** (warmup 없이도)
- Threshold: -2%p (fn 또는 combo)
- 모델: `nlu_v28_v72_ensemble.onnx`

## Known-good settings

현재 baseline (`data/ci_baseline_quick.json`, 2026-05-13):

| 셋 | fn | combo |
|---|---|---|
| golden_99 | 94.9% | 83.8% |
| golden_ruel_219 | 82.2% | 59.4% |

명령:
```bash
# 수동 실행
python3 scripts/ci_quick_check.py

# baseline 갱신 (의도된 개선 후)
python3 scripts/ci_quick_check.py --update

# hook 강제 스킵 (긴급)
SKIP_NLU_CHECK=1 git commit ...
git commit --no-verify ...
```

## Open issues

- ONNX 모델 (110MB)이 git에 없음 → 다른 머신에서 clone 시 hook 동작 불가. 환경 셋업 가이드 필요
- GitHub Actions 자동 CI는 미가동 (모델 LFS 또는 외부 스토리지 필요)
- Test Suite 3043 + golden 491 + indirect 56은 full check (`scripts/ci_regression_check.py`)에서만 — 별도 권장 cadence (주 1회 또는 PR 전)
- baseline 갱신 룰 미정립 (누가/언제 update할지)

## Related sources

- `scripts/ci_quick_check.py` — pre-commit용 빠른 체크
- `scripts/ci_regression_check.py` — full check (5개 셋)
- `data/ci_baseline_quick.json` — quick baseline
- `data/ci_baseline.json` — full baseline
- `.git/hooks/pre-commit` — 자동 실행 hook
- `.github/workflows/regression.yml` — Actions workflow (미가동)

## Last updated

- `2026-05-13`
