# CI 회귀 테스트 자동화

작성: 2026-05-07

## 배경

이전 v28+v46 → v72로 학습 데이터 추가 시 일부 케이스 회귀 (combo 100%→94.1%). 
당시 골든셋/CI 자동 감지 없어서 회귀를 놓쳤음.

이번에 산업 표준 (Microsoft NLU.DevOps, Hamming AI) 따라 자동화 도구 구축.

## 도구

```bash
# 모델/규칙 변경 후 회귀 자동 감지
python3 scripts/ci_regression_check.py

# baseline 업데이트 (의도된 변경 후)
python3 scripts/ci_regression_check.py --update

# 다른 모델 테스트
python3 scripts/ci_regression_check.py --model checkpoints/nlu_v28_v72_ensemble.onnx
```

## 골든셋 5종 자동 평가

| 골든셋 | 케이스 수 | 용도 |
|--------|---------|------|
| TS 3043 | 3043 | 학습 데이터 회귀 |
| golden_99 | 99 | 작은 빠른 검증 (over-fit 위험 인지) |
| golden_491 | 491 | 정직 측정 |
| golden_indirect_56 | 56 | 비유/완곡/STT 변형 (PostRules 효과 측정) |
| golden_ruel_219 | 219 | 르엘 공식 시나리오 (자동 매핑) |

## Baseline (v28+v46, 현재 배포 모델)

| 골든셋 | fn% | exec% | dir% | combo% |
|--------|-----|-------|------|--------|
| TS 3043 | 100.00 | 100.00 | 100.00 | **100.00** |
| golden_99 | 100.00 | 97.98 | 100.00 | 97.98 |
| golden_491 | 100.00 | 100.00 | 100.00 | **100.00** |
| golden_indirect_56 | 96.43 | 64.29 | 94.64 | 62.50 |
| golden_ruel_219 | 81.74 | 78.08 | 84.47 | 60.73 |

## 회귀 감지 임계값

`-2%p` 이상 떨어지면 회귀 감지 → exit code 1

예: v72 모델 시뮬
```
❌ 회귀 3건:
  test_suite combo: 100.0% → 94.1% (Δ -5.9%p)
  golden_99 combo:   98.0% → 91.9% (Δ -6.1%p)
  golden_491 combo: 100.0% → 93.5% (Δ -6.5%p)
```

## 사용 시나리오

### 학습 데이터 추가 → train_v73 학습 후
```bash
# 새 모델 평가
python3 scripts/ci_regression_check.py --model checkpoints/nlu_v28_v73_ensemble.onnx
# → 회귀 없으면 OK, 있으면 차단
```

### PostRules 새 규칙 추가
```bash
# 즉시 검증 (모델 변경 없어도)
python3 scripts/ci_regression_check.py
# → indirect_56 골든셋에서 향상 + 다른 셋에서 회귀 없는지 자동 확인
```

### Baseline 업데이트 (의도된 향상 확정 시)
```bash
# 새 모델이 정말 좋다고 판단 시
python3 scripts/ci_regression_check.py --update
# → ci_baseline.json 갱신
```

## GitHub Actions / CI 통합 (다음 단계)

```yaml
# .github/workflows/regression.yml (예시)
on: [push]
jobs:
  regression:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pip install -r requirements.txt
      - run: python3 scripts/ci_regression_check.py
```
