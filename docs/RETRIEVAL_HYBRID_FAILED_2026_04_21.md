# Retrieval Hybrid 실험 — 기각 (2026-04-21)

ROADMAP Option B (단기 최우선 추천)로 제안했던 **Retrieval Hybrid** 실험 결과 기각.

## 가설 (예상)

> 입력 → ko-sbert embedding → GT pool cosine similarity  
> sim > 0.85: GT 라벨 직접 사용 → 라벨 품질 의존 감소  
> 0.5 < sim < 0.85: Multi-head fallback  
> sim < 0.5: unknown

**기대**: GT 품질만 좋으면 해당 범위에서 100% 정확

## 실제 결과

### 전체 성능 비교 (vs v46 baseline)

| 설정 | TS combo | KE fn |
|------|:---:|:---:|
| **v46 단독 (baseline)** | **93.20%** | **97.79%** |
| Hybrid HIGH=0.95 | 91.49% (-1.7%p) | 94.01% (-3.8%p) |
| Hybrid HIGH=0.90 | 90.21% | 93.95% |
| Hybrid HIGH=0.85 | 85.21% | 93.29% |
| Hybrid HIGH=0.80 | 74.47% | 91.47% |
| Hybrid HIGH=0.70 | 57.57% | 84.31% |

**모든 threshold에서 baseline보다 나쁨**. Threshold 낮출수록 급격히 악화.

### 모드별 사용률 (Test Suite)

| HIGH | Retrieval% | Model% | Unknown% |
|:---:|:---:|:---:|:---:|
| 0.70 | 67.1% | 23.5% | 9.3% |
| 0.85 | 17.9% | 72.8% | 9.3% |
| 0.95 | 1.5% | 89.2% | 9.3% |

Retrieval 쓸 때 **매번 틀림** → 낮은 threshold = 많이 사용 = 더 악화.

## 근본 원인 — GT Leave-One-Out 자기 테스트

GT 219개에 대해 "자신 제외, 나머지로 retrieval" → 예측 정확도:

```
fn:    161/219 = 73.5%
exec:  149/219 = 68.0%
dir:   166/219 = 75.8%
combo: 109/219 = 49.8%  ← 절반도 못 맞춤!

Top-1 sim 분포:
  mean: 0.732
  min:  0.344
  sim>0.85: 46/219 (21%)
  sim>0.9:  26/219 (12%)
```

**해석**:
- GT 219개는 **종류별 시나리오 커버**지, **의미 중복**이 아님
- 대부분 sim < 0.85 → retrieval로 가장 가까운 이웃이 **의미적으로 다른** 시나리오
- 예: "거실 불 켜줘"의 가장 가까운 GT는 다른 fn의 발화일 수 있음

## 왜 가설이 틀렸나

### 예상
- GT = dense semantic pool (같은 의미의 변형들)
- Retrieval = 내 발화와 의미상 동일한 GT 찾기

### 현실
- GT = sparse scenario coverage (각기 다른 시나리오)
- Retrieval = 의미상 다른 시나리오 중 "문자적으로" 가장 가까운 것
- 결과: 우연히 맞추는 게 절반 이하

## 시도해본 우회
- Pool에 24K 증강 데이터 추가 검토 → **기각**: 증강 자체가 rule-based라 라벨 노이즈 증폭
- Threshold 조정 → **모두 baseline보다 나쁨**
- v46 fallback 활용 → 여전히 retrieval 선택된 부분에서 손실

## 결론

**"GT pool이 좋으면 retrieval 정확도 올라간다"는 가정 자체가 현 상황에선 성립 안 함**.  
dense pool을 만들려면 **수천 건 manual 증강** 필요 → ROADMAP Option A (수동 라벨 프로젝트)와 비슷한 비용.

실용적 결론: **v46 (또는 v28+v46 앙상블) 유지가 최적**.  
Retrieval은 **pool 밀도 충분할 때** (예: 사용 로그 수만 건 축적 후) 재검토.

## 산출물 (기각됐지만 재현 가능)

- `scripts/retrieval_hybrid_fast.py` — 배치 처리 효율 버전 (1M context CUDA, 5초)
- `scripts/retrieval_hybrid.py` — 초기 single-query 버전 (느림, 교훈으로 보존)
- `scripts/retrieval_gt_selftest.py` — GT leave-one-out 자기 테스트

## 다음 방향 (ROADMAP 재조정)

~~B. Retrieval Hybrid~~ ❌ 기각

남은 옵션:
- **A. GT 수동 재라벨링** (2~3일, 근본 해결)
- **C. Value Pointer Head** (3~5일, 복합 명령 대응)
- **F. 실사용 로그 + 피드백 루프** (장기)

다음 iteration은 **C (Value Pointer)** — 코드 수정으로 즉시 검증 가능.
