# NLU Iteration 2026-04-21 Late (Multi-experiment session)

## 목표
ROADMAP Option B (Retrieval Hybrid) 실험 → 실패 → 방향 전환 → 라벨 정제 + 앙상블 전략 재검토.

## Experiment 1: Retrieval Hybrid ❌

- 결과: 모든 threshold에서 v46 baseline 미달
- 근본 원인: GT 219 self-retrieval combo 49.8% (sparse pool)
- 상세: `docs/RETRIEVAL_HYBRID_FAILED_2026_04_21.md`

## Experiment 2: v46 오류 패턴 재분석 + 라벨 수정 v2

### 발견
이전 test_suite 라벨 수정(v1, 11건) 이후 추가 24건 발견:

**Category A** (모델+규칙 일치, 매우 확실) — 6건
- "남방 올려쥬/날방 올려/..." dir: on → up
- "도어록 열어쥬" dir: none → open

**Category B** (규칙만, 선별) — 15건
- "밝게" → up (down으로 라벨된 것 4건): `거실 등 좀 밝게`, `안방 등 밝게`, `주방 불 좀 밝게`, `작은방 좀 밝게`
- 블라인드/커튼 올려 → up (stop 5건)
- 수치 질의 exec → query (6건)

**Category C** (모델 high-conf) — 3건
- "남방 꺼쥬", "에어컨꺼" dir: on → off
- "히팅 켜줘" dir: up → on

스크립트: `scripts/fix_test_suite_v2.py`

### 결과 (수정된 라벨 기준)

| 모델 | 이전 TS | 수정 후 TS |
|------|:---:|:---:|
| v28 alone | 96.3% | **95.53%** |
| v46 alone | 93.3% | 93.36% |
| Ensemble B | 94.3% | 93.59% |

## Experiment 3: Ensemble 전략 재비교

**v28은 `밝게→dir=down`으로 학습** (잘못된 라벨 학습), v46은 `밝게→dir=up` (pseudo+mixup으로 교정).
Ensemble Strategy B는 dir을 v28에서 가져오므로 v28의 실수를 상속.

### 전략별 성능 (수정된 라벨, TS 3043 + KE 1536)

| Strategy | TS combo | KE fn | balanced |
|----------|:---:|:---:|:---:|
| **v28 alone** | **95.53%** | 75.59% | 85.0 |
| Avg (logit avg) | **94.45%** | 87.76% | 91.0 |
| B (fn=v46, e=v28, d=v28) | 93.59% | **97.79%** | **95.66** |
| B2 (fn=v46, e=v28, d=v46) | 93.07% | 97.79% | 95.39 |
| B3 (fn=v46, e=v46, d=v46) | 93.36% | 97.79% | 95.53 |
| v46 alone | 93.36% | 97.79% | 95.53 |

### 인사이트
1. **v28은 TS 최고**지만 KE 75%로 out-of-domain 취약
2. **Strategy B가 여전히 balanced 최고** (KE 덕)
3. **Avg(logit averaging)**: KE 87.76%로 v28과 B 사이 trade-off
4. 배포 추천: **Strategy B 유지** (KE가 외부 일반화 대리 지표로 더 중요)

## 교훈

### 검증된 것
- **v46의 dir head가 v28보다 낫다** (밝게, 냉방모드 등 pseudo-label 효과)
- Ensemble Strategy B의 dir을 v28에서 가져오는 건 **일부 케이스에 약점**
- 라벨 수정 24건 — 의미적으로 올바른 라벨로 교체하면 v28이 약간 하락

### 한계
- 라벨 수정 후에도 v28이 95.53%로 여전히 **학습 데이터에 과적합**
- Retrieval Hybrid는 dense pool 없이 불가능
- 추가 구조적 개선 없이 TS+KE 동시 최적화 여전히 어려움

## 다음 방향 (새 iteration)

1. **v46 + v28 dir-aware 앙상블**: dir이 "밝게/어둡게/냉방모드" 포함 시 v46 dir, 나머지는 v28
2. **Value Pointer Head** 프로토타입 (ROADMAP Option C)
3. **학습 데이터 1,176건 라벨 수정 후 재학습** (ROADMAP Option A 축소판)

## 산출물

- `scripts/retrieval_hybrid_fast.py` — Retrieval 실험 (기각, 보존)
- `scripts/find_label_errors_v2.py` — 모델+규칙 복합 탐지
- `scripts/fix_test_suite_v2.py` — 라벨 수정 v2 (24건)
- `docs/RETRIEVAL_HYBRID_FAILED_2026_04_21.md` — Option B 실패 분석
- `docs/ITERATION_2026_04_21_LATE.md` — **이 문서**
- `data/v46_errors_latest.csv` — 현재 v46 오류 CSV
- `data/label_errors_v2.json` — 탐지된 라벨 오류 카테고리 분류
