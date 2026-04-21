# NLU Iteration 3 (2026-04-21)

## 목표
이전 iteration에서 발견한 v28의 dir 학습 오류 (밝게→down) 해결 시도.

## Experiment 1: v28b — dir 보강 재학습

### 접근
- v28 warm-start + 원본 train에 189 추가 샘플 (밝게/어둡게/모드로)
- 5 epoch, lr=1e-4 fine-tune

### 결과 ❌
```
v28 원본:   TS combo 96.3%, KE fn 75.5%
v28b:       TS combo 92.74%, KE fn 75.33% (-3.6%p regression)
```

"밝게" 교정 성공:
- "거실 등 좀 밝게" → dir=up ✓
- "안방 등 밝게" → dir=up ✓
- "주방 불 어둡게" → dir=down ✓
- "거실 에어컨 냉방모드로" → dir=down ✗ (여전히 틀림)

하지만 overall regression — patch retrain의 고질적 문제.

### 교훈
- **5 epoch × 189 sample로 v28 patch** 시도 실패
- v28의 안정성을 깨뜨림
- 메모리 노트: "소규모 패치는 항상 regression 유발" 재확증

## Experiment 2: Ensemble + Post-processing Rules

### 접근
`scripts/ensemble_inference_with_rules.py`:
- 밝게 → dir=up (down 예측 시 교정)
- 어둡게 → dir=down
- 엘리베이터 호출/올라와 → exec=control
- N모드로 → dir=set

### 결과 ⚠️ 미미
```
NO rules:   TS combo 93.59%, KE fn 97.79%
With rules: TS combo 93.53% (-0.06%p)
```

- dir 97.21% 동일 (밝게 교정은 됐지만 다른 rule로 손실)
- exec +0.07%p (엘리베이터 rule 덕)
- 전체 손실 — 일부 rule의 부작용

### 분석
- **"엘리베이터 올라와" 유사 케이스**에서 test_suite 라벨이 query인 것도 있음
- Rule이 일괄 control로 바꾸면 이 케이스들 틀림
- **더 좁은 조건**의 rule 필요

## 종합: 현 최적 조합

여전히 **Ensemble Strategy B (no rules)**가 가장 좋음:
- TS combo: 93.59%
- KE fn: 97.79%
- balanced: 95.66

sap_inference_v2.py의 rule은 **단일 모델(v46)** 후처리로 유지 (ensemble과 별도).

## 다음 iteration 후보
1. **Rule 정밀화** — 엘리베이터 rule을 더 좁게 (특정 문맥만)
2. **Value Pointer** 프로토타입
3. **v46 학습 데이터 전체 라벨 수정 후 재학습** (진짜 근본 해결)
4. **실사용 로그 수집** (장기)

## 산출물

- `scripts/train_v28b_dir_fix.py` — v28b 시도 (기각)
- `scripts/ensemble_inference_with_rules.py` — rule 적용 버전
- `checkpoints/cnn_multihead_v28b.pt` — v28b 체크포인트 (archived)
- `docs/ITERATION_2026_04_21_ITER3.md` — 이 문서

## 누적 교훈

이 세션에서 모두 확인:
- ❌ Retrieval Hybrid (Option B): pool sparse
- ❌ v28 patch retrain (v28b): regression
- ⚠️ Post-processing rules: 미세 부작용
- ✅ **Ensemble Strategy B (no tweaks) 유지가 정답**

"완벽한 single silver bullet은 없다. 조합과 trade-off로 해결."
