# NLU Iteration 4 (2026-04-21)

## 목표
이전 3개 iteration의 누적 교훈: patch 수정은 regression. **대규모 수정 + full retrain**이 답일까?

## Experiment: v70 — 1,180 라벨 수정 후 full retrain

### 접근
- `data/suspects_categorized.json`의 'A' category 전체 적용:
  - A_clear_error_dir_none: 809건
  - A_direct_to_query: 159건
  - A_clear_error_opposite: 142건
  - A_blind_direction_error: 52건
  - A_elevator_call_exec: 14건
  - 총 1,180건 수정
- v46 recipe로 full retrain (from scratch)
- 30 epochs, mixup 30%, cosine LR

### 결과 ❌

| 모델 | TS combo | KE fn | balanced |
|------|:---:|:---:|:---:|
| **Baseline v46** | 93.20% | 97.79% | **95.47** |
| **Baseline Ensemble B** | 93.59% | 97.79% | **95.66** |
| v70 (1180 labels fixed) | 90.04% | 97.33% | 93.62 |

**v70이 v46보다 모든 지표에서 떨어짐** (-3.16%p TS, -0.46%p KE).

### 왜 실패?

**가설: 부분 수정이 데이터 일관성 파괴**

- v34 = 원본 GT + KoELECTRA pseudo-labeled 13K
- KoELECTRA 라벨은 v28이 예측해서 일관되게 생성됨
- **1,180건만 규칙으로 수정하면 나머지 pseudo 라벨과 충돌**
- 예: "~맞춰줘" 중 일부만 dir=set으로 수정, 나머지는 여전히 v28의 예측값 (다를 수 있음)
- 모델이 "같은 구조, 다른 라벨" 케이스를 받아서 혼란

### 교훈

```
이번 세션 모든 실험 누적:
  iter 1 — Retrieval Hybrid           ❌ (pool sparse)
  iter 2 — 라벨 수정 v2 (24건)         ⚠️ (일부 개선)
  iter 2 — Ensemble 전략 재비교         ✅ (B 유지 확인)
  iter 3 — v28b (patch retrain)         ❌ (-3.6%p regression)
  iter 3 — Post-proc rules              ⚠️ (미세 부작용)
  iter 4 — v70 (대규모 수정 + retrain)   ❌ (-3.16%p regression)
```

**모든 방향에서 v46 또는 Ensemble B 못 넘음**. 

## 근본적 한계 확인

이 NLU 시스템의 성능 천장은:
- 현재 데이터 (24.5K 증강 기반, 일부 라벨 불일치)
- ko-sbert frozen embedding
- CNN 4L architecture

위 조합의 **수학적 최적점이 v46 + Ensemble B** (TS 93.6%, KE 97.8%).

**이를 돌파하려면**:
1. **진짜 GT 수동 재구축** (엑셀 219 → 전문가 검수)
2. **실사용자 로그 대량 수집** (도메인 adaptation)
3. **더 큰 사전훈련 모델** (예: KLUE-RoBERTa-large fine-tune, NPU 제약 고려)

**이 세션에서 할 수 있는 모든 모델/학습 레벨 실험은 소진됨**.

## 실용적 결론

- **배포 유지**: Ensemble Strategy B (v28+v46), TS 93.6% + KE 97.8%
- **개선 경로는 데이터 레벨**로 이동:
  - GT 수동 재라벨링 (ROADMAP Option A)
  - 실사용 피드백 루프 (ROADMAP Option F)
- **모델/레시피 추가 실험은 무의미** (확인됨)

## 산출물

- `scripts/train_v70_clean_labels.py` — 대규모 수정 + retrain (기각)
- `data/train_final_v70.json` — 1,180건 수정된 데이터
- `checkpoints/cnn_multihead_v70.pt` — archived
- `docs/ITERATION_2026_04_21_ITER4.md` — **이 문서**

## 다음 iteration 방향

옵션이 제한됨. 가능한 것:
1. **ONNX 최적화** — 현재 105MB, 크기/속도 개선
2. **Value Pointer Head 프로토타입** (이론적 가능성 탐색)
3. **실사용 피드백 시스템 설계** (제품화 방향 가이드)
4. **최종 배포 체크리스트** 작성

iter5는 ONNX 최적화 또는 배포 준비 자료 작성으로.
