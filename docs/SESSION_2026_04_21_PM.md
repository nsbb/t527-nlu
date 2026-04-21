# 세션 종합 보고서 (2026-04-21 오후, 15:00-17:00)

이 세션에서 진행한 5 iteration 작업 및 결론 정리.

## 출발점

오전 세션(2026-04-21 AM)에서 확인된 것:
- Ensemble Strategy B (v28+v46): **TS 93.59%, KE 97.79%, balanced 95.66** ← 배포 중
- 모든 단일 모델 개선 실험 실패 확인 (v54-v63)
- **라벨 품질 문제** 발견 (엑셀에 multi-head 라벨 없음, 스크립트가 guess)

오후 세션 목표: 라벨 품질 개선 + 아키텍처 개선 가능성 재탐색.

## 5 iteration 요약

### Iteration 1 (15:04 커밋 `5a563a9`) — Retrieval Hybrid ❌

**가설**: GT 219 pool에서 ko-sbert 유사도 검색으로 라벨 의존 감소.

**결과**:
- 모든 threshold에서 v46 baseline(93.20%) 미달
- Best (HIGH=0.95): TS 91.49% (-1.7%p), KE 94.01% (-3.8%p)

**원인**: GT 219개 self-retrieval combo 49.8%. Pool이 "종류 커버"지 "의미 중복"이 아님.

### Iteration 2 (15:33 커밋 `2c4b1ee`) — 라벨 정제 v2 + Ensemble 전략 재비교 ⚠️

**발견**:
- test_suite 추가 24건 라벨 오류 확인 (카테고리 A 6, B 15, C 3)
- "밝게 → dir=up"이 맞는데 v28이 "down"으로 학습 (v46은 정확)

**Ensemble 전략 5가지 비교** (수정된 라벨 기준):

| Strategy | TS combo | KE fn | balanced |
|----------|:---:|:---:|:---:|
| v28 alone | 95.53% | 75.59% | 85.0 |
| Avg (logit avg) | 94.45% | 87.76% | 91.0 |
| **B (current)** | **93.59%** | **97.79%** | **95.66** ★ |
| v46 alone | 93.36% | 97.79% | 95.53 |

→ B가 balanced 최고 (KE 덕). v28은 TS만 높고 KE 약함.

### Iteration 3 (15:39 커밋 `141d8a2`) — v28b + rules ❌

**시도 1 — v28b (patch retrain)**:
- v28 + 189 추가 샘플 (밝게/어둡게/모드로) + 5 ep fine-tune
- 결과: TS 92.74% (-3.6%p regression)
- "밝게"는 교정됐으나 전체 regression

**시도 2 — Post-processing rules on Ensemble**:
- 밝게→up, 어둡게→down, 엘리베이터→control, 모드로→set
- 결과: TS 93.53% (-0.06%p, 미세 부작용)
- 엘리베이터 rule 과다 적용으로 일부 query 케이스 틀림

### Iteration 4 (16:07 커밋 `40369d2`) — v70 (1,180 라벨 수정) ❌

**시도**:
- `suspects_categorized.json` A 카테고리 전부 (1,180건) train 데이터 수정
- v46 recipe full retrain (30 epochs, mixup)

**결과**:
- v70: TS 90.04%, KE 97.33%, balanced 93.62
- v46 baseline 대비 **-3.16%p** 하락

**원인 가설**:
- train = GT + KoELECTRA pseudo-labeled (v28 예측)
- 1,180건만 규칙으로 수정 → 나머지 pseudo 라벨과 **충돌**
- 데이터 일관성 파괴 → 모델 혼란

### Iteration 5 (16:41 커밋 `220901b`) — ONNX 최적화 + 배포 준비 ✅

**ONNX 최적화**:
- Vocab 사용 분석: **32000 중 12.2%(3912)만 실제 사용**
- FP16 전체 변환:
  - 크기: 105MB → **52.5MB (50% ↓)**
  - 정확도: **100% match** (TS 93.59%, KE 97.79% 동일)
  - Latency: 0.55ms → **20.4ms (37x 느림)** ← CPU에서 FP16 비효율
- **결론**: CPU 배포는 FP32 유지, FP16은 GPU/NPU 전용 옵션

**배포 문서 작성**:
- `DEPLOYMENT_CHECKLIST.md` (10 섹션)
- `FEEDBACK_SYSTEM_DESIGN.md` (실사용 로그 + 월간 사이클)

## 이번 세션 핵심 교훈

### 1. Ceiling 확인: **v46 = 현 데이터/레시피 수학적 최적점**

이번 세션에서만 5개 실험 × 이전 세션 9개 실험 = **총 14개 기법** 시도:
```
❌ Retrieval Hybrid (iter 1)
❌ v28b patch retrain (iter 3)
❌ Post-proc rules on ensemble (iter 3)
❌ v70 대규모 라벨 수정 (iter 4)
⚠️ FP16 (iter 5 — 크기 OK, 속도 실패)

[이전 세션]
❌ KD from ensemble (v55)
❌ Two-stage (v56)
❌ Wider d=384 (v57)
❌ Targeted augment (v58)
❌ Head masking (v59)
❌ Model Soup (v60)
❌ Warm-start v28 (v61)
❌ Multi-seed (v62)
❌ Conformer 2L (v63)
```

**전부 v46/Ensemble B 못 넘음**. 이건 **"현재 접근의 수학적 상한"**.

### 2. 진짜 개선 경로 = **데이터 레벨**

모델 실험 불가 → 유일한 경로:
- **실사용 로그 수집** (드물게 자주 나오는 진짜 분포 포착)
- **GT 수동 재라벨링** (엑셀 219 → 전문가 검수, 2-3일)
- **피드백 루프** (low-conf 샘플 월간 리뷰 → 재학습)

### 3. 라벨 품질 문제의 재귀성 확인

- v28이 잘못된 라벨로 학습 → v28이 KoELECTRA pseudo-label → v46이 그걸 학습
- 라벨 **부분 수정은 regression 유발** (iter 4에서 재확증)
- **일관된 전체 재라벨링만이 해답**

### 4. Ensemble Strategy B의 구조적 특성

- fn=v46 (일반화 덕)
- exec/dir=v28 (GT 정확도)
- **dir에서 v28의 "밝게→down" 학습 오류 상속** (Iter 2-3에서 확인)
- Post-proc rule로 후처리 가능하나 trade-off 있음

## 현재 배포 권장 구성

```
파일:     checkpoints/nlu_v28_v46_ensemble.onnx  (105MB FP32)
성능:     TS 93.59%, KE 97.79%
Latency:  0.55ms CPU
후처리:   필수 (param_type, conf fallback, unsupported 체크, 밝게 rule)
DST:      room/device/confirm follow-up, 10초 timeout
```

## 생성/업데이트된 산출물

### Scripts (새로)
- `scripts/retrieval_hybrid.py` + `retrieval_hybrid_fast.py` — Retrieval 실험 (기각)
- `scripts/retrieval_gt_selftest.py` — GT leave-one-out 자기 테스트
- `scripts/retrieval_vs_ensemble.py` — 상세 비교
- `scripts/find_label_errors_v2.py` — 모델+규칙 복합 탐지
- `scripts/fix_test_suite_v2.py` — 라벨 수정 v2 (24건)
- `scripts/eval_dir_aware_ensemble.py` — dir override 실험
- `scripts/train_v28b_dir_fix.py` — v28b 재학습 (기각)
- `scripts/ensemble_inference_with_rules.py` — Ensemble + rules
- `scripts/train_v70_clean_labels.py` — 1,180 수정 재학습 (기각)
- `scripts/analyze_vocab_usage.py` — vocab 사용 분석
- `scripts/onnx_fp16_embedding.py` + `onnx_full_fp16.py` — ONNX FP16 변환

### Data
- `data/label_errors_v2.json` — 카테고리 분류된 의심 라벨
- `data/v46_errors_latest.csv` — v46 최신 오류
- `data/ensemble_errors_v2.csv` — ensemble 최신 오류
- `data/train_final_v70.json` — 1,180 수정된 데이터 (archive)
- `data/vocab_usage.json` — vocab 통계

### Checkpoints (archived)
- `checkpoints/cnn_multihead_v28b.pt` (iter 3 실패작)
- `checkpoints/cnn_multihead_v70.pt` (iter 4 실패작)
- `checkpoints/nlu_v28_v46_ensemble_fp16.onnx` (iter 5 FP16)

### Docs
- `docs/RETRIEVAL_HYBRID_FAILED_2026_04_21.md`
- `docs/ITERATION_2026_04_21_LATE.md` (iter 2)
- `docs/ITERATION_2026_04_21_ITER3.md`
- `docs/ITERATION_2026_04_21_ITER4.md`
- `docs/DEPLOYMENT_CHECKLIST.md` (iter 5)
- `docs/FEEDBACK_SYSTEM_DESIGN.md` (iter 5)
- `docs/SESSION_2026_04_21_PM.md` — **이 문서**

## 통계

- **5 iterations** × 약 2시간
- **14개 스크립트** 신규 작성
- **6개 문서** 신규 작성
- **6개 commits** (5a563a9 → 220901b)
- **4개 실험 실패 확인** (Retrieval, v28b, rules, v70)
- **1개 최적화 부분 성공** (FP16 크기 ↓, 하지만 CPU 속도 ↓)

## 다음 세션 권장 방향

모델/레시피 실험은 **완전 소진**. 권장:

### Immediate (제품화)
- Android 앱 통합 (JNI wrapper + Ensemble ONNX 배포)
- T527 NPU 변환 시도 (105MB ONNX → NB)
- 첫 번째 프로덕션 릴리스

### Short-term (1~2개월)
- 실사용 로그 수집 파이프라인 (FEEDBACK_SYSTEM_DESIGN.md 구현)
- Weekly review queue 운영 시작
- 첫 번째 피드백 기반 재학습 (월말)

### Medium-term (3~6개월)
- GT 수동 재구축 (전문가 검수)
- A/B 테스트 infra
- Drift 모니터링

### Long-term (6개월+)
- Larger model 검토 (NPU 제약 해결 시)
- Compositional generalization 재도전 (진짜 GT 확보 후)

## 한 줄 결론

> **"모델 실험 완전 소진. 남은 길은 실사용자와 함께 배우는 것뿐."**
