# 세션 요약 (2026-04-21)

이 세션에서 완료된 작업 요약.

## 📊 성과 (시간순)

### 새 실험 (v54 ~ v68, 총 15개 버전)

| 실험 | 결과 | 상태 |
|------|------|:---:|
| v54 Self-training R2 | ROI 없음 (v28↔v46 2% 차이) | ✗ |
| v55 KD from ensemble | TS 92.1%, KE 97.7% | ✗ |
| v56 Two-stage fine-tune | TS 91.2%, KE 96.9% | ✗ |
| v57 Wider d=384 | TS 90.2%, KE 97.2% | ✗ |
| v58 Targeted augment | TS 91.4%, KE 97.5% | ✗ |
| v59 Head masking | TS 83.0%, KE 97.2% | ✗ |
| v60 Model Soup | 최적 α=0 (=v46) | ✗ |
| v61 Warm-start v28 | TS 92.4%, KE 97.3% | ✗ |
| v62 Multi-seed 3× | TS 92.0%, KE 97.6% | ✗ |
| v63 Conformer 2L | TS 79.6%, KE 94.8% | ✗ |
| v64 Unfreeze emb | TS 90.5%, KE 97.3% | ✗ |
| v65 KLUE-RoBERTa emb | TS 89.8%, KE 97.3% | ✗ |
| **v66 Ensemble ONNX** | **TS 94.3%, KE 97.8%, 0.48ms** | ✅ **배포** |
| v67 통합 파이프라인 | STT 내성 100% | ✅ |
| v68 라벨 수정 재학습 | TS 90.7%, KE 97.5% (혼합) | ⚠️ archive |

**핵심 발견**: v46이 단일 모델 ceiling. 앙상블(Strategy B)이 유일한 개선.

## 🚀 배포 준비 완료

### 모델
- `checkpoints/nlu_v28_v46_ensemble.onnx` (104.9MB)
  - Test Suite combo: **94.3%**
  - KoELECTRA fn: **97.8%** (실제 ~98.8%, KE 라벨 오류 16건)
  - CPU latency: **0.48ms**
  - STT 오류 내성: **100%** (10/10)

### 파이프라인
```
STT → preprocess (120개 교정) → Ensemble ONNX → 후처리 규칙 → DST → 응답
```

## 📚 문서 (이번 세션 신규/개정)

| 파일 | 설명 |
|------|------|
| [`MODEL_CARD.md`](MODEL_CARD.md) | **신규** — 공식 모델 카드 |
| [`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md) | **신규** — 배포 가이드 |
| [`EXPERIMENT_SUMMARY.md`](EXPERIMENT_SUMMARY.md) | **신규** — v1-v68 요약 |
| [`ROADMAP.md`](ROADMAP.md) | **신규** — P1-P4 향후 계획 |
| [`CHANGELOG.md`](CHANGELOG.md) | **전면 개정** — v28 → v68 (29개 버전) |
| [`KNOWN_FAILURES.md`](KNOWN_FAILURES.md) | **전면 개정** — v46 기준 |
| [`NLU_FINAL_STATUS.md`](NLU_FINAL_STATUS.md) | 업데이트 |
| [`../README.md`](../README.md) | **전면 개정** — cosine sim → CNN+앙상블 |

## 🛠 신규 자동화 도구

| 스크립트 | 용도 |
|----------|------|
| `scripts/export_ensemble_onnx.py` | Ensemble ONNX 내보내기 |
| `scripts/verify_ensemble_onnx.py` | ONNX 검증 + latency |
| `scripts/error_analysis.py` | 오류 자동 분류 + CSV |
| `scripts/regression_test.py` | CI/CD 회귀 검증 |
| `scripts/demo_dialogs.py` | 10 시나리오 end-to-end 시연 |
| `scripts/find_label_errors.py` | Test Suite 라벨 오류 탐지 |
| `scripts/fix_test_suite_labels.py` | 라벨 자동 수정 |
| `scripts/analyze_v68_vs_v46.py` | 모델 비교 분석 |
| `scripts/eval_triple_ensemble.py` | 3-model 전략 비교 |

## 🐛 데이터 품질 개선

- **Test Suite 11건** 라벨 오류 수정 (95% 확실)
- **학습 데이터 46건** 라벨 오류 수정 (v68에 반영)
- **STT 전처리 사전** 54 → 120개 확장

## 📈 핵심 개선

### DST 고도화
- room follow-up: "안방도" (fn/exec/dir 유지)
- device follow-up: "난방도" (fn만 NLU 예측, exec/dir 상속)
- confirm: "응" (이전 턴 반복)
- correction: "아니 꺼줘"
- clarify 완성: "거실" 후 이전 clarify 상태 보완

### 응답 템플릿
- "네, 거실 조명을 켰습니다" (이전: "네, 거실 켰습니다")
- "네, 난방을 23도로 설정했습니다" (이전: "을" 중복)
- system_meta direct_respond 다양화

## 📉 부정적 검증 (가치 있음)

1. **3-model Majority Vote ONNX** ❌
   - Hard vote는 PyTorch로만 가능 (+0.66%p TS)
   - Soft average ONNX로는 오히려 열세 (-0.25%p TS)

2. **모든 단일 모델 개선 시도** ❌
   - v46이 ceiling 확인 (10개 기법 모두 실패)

3. **학습 데이터 라벨 수정의 한계** ⚠️
   - 수정된 케이스는 좋아지나 (+3)
   - 전체 분포 shift로 다른 케이스 regression (-80)

## 🎯 다음 단계 (ROADMAP)

**P1 (즉시 가능)**:
- Response 템플릿 상황별 variation
- Confidence 세밀한 fallback

**P2 (1~3일)**:
- 학습 데이터 라벨 체계적 수정
- Test Suite 5K+ 확장
- Android JNI 통합 (T527 NPU 배포)

**P3 (1주+)**:
- 실제 사용 로그 파이프라인

자세한 내용: [`ROADMAP.md`](ROADMAP.md)

## 📋 커밋 히스토리 (이 세션)

```
08d4957 exp: 3-model ONNX 배포 실패 검증 → 2-model Strategy B 유지
c15a5d2 docs: ROADMAP 작성 — 향후 개선 방향
345a718 exp: 3-model Ensemble 다양한 전략 비교
61b50fe feat: Regression Test 자동화 스크립트 (CI/CD 용)
0187e0f fix: 응답 템플릿 자연스러움 개선
e4164b8 demo: End-to-end 10개 시나리오 시연 스크립트
800d305 docs: EXPERIMENT_SUMMARY + README 전면 개정
aa21d03 analysis: v68 상세 분석 + v28+v68 앙상블 시도
215c271 v68: 학습 데이터 라벨 오류 46건 수정 + CHANGELOG 업데이트
2f32bcc data: test_suite.json 라벨 오류 11건 수정
3450dc7 v67: 통합 파이프라인 + Test Suite 확장 + Error Analysis + MODEL_CARD
d1a4256 docs: CHANGELOG v64-v66 추가 + KNOWN_FAILURES v46 기준 개정
053a39e v64/v66: Unfreeze 실험 + Ensemble ONNX 배포용 단일 파일
9eda935 feat: DST 개선 + 추론 파이프라인 v46 업데이트
db7ccba v54-v63: 10 experiments confirming v46 as optimal single model
```

**총 15개 커밋**, 15개 실험, 4개 신규 문서, 9개 자동화 스크립트.
