# NLU 최종 상태 (2026-04-21)

## 모델 라인업

| 용도 | 모델 | 체크포인트 | ONNX | Test Suite | KoELECTRA fn |
|------|------|-----------|------|:---:|:---:|
| **기존 패턴 전용** | v28 | cnn_multihead_v28.pt | nlu_v28_final.onnx | **96.3%** | 75.5% |
| **균형 (pseudo-label)** | v34 | cnn_multihead_v34.pt | nlu_v34_production.onnx | 93.3% | 96.8% |
| **일반화 최고 (mixup)** | v46 | cnn_multihead_v46.pt | nlu_v46_generalization.onnx | 93.3% | **97.8%** |
| **앙상블 최적** | v28+v46 | 두 체크포인트 | — | 94.3% | **97.8%** |

### 추가 실험 결과 (v54-v62, 모두 v46 미달)

| 버전 | 기법 | TS combo | KE fn |
|------|------|:---:|:---:|
| v55 | KD from ensemble | 92.1% | 97.7% |
| v56 | Two-stage fine-tune | 91.2% | 96.9% |
| v57 | Wider (d=384) | 90.2% | 97.2% |
| v58 | Targeted augment | 91.4% | 97.5% |
| v59 | Head masking | 83.0% | 97.2% |
| v61 | Warm-start from v28 | 92.4% | 97.3% |
| v62 | Multi-seed (3×avg) | 92.0% | 97.6% |

→ **v46이 단일 모델 최적점. 9개 기법 모두 돌파 실패.**

## 아키텍처

```
ko-sbert 768d frozen → Linear 768→256 → CNN 4L (k=3,5,7,3 residual) → Global Mean Pool
  → fn_head (20: 19 known + unknown)
  → exec_head (5: control/query/judge/direct/clarify)
  → direction_head (9: on/off/up/down/open/close/set/stop/none)
  → param_head (5: none/temperature/brightness/mode/speed) + 규칙 보정
  → judge_head (5: none/outdoor/clothing/air_quality/cost)
  + Rule: room (키워드), value (regex), 미지원 액션, param_type 보정
  + STT 전처리: 54개 교정 사전 + 한글숫자 변환
  + Confidence fallback: conf<0.5 → unknown
  + DST: 멀티턴 follow-up/correction (scripts/dialogue_state_tracker.py)
```

- Total: 26.1M params (1.5M trainable)
- ONNX: 99.7MB, 0.32~0.37ms CPU
- 학습: v34 33,839개 (pseudo-labeling), v46은 mixup augmentation 추가

## 성능 한계 분석 (v54-v62 실험 근거)

1. **TS↔KE 트레이드오프는 데이터 분포 차이에 의한 구조적 한계**
   - GT exec/dir 레이블과 KoELECTRA 레이블 간 체계적 불일치
   - 단일 모델로 양쪽 동시 최적화 불가 → 앙상블이 유일한 해법

2. **Model Soup (weight 보간) 실패** — v28↔v46 loss landscape 비볼록
   - α=0.3~0.6에서 성능 급락 → prediction-level 앙상블만 유효

3. **초기화 무관** — v28 warm-start vs random init 동일 수렴점

4. **소규모 패치 데이터 역효과** — 분포 왜곡으로 항상 regression

5. **KoELECTRA val 실제 정확도 ~98.8%** — 16/34 오류가 KE 라벨 오류

## 테스트 방법

```bash
python3 scripts/test_interactive.py         # 대화형 (v46)
python3 scripts/ensemble_inference.py       # 앙상블 (v28+v46)
python3 scripts/run_test_suite.py           # 자동 Test Suite 3,043개
python3 scripts/run_test_suite.py v46       # v46으로 테스트
python3 scripts/dialogue_state_tracker.py   # 멀티턴 DST 시뮬레이션
```

## 핵심 파일

| 파일 | 설명 |
|------|------|
| `scripts/model_cnn_multihead.py` | 모델 정의 (20 fn classes) |
| `scripts/sap_inference_v2.py` | 추론 파이프라인 (v46) |
| `scripts/ensemble_inference.py` | 앙상블 추론 (v28+v46) |
| `scripts/test_interactive.py` | 대화형 테스트 |
| `scripts/run_test_suite.py` | 자동 Test Suite |
| `scripts/preprocess.py` | STT 전처리 (54개 사전 + 한글숫자 + 부사 제거) |
| `scripts/dialogue_state_tracker.py` | 멀티턴 DST |
| `scripts/model_soup.py` | Model Soup 실험 (weight interpolation) |
| `data/test_suite.json` | 3,043개 테스트 케이스 |
| `docs/MODEL_LIMITATIONS.md` | 모델 한계 9가지 분석 |
| `docs/VERSION_LOG.md` | 전체 버전 히스토리 (v1~v62) |
| `docs/KNOWN_FAILURES.md` | 알려진 실패 32개 |
| `docs/HEAD_CLASSES.md` | 5-head 클래스 정의 |

## 전체 세션 성과 (2026-04-19~21)

- **KoELECTRA(외부 데이터) fn: 75.5% → 97.8% (+22.3%p)**
- 62개 버전 실험 (v1~v62)
- 핵심 기법: pseudo-labeling (+21.3%p), mixup (+1.0%p), 앙상블
- 9개 추가 기법 실험으로 v46이 단일 모델 최적점 확인
- 한계 분석 확장 (구조적 한계 5가지)
- Test Suite 3,043개 + DST 기초 구현
