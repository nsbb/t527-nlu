# NLU 최종 상태 (2026-04-20)

## 모델 라인업

| 용도 | 모델 | 체크포인트 | ONNX | Test Suite | KoELECTRA fn |
|------|------|-----------|------|:---:|:---:|
| **기존 패턴 전용** | v28 | cnn_multihead_v28.pt | nlu_v28_final.onnx | **96.3%** | 75.5% |
| **균형 (pseudo-label)** | v34 | cnn_multihead_v34.pt | nlu_v34_production.onnx | 93.3% | 96.8% |
| **일반화 최고 (mixup)** | v46 | cnn_multihead_v46.pt | nlu_v46_generalization.onnx | 93.3% | **97.8%** |
| **앙상블 최적** | v28+v46 | 두 체크포인트 | — | 94.3% | **97.8%** |

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
| `data/test_suite.json` | 3,043개 테스트 케이스 |
| `docs/MODEL_LIMITATIONS.md` | 모델 한계 9가지 분석 |
| `docs/VERSION_LOG.md` | 전체 버전 히스토리 (v1~v51) |
| `docs/KNOWN_FAILURES.md` | 알려진 실패 32개 |
| `docs/HEAD_CLASSES.md` | 5-head 클래스 정의 |

## 세션 성과 (2026-04-19~20)

- **KoELECTRA(외부 데이터) fn: 75.5% → 97.8% (+22.3%p)**
- 51개 버전 실험 (v1~v51)
- 핵심 기법: pseudo-labeling, mixup augmentation, 앙상블
- 한계 분석 9가지 + 해결 방향 문서화
- Test Suite 3,043개 + DST 기초 구현
