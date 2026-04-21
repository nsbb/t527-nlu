# scripts/ 디렉토리 인덱스

80+ 스크립트 중 **실제 배포/운영에 필요한 핵심**과 **아카이브된 실험용** 구분.

## 🟢 배포/운영 필수 (5개)

| 스크립트 | 용도 |
|----------|------|
| `deployment_pipeline.py` | **End-to-end 배포 파이프라인** (preprocess+ensemble+rules+DST+response) |
| `ensemble_inference_with_rules.py` | Ensemble ONNX + 10개 후처리 rule |
| `preprocess.py` | STT 교정 사전 (210+ entries) |
| `dialogue_state_tracker.py` | DST (slot filling, 5-turn history) |
| `model_cnn_multihead.py` | 모델 아키텍처 정의 (ONNX 로드 필수) |

## 🟢 Inference/데모 (5개)

| 스크립트 | 용도 |
|----------|------|
| `sap_inference_v2.py` | 단일 모델 v46 inference (rules 포함) |
| `demo_comprehensive.py` | 종합 데모 (5 카테고리 시연) |
| `demo_dialogs.py` | 대화 시나리오 데모 |
| `test_interactive.py` | 대화형 테스트 |
| `regression_test_iter9.py` | iter8/9 rule 검증 (26 tests) |

## 🟢 평가/검증 (7개)

| 스크립트 | 용도 |
|----------|------|
| `test_gt_scenarios.py` | GT 219 시나리오 전수 테스트 |
| `eval_strategies_variants.py` | 9개 ensemble 전략 비교 (iter8) |
| `eval_test_suite_v67.py` | TS v67 evaluation |
| `eval_ruel.py` | 르엘 데이터 평가 |
| `verify_ensemble_onnx.py` | ONNX 유효성 검증 |
| `run_test_suite.py` | Test Suite 통합 runner |
| `regression_test.py` | 전체 regression suite |

## 🟡 학습 스크립트 (대표 예시)

| 스크립트 | 결과 |
|----------|------|
| `train_cnn_multihead_v46.py` | v46 (단일 모델 최적) |
| `train_cnn_multihead_v28.py` | v28 (GT 정확도 최고) |
| `train_v70_clean_labels.py` | 실패 (부분 라벨 수정 regression) |
| `train_v28b_dir_fix.py` | 실패 (patch retrain regression) |

더 자세한 학습 내역: `docs/EXPERIMENT_SUMMARY.md`

## 🟡 ONNX 변환/최적화

| 스크립트 | 용도 |
|----------|------|
| `export_ensemble_onnx.py` | 앙상블 ONNX export |
| `export_triple_ensemble_onnx.py` | 3-model ensemble (기각됨) |
| `onnx_full_fp16.py` | FP16 변환 (GPU/NPU 전용) |
| `analyze_vocab_usage.py` | Vocab 사용률 분석 (12.2%) |

## 🔴 아카이브/실험용 (수정 금지)

| 스크립트 | 상태 |
|----------|------|
| `retrieval_hybrid*.py` | iter1 기각 |
| `train_v54~v70` | 모두 v46 미달 |
| `eval_triple_ensemble.py` | Majority vote 기각 |
| `find_label_errors*.py` | 라벨 오류 탐지 (1회성) |

## 사용 흐름

### 신규 개발자
1. `deployment_pipeline.py` 읽기 (API 이해)
2. `demo_comprehensive.py` 실행 (실동작 확인)
3. `regression_test_iter9.py` 실행 (rule 동작 확인)
4. 필요 시 `sap_inference_v2.py` 참고 (rule 설명)

### 성능 평가
1. `eval_test_suite_v67.py` — TS 전수 평가
2. `test_gt_scenarios.py` — GT 219 평가
3. `eval_strategies_variants.py` — ensemble 전략 비교

### 신규 rule 추가 시
1. `ensemble_inference_with_rules.py`의 `apply_post_rules()`에 rule 추가
2. `sap_inference_v2.py`의 predict 메서드에도 동일 rule 추가
3. `regression_test_iter9.py`에 assertion 추가
4. TS + KE 모두 측정 (regression 확인)
