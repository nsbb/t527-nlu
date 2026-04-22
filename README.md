# T527 온디바이스 NLU — CNN 5-Head + Ensemble + 후처리 rules (iter9)

르엘 어퍼하우스 AI 월패드 (T527 NPU) STT 텍스트 → Structured Action → 기기 제어.

## 핵심 지표 (iter9 최종, 2026-04-22)

| 지표 | 값 |
|------|:---:|
| **Test Suite combo** | **97.11%** (3,043 케이스, +2.23%p iter8→9) |
| **KoELECTRA fn** | **97.20%** (외부 1,536 케이스) |
| GT 219 combo | 95.0% (ensemble + rules) |
| STT 오류 내성 | 100% (10/10) |
| CPU 추론 지연 | 0.67ms (preprocess+ensemble+rules) |
| 파일 크기 | 104.9MB (FP32 ONNX, 변경 없음) |

→ **배포 모델**: `checkpoints/nlu_v28_v46_ensemble.onnx`  
→ **배포 스크립트**: `scripts/deployment_pipeline.py` (end-to-end 통합)

## 아키텍처

```
사용자 발화 "거실 에어컨 23도로 맞춰줘"
    ↓ STT (Citrinet)
"거실 에어컨 23도로 맞춰줘"
    ↓ preprocess.py (STT 교정 120개 + 한글숫자 변환)
"거실 에어컨 23도로 맞춰줘" (정제)
    ↓ 토크나이저 (ko-sbert-sts BertTokenizer, max_len=32)
input_ids [1, 32]
    ↓ Ensemble ONNX v28+v46 (CNN 5-Head)
5개 logits (fn/exec/dir/param/judge)
    ↓ argmax + param_type 규칙 보정 + confidence fallback
preds = {fn, exec, dir, param, judge}
    ↓ Rule slots (room 정규식, value 정규식)
    ↓ DST (10초 timeout, 멀티턴)
resolved = {fn, exec, dir, room, value, judge}
    ↓ generate_response (템플릿)
"네, 거실 에어컨 온도를 23도로 설정합니다."
    ↓ TTS
```

## 5-Head 구조

| Head | 클래스 수 | 예시 |
|------|:---:|------|
| **fn** | 20 | light_control, heat_control, ac_control, ..., unknown |
| **exec_type** | 5 | query_then_respond, control_then_confirm, query_then_judge, direct_respond, clarify |
| **param_direction** | 9 | none, up, down, set, on, off, open, close, stop |
| **param_type** | 5 | none, temperature, brightness, mode, speed |
| **judge** | 5 | none, outdoor_activity, clothing, air_quality, cost_trend |

## 모델 라인업

| 용도 | 모델 | 파일 | TS combo | KE fn |
|------|------|------|:---:|:---:|
| **배포 (권장)** | v28+v46 Ensemble | `nlu_v28_v46_ensemble.onnx` | **94.3%** | **97.8%** |
| GT 패턴 전용 | v28 | `nlu_v28_final.onnx` | 96.3% | 75.5% |
| 일반화 최고 | v46 | `nlu_v46_generalization.onnx` | 93.3% | 97.8% |
| 균형 (pseudo-label) | v34 | `nlu_v34_production.onnx` | 93.3% | 96.8% |

## 아키텍처 세부

```
ko-sbert 768d frozen → Linear 768→256 → CNN 4L (k=3,5,7,3 residual) → Global Mean Pool
  → fn_head (20: 19 known + unknown)
  → exec_head (5)
  → direction_head (9)
  → param_head (5) + 규칙 보정
  → judge_head (5)
  + Rule: room (키워드), value (정규식), 미지원 액션, param_type 보정
  + STT 전처리: 120개 교정 사전 + 한글숫자 변환
  + Confidence fallback: conf<0.5 → unknown
  + DST: 멀티턴 (scripts/dialogue_state_tracker.py)
```

- 파라미터: **26.1M 전체, 1.5M trainable** (임베딩 frozen)
- 앙상블: **52.3M 전체 (v28+v46 내장)**

## 빠른 시작

```bash
# 대화형 테스트
python3 scripts/sap_inference_v2.py

# 단일 발화 테스트
python3 scripts/sap_inference_v2.py "거실 에어컨 23도로 맞춰줘"

# 앙상블 ONNX 검증
python3 scripts/verify_ensemble_onnx.py

# Test Suite 평가 (3,043개)
python3 scripts/run_test_suite.py            # v28
python3 scripts/run_test_suite.py v46        # v46

# 오류 자동 분석
python3 scripts/error_analysis.py ensemble

# DST 시뮬레이션
python3 scripts/dialogue_state_tracker.py
```

## 핵심 파일

### 모델 & ONNX
- `checkpoints/nlu_v28_v46_ensemble.onnx` — **배포 모델 (권장)**
- `checkpoints/cnn_multihead_v28.pt`, `cnn_multihead_v46.pt` — PyTorch 원본
- `checkpoints/nlu_v46_generalization.onnx`, `nlu_v28_final.onnx` — 단일 ONNX

### 스크립트
- `scripts/sap_inference_v2.py` — end-to-end 추론 파이프라인
- `scripts/preprocess.py` — STT 전처리 (120개 교정)
- `scripts/dialogue_state_tracker.py` — DST (room/device/confirm follow-up)
- `scripts/run_test_suite.py` — Test Suite 평가
- `scripts/error_analysis.py` — 오류 자동 분류 + CSV
- `scripts/export_ensemble_onnx.py` — Ensemble ONNX 내보내기
- `scripts/verify_ensemble_onnx.py` — ONNX 검증 & latency 측정

### 데이터
- `data/test_suite.json` — 3,043개 테스트 (라벨 오류 11건 수정됨)
- `data/test_suite_v67.json` — 확장 3,109개
- `data/koelectra_converted_val.json` — 외부 평가 1,536개

### 문서
- `docs/MODEL_CARD.md` — 공식 모델 카드
- `docs/DEPLOYMENT_GUIDE.md` — 배포/통합 가이드
- `docs/KNOWN_FAILURES.md` — 실패 패턴 분석
- `docs/CHANGELOG.md` — 버전별 변경 (v28-v68, 29개 버전)
- `docs/EXPERIMENT_SUMMARY.md` — 전체 실험 요약 (v1-v68)
- `docs/VERSION_LOG.md` — 실험별 상세
- `docs/MODEL_LIMITATIONS.md` — 구조적 한계 9가지
- `docs/HEAD_CLASSES.md` — 5-head 클래스 정의

## 실험 히스토리 요약

68개 버전 (2026-04-13 ~ 04-21), 주요 마일스톤:

- **v28** (04-19): GT 기반 Test Suite 3K, combo 96.3%
- **v34** (04-19): Pseudo-labeling → KE 75.5%→96.8% (+21.3%p) 핵심 기여
- **v46** (04-20): Mixup → KE 97.8% 단일 모델 최적
- **v54-v63** (04-21): 10개 추가 실험 모두 v46 미달 → ceiling 확인
- **v66** (04-21): **Ensemble ONNX 배포** — TS 94.3%, KE 97.8%, 0.48ms
- **v67** (04-21): 통합 파이프라인 + STT 내성 100%
- **v68** (04-21): 학습 라벨 수정 실험
- **iter9** (04-21~22): **후처리 rule 10개 + DST 고도화** — **TS 97.11%, KE 97.20%, 0.67ms** ★

자세한 내역: [`docs/EXPERIMENT_SUMMARY.md`](docs/EXPERIMENT_SUMMARY.md)

## 이전 방식 (폐기)

> 초기에는 **문장 임베딩 + cosine similarity** 방식 사용.
> KoELECTRA 분류(94개 intent)도 시도했으나 조합 폭발 + 간접 표현 문제.
> 현재는 **CNN 5-head 멀티태스크 분류**로 전환 (v1부터).

## 라이센스

내부 사용 (HDC Labs)

## 개발팀

HDC Labs T527 NLU Team
