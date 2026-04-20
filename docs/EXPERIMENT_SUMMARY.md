# 실험 전체 요약 (v1 ~ v68)

## 핵심 결과

| 지표 | v1 (초기) | v28 (GT 최적) | v46 (일반화) | **v28+v46 앙상블 (권장 배포)** |
|------|:---:|:---:|:---:|:---:|
| Test Suite combo | ~85% | **96.3%** | 93.3% | **94.3%** |
| KoELECTRA fn | ~45% | 75.5% | **97.8%** | **97.8%** |
| 파일 | - | nlu_v28_final.onnx | nlu_v46_generalization.onnx | nlu_v28_v46_ensemble.onnx |
| 크기 | - | 100MB | 100MB | **105MB** |
| 추론 | - | 0.37ms | 0.37ms | **0.48ms** (CPU) |

## 주요 발견 (시간순)

### 1. v1 → v28 (2026-04-13 ~ 04-19): 기반 구축
- CNN 5-head + frozen ko-sbert 임베딩 + 증강 데이터 20K
- STT 전처리 사전 54→120개
- confidence fallback + param_type 규칙 보정
- Test Suite 3,043개 자동화
- **GT 패턴에서 96.3% combo 달성**

### 2. v29 ~ v33 (2026-04-19): KoELECTRA 통합 시도
- v33 직접 병합 → TS 90.5% (regression)
- **v34 Pseudo-labeling** (v28로 KoELECTRA exec/dir 재라벨) → KE 96.8% (+21.3%p)
- 핵심 교훈: **라벨 품질 > 양**

### 3. v35 ~ v53 (2026-04-20): Augmentation 기법
- **v46 Mixup** (같은 fn 내 발화 교체 30%) → KE 97.8% **단일 모델 최적점**
- Label Smoothing, Focal Loss, SupCon, CutMix, R-Drop 등 모두 시도
- 교훈: **Mixup만 효과 있음. 다른 기법은 복잡도만 증가**

### 4. v54 ~ v63 (2026-04-21): 10개 추가 실험, 모두 실패
- KD, Two-stage, Wider, Head masking, Targeted augment, Model Soup, Warm-start, Multi-seed, Conformer
- **모두 v46 미달** → v46이 단일 모델 ceiling
- Model Soup이 특히 의미 있음: v28↔v46 weight 공간 비볼록 확인

### 5. v64 ~ v68 (2026-04-21): 기타 탐색
- v64 Unfreeze, v65 KLUE-RoBERTa: 실패 (ko-sbert가 이미 최적)
- **v66 Ensemble ONNX 배포** (TS 94.3%, KE 97.8%, 0.48ms) ★
- v67 통합 파이프라인 + Test Suite 확장 (3109개, STT 내성 100%)
- v68 라벨 수정 재학습: 의도한 개선(+3) < 전체 regression(-80)

## 실험 횟수 & 시간

- **총 68개 버전** 실험 (v1 ~ v68)
- 2026-04-13 ~ 2026-04-21 (약 9일간)
- 성공한 주요 기법: pseudo-labeling (+21%), mixup (+1%), 앙상블 (+1%)

## 핵심 교훈

1. **데이터 품질 > 모델 복잡도** — 작은 모델(1.5M trainable)로 98% 달성
2. **앙상블 = 최후의 수단** — 단일 모델 한계 시 효과적
3. **Weight space는 비볼록** — Model Soup 실패가 증거
4. **소규모 패치는 분포 왜곡** — regression 유발 (v29-v33, v58, v68 모두 동일)
5. **STT 전처리가 중요** — 120개 교정 사전으로 STT 내성 100%
6. **Test Suite의 라벨 품질 중요** — 11건 발견, 학습 데이터에도 오류 존재
7. **CNN이 Conformer보다 적합** at 24.5K 샘플 (Attention은 더 많은 데이터 필요)

## TS vs KE 트레이드오프

**구조적 한계**: GT 데이터는 정확한 exec/dir 패턴(→높은 TS), KoELECTRA 데이터는 다양한 fn 표현(→높은 KE).
두 데이터의 exec/dir 라벨이 체계적으로 다름 → 단일 모델로 양쪽 동시 최적 불가.

**유일한 해결책**: Prediction-level 앙상블 (weight averaging은 loss landscape 비볼록으로 실패).

## 최종 권장 배포 스택

```
STT (Citrinet) 
  ↓
preprocess.py (120개 STT 교정 + 한글숫자)
  ↓
Ensemble ONNX (v28+v46 Strategy B, 105MB, 0.48ms)
  ↓
param_type 규칙 보정 + confidence fallback(conf<0.5→unknown)
  ↓
Rule-based: room(정규식), value(정규식), 미지원 액션 체크
  ↓
DST (멀티턴, 10초 timeout, room/device/confirm follow-up)
  ↓
generate_response (템플릿 기반)
  ↓
TTS
```

**성능**: Test Suite 94.3% + KoELECTRA 97.8% + STT 내성 100%

## 향후 개선 방향 (미탐색)

1. **더 큰 학습 데이터** — 24K → 50K+ (수동 라벨 포함)
2. **실사용 로그 수집** — 피드백 루프
3. **학습 데이터 라벨 오류 체계적 수정** — v68이 보여준 방향, 전체 규모 필요
4. **KoELECTRA 라벨 품질 개선** — 16건은 우리 모델이 맞음, 더 많을 수도
5. **DST 확장** — 5턴 이상 문맥 + slot filling
6. **도메인 적응** — 월패드 실제 사용 패턴 반영

## 파일 위치

### 모델
- `checkpoints/nlu_v28_v46_ensemble.onnx` — **배포 권장**
- `checkpoints/cnn_multihead_v28.pt`, `cnn_multihead_v46.pt` — PyTorch 원본

### 스크립트
- `scripts/sap_inference_v2.py` — end-to-end 추론 (v46 단독)
- `scripts/ensemble_inference.py` — PyTorch 앙상블
- `scripts/export_ensemble_onnx.py` — Ensemble ONNX 내보내기
- `scripts/verify_ensemble_onnx.py` — Ensemble ONNX 검증
- `scripts/preprocess.py` — STT 교정 (120개)
- `scripts/dialogue_state_tracker.py` — DST
- `scripts/run_test_suite.py` — Test Suite 평가
- `scripts/error_analysis.py` — 오류 자동 분류

### 문서
- `docs/MODEL_CARD.md` — 공식 모델 카드
- `docs/DEPLOYMENT_GUIDE.md` — 배포 가이드
- `docs/KNOWN_FAILURES.md` — 알려진 실패 패턴
- `docs/MODEL_LIMITATIONS.md` — 구조적 한계 9가지
- `docs/CHANGELOG.md` — 버전별 변경 (29개 버전)
- `docs/VERSION_LOG.md` — 실험별 상세
- `docs/HEAD_CLASSES.md` — 5-head 클래스 정의
- `docs/EXPERIMENT_SUMMARY.md` — **이 문서 (전체 요약)**
