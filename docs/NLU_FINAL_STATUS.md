# NLU 최종 상태 — CNN 5-Head v28

## 최종 모델: `checkpoints/cnn_multihead_v28.pt`

| 지표 | 값 |
|------|:---:|
| **Test Suite 2,000개 combo** | **100% (2000/2000)** |
| Test Suite fn | 100% |
| Test Suite exec | 100% |
| Test Suite direction | 100% |
| 219 GT fn | 99.1% (217/219) |
| Val combo | 94.7% |
| False rejection | 0건 |
| ONNX CPU 추론 | 0.4ms |

## 아키텍처

```
ko-sbert 768d frozen → Linear 768→256 → CNN 4L (k=3,5,7,3 residual) → Global Mean Pool
  → fn_head (20: 19 known + unknown)
  → exec_head (5: control/query/judge/direct/clarify)
  → direction_head (9: on/off/up/down/open/close/set/stop/none)
  → param_head (5: none/temperature/brightness/mode/speed) + 규칙 보정
  → judge_head (5: none/outdoor/clothing/air_quality/cost)
  + Rule-based: room (키워드), value (regex), 미지원 액션 (키워드), param_type 보정
```

- Total: 26.1M params (1.5M trainable)
- 학습 데이터: 20,815개
- 시나리오 증강 + CNN 보충 + HA 번역 + MASSIVE OOD + 누적 fix 778개

## 버전 히스토리 요약

| 버전 | Test Suite | 주요 개선 |
|------|:---:|------|
| v10 (시작) | 82.3% (96개) | HA 번역 + light_off |
| v14 | 100% (78개) | 간접/STT/존댓말/경계 전수 수정 |
| v19 | - | 구어체 (조용히/들어왔어/자려고/히팅) |
| v24 | 98.2% (114개) | direction 100%, 내려/줄여/커줘 수정 |
| v25 | 99.1% (114개) | 공기탁해/오늘어때 수정 |
| **v28** | **100% (2,000개)** | 띄어쓰기/감탄형/전처리/Room조합 |

## 테스트 방법

```bash
# 대화형 테스트
python3 scripts/test_interactive.py

# 자동 Test Suite (135개)
python3 scripts/run_test_suite.py

# 단일 발화
python3 scripts/test_interactive.py "거실 에어컨 23도로 맞춰줘"
```

## 파일 목록

| 파일 | 설명 |
|------|------|
| `scripts/model_cnn_multihead.py` | 모델 정의 |
| `scripts/sap_inference_v2.py` | 추론 파이프라인 |
| `scripts/test_interactive.py` | 대화형 테스트 |
| `scripts/run_test_suite.py` | 자동 Test Suite |
| `data/test_suite.json` | 2,000개 테스트 케이스 |
| `scripts/preprocess.py` | STT 전처리 (54개 교정 사전 + 한글숫자 변환) |
| `data/accumulated_fixes.json` | 누적 fix 패턴 778개 |
| `checkpoints/cnn_multihead_v28.pt` | **최종 모델** |
