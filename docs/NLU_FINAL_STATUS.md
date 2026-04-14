# NLU 최종 상태 — CNN 5-Head v7

## 최종 모델: `checkpoints/cnn_multihead_v7.pt`

| 지표 | 값 |
|------|:---:|
| **219 GT fn** | **100% (219/219)** |
| Val fn | 99.5% (known 99.7%, unknown 98.4%) |
| **Val combo (5-head 전부)** | **94.5%** |
| **Unknown 거부** | **95% (19/20)** |
| **False rejection** | **0건 (100%)** |

## 아키텍처

```
ko-sbert 768d frozen → Linear 768→256 → CNN 4L (k=3,5,7,3 residual) → Global Mean Pool
  → fn_head (20: 19 known + unknown)
  → exec_head (5: control/query/judge/direct/clarify)
  → direction_head (9: on/off/up/down/open/close/set/stop/none)
  → param_head (5: none/temperature/brightness/mode/speed)
  → judge_head (5: none/outdoor/clothing/air_quality/cost)
  + Rule-based slots: room (키워드), value (regex), 미지원 액션 (키워드)
```

- Total: 26.1M params (1.5M trainable)
- 학습 데이터: 24,351개 (known 19,533 + unknown 4,818)
- 시나리오당 평균 94개 유니크 변형

## 동작 예시

```
"찜통이야"          → fn=ac_control, exec=control, dir=on     → "네, 에어컨을 켰습니다"
"좀 춥다"           → fn=heat_control, exec=control, dir=on    → "네, 난방을 켰습니다"
"거실 에어컨 23도"   → fn=ac_control, dir=set, value=23°       → "네, 거실 에어컨 온도를 23도로 설정합니다"
"세차해도 돼?"       → fn=weather, exec=judge, judge=outdoor    → "기온과 대기질 양호. 외출 무리 없음"
"나 나간다"          → fn=security, exec=control, dir=on        → "네, 외출모드로 전환합니다"
"피자 주문해줘"      → fn=unknown                               → 서버 LLM으로 전송
"택시 불러줘"        → fn=traffic (미지원 액션)                   → "해당 기능은 지원하지 않습니다"
```

## 파일 목록

| 파일 | 설명 |
|------|------|
| `scripts/model_cnn_multihead.py` | 모델 정의 (20 fn classes) |
| `scripts/sap_inference_v2.py` | 대화형 추론 파이프라인 |
| `scripts/parse_gt_scenarios.py` | 219 GT → known/unknown 분리 |
| `scripts/augment_diverse.py` | 고다양성 증강 (시나리오당 ~94 unique) |
| `scripts/train_cnn_multihead.py` | 학습 스크립트 |
| `scripts/eval_ruel.py` | 평가 스크립트 |
| `checkpoints/cnn_multihead_v7.pt` | **최종 모델** |
| `data/gt_known_scenarios.json` | 204개 known GT 시나리오 |
| `data/gt_unknown_scenarios.json` | 15개 unknown GT 시나리오 |
| `data/train_gt_augmented_v2.json` | 고다양성 증강 데이터 |
| `data/ha_english_for_translation.json` | 번역 대기 영어 발화 1,100개 |

## 진행 히스토리

| 버전 | 데이터 | fn | combo | unknown | 비고 |
|------|------:|:---:|:-----:|:-------:|------|
| SAP v12 | 7,053 | 86% | 38.8% | 없음 | Conformer 8-head |
| CNN v1 | 13,904 | 96.5% | 90.8% | 없음 | CNN 전환 |
| CNN v4 | 10,342 | 96.3%* | 86.3% | 70% | unknown 추가 |
| CNN v6 | 10,342 | 97.3%* | 86.5% | 50% | hard negative |
| **CNN v7** | **24,351** | **100%*** | **94.5%** | **95%** | **고다양성 증강** |

*219개 원본 GT 기준

## 남은 과제

1. **HA 영어 번역**: `data/ha_english_for_translation.json` 1,100개 → 한국어 번역 → 학습 데이터 보강
2. **Edge case 보강**: "볼륨 올려줘", "안방 난방 25도", "우산 가져가야 해", "미세먼지 어때" 패턴 (v8에서 수정했으나 false rejection 발생)
3. **ONNX 변환**: CPU 추론용 ONNX export
4. **멀티턴 DST**: 대화 상태 추적 레이어 추가
5. **실제 STT 연동**: STT output → NLU → 응답 E2E 테스트
