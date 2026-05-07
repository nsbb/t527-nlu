# T527 NPU 돌파구: v46 int16 + PostRules = 100% fn / 89.9% combo

작성: 2026-05-07

## 결론

**v46 int16 NB 단독 + PostRulesV4 = 골든셋 99 fn 100%, combo 89.9%** — ONNX CPU 88.9% 능가.

이전 uint8 양자화 손실 -42%p가 int16으로 완전 회복.

## 골든셋 99 최종 비교

| 모드 | fn | combo | 비고 |
|---|---|---|---|
| v28-uint8 raw | 71.7% | 51.5% | |
| v28-uint8 + rule | 78.8% | 57.6% | |
| v46-uint8 raw | 54.5% | 31.3% | dynamic range 넓어 양자화 손실 큼 |
| v46-uint8 + rule | 59.6% | 37.4% | |
| **v46-int16 raw** | **97.0%** | **80.8%** | ONNX 97% 동등 — 양자화 손실 거의 없음 |
| **v46-int16 + rule** ⭐ | **100.0%** | **89.9%** | 최강 |
| ens (v28u8 + v46i16) raw | 96.0% | 68.7% | v28이 끌어내림 |
| ens (v28u8 + v46i16) + rule | 100.0% | 75.8% | v46 단독보다 떨어짐 |
| (참고) ONNX ensemble CPU + rule | ~97% | 88.9% | |

## 핵심 결정

**v46 int16 NB 단독 사용. ensemble 불필요.**

이유:
- v46 단독 + rule이 ensemble보다 +14%p combo 정확
- v28이 uint8 양자화 한계로 평균 끌어내림
- v46 int16 raw 97%가 이미 ONNX 수준

## 각 단계 정량

### v46 int16 변환
- 양자화: `dynamic_fixed_point` + `int16` + `kl_divergence` + 300 diverse calib
- NB 크기: 2.7MB (uint8 1.35MB의 2배)
- 입력: int16 fixed point (fl=15, 49152 bytes/sample = uint8의 2배)

### PostRules 효과
- v46 int16: 97 → 100% fn (+3%p), 80.8 → 89.9% combo (+9.1%p)
- 룰이 dir/exec 헤드의 잔여 오류를 보정

## Latency

- v46 int16 단독: **~1.5ms/sample** (추정)
- 2-NB ensemble: 3.01ms/sample
- ONNX CPU full ensemble: 21~30ms/sample

→ v46 int16 단독이면 **NPU가 ONNX CPU 대비 14~20배 가속**

## 양자화 옵션 매트릭스 (실험 결과)

| dtype | algorithm | calib | v46 fn |
|---|---|---|---|
| uint8 | kl_divergence | 100 | 54.5% |
| uint8 | moving_average | 300 diverse | 54.5% |
| **int16** | **kl_divergence** | **300 diverse** | **97.0%** ⭐ |

v46처럼 dynamic range 넓은 모델은 **int16 필수**. uint8은 정밀도 8비트(+offset)로 -0.5~0.3 범위 표현 시 손실 큼.

## 파일

- NB: `checkpoints/cnn_body_acuity/wksp_int16_nbg_unify/network_binary.nb` (2.7MB)
- 디바이스: `/data/local/tmp/cnn_body_v46_int16.nb`
- nbg_meta: input qtype=i16, fl=15

## 다음 작업

1. **golden_test_500 (491 샘플) 평가** — 작은 셋 100% 정확도가 큰 셋에서도 유지되는지
2. **르엘 219 GT 평가** — production 시나리오
3. **JNI를 t527_smart_v2 IntentClassifier에 정식 통합** — ONNX CPU 대신 NPU 호출
4. **token_emb lookup 디바이스 구현** — 현재는 .bin 사전 생성 사용 중 (실서비스용 아님)
