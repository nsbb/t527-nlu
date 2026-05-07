# T527 NPU 정확도 회귀 분석 (2026-05-07)

## 결론

**현재 NPU ensemble: fn 57.6%, combo 34.3%** — ONNX 88.9% 대비 큰 격차.
원인: **uint8 + kl_divergence + 100 calib** 양자화가 모델 표현력 손상.

## 골든셋 99 정량 비교

| 단계 | fn 정확도 | combo |
|---|---|---|
| ONNX ensemble (CPU full) | **97.0%** | **88.9%** |
| ONNX cnn_body v28 (no quant) | 92.9% | — |
| ONNX cnn_body v46 (no quant) | 97.0% | — |
| ONNX cnn_body logit avg (no quant) | 96.0% | — |
| **NPU ensemble (uint8)** | **57.6%** | **34.3%** |

→ **양자화 단계에서만 약 40%p 손실.** subgraph 추출/입력 양자화는 무손실 (cnn_body ONNX vs full ONNX 결과 동일).

## NPU vs ONNX agreement (양자화 손실 측정)

| 헤드 | NPU=ONNX 일치 |
|---|---|
| fn   | 57.6% |
| exec | 80.8% |
| dir  | 54.5% |

→ 양자화로 logit 분포가 대폭 변경됨.

## 오답 패턴

heat_control이 죄다 unknown으로 무너짐:
```
'아이방 온돌 켜'        GT=heat_control → NPU=unknown (ONNX=heat_control)
'에어컨 이제 안 써도 될 것 같아' GT=ac_control → NPU=unknown (ONNX=ac_control)
'에어컨 켰는데도 더워'   GT=ac_control → NPU=unknown
'아이방 난방 꺼줘'       GT=heat_control → NPU=unknown
'거실 난방 켜'          GT=heat_control → NPU=unknown
```

특히 **간접/은유 표현**이 무너짐. unknown class에 신호가 강하게 양자화된 것으로 보임.

## 추정 원인

1. **calib 데이터 부족/편향** — 100개 calib npy의 출처/분포 불명
   (`checkpoints/cnn_body_acuity/calib/`는 어떤 텍스트의 임베딩인지 미문서화)
2. **kl_divergence 알고리즘** — 동적 범위 큰 모델에선 moving_average가 더 나음 (KoCitrinet 성공 사례)
3. **uint8** — int16/pcq 시도 안 함

## 다음 단계 — 양자화 재시도

1. **calib 데이터 재구성**
   - 학습 데이터(`data/`) + 골든셋 + 르엘 시나리오 → 500개 다양한 발화에서 임베딩 추출
   - NVIDIA 공식 권장: calib = 학습 분포에서 추출
2. **알고리즘 비교** (실험 매트릭스)
   ```
   [kl_divergence | moving_average | quantized_aware]
   × [uint8 | int16 | pcq]
   ```
3. **각 조합으로 재변환 후 골든셋 99 평가**

## 측정 인프라 (이번에 만들어진 것)

- `scripts/eval_npu_vs_onnx.py` — 골든셋 → uint8 .bin + ONNX 레퍼런스 logits 일괄 생성
- `t527_smart_v2/.../NpuEvalActivity.kt` — 디바이스에서 batch NPU + GT/ONNX 비교
- `/data/local/tmp/npu_eval_inputs.bin` — 99개 × 24576 bytes
- `/data/local/tmp/npu_eval_refs.json` — GT + ONNX argmax/logits

## Latency (참고)

- NPU ensemble: **1.97ms/sample** (두 NB 시퀀셜)
- Total 99 샘플: 195ms

→ 정확도만 회복하면 NPU 가속 효과는 충분히 큼.
