# T527 NPU NB 변환 — 시도 결과 정리

작성: 2026-05-07

## 진전 요약

✅ **단계 1: CNN body 추출 (성공)**
- v46 PyTorch checkpoint (105MB) → CNN body only ONNX (5.91MB)
- 임베딩(token_emb 93.8MB) 분리 → CPU lookup
- 31 nodes, 30 initializers
- I/O: input `[1, 32, 768]` (embedded vector) → 5 logits

✅ **단계 2: Acuity import (성공)**
- `pegasus import onnx` → cnn_body_v46.json + cnn_body_v46.data
- Error 0, Warning 1
- Conv 4개 + Gemm 10개 + ReLU/Add/Mean 정상 변환

🟡 **단계 3: Acuity quantize (막힘 — inputmeta 호환성)**

5가지 시도 모두 실패:

| # | 시도 | 결과 |
|---|------|------|
| 1 | NPY 직접 type | "Cannot load file containing pickled data when allow_pickle=False" |
| 2 | BIN type | "Unsupport database type" |
| 3 | H5FS type | "TypeError: expected bytes, NoneType found" |
| 4 | TEXT + 3D shape | "Network doesn't have a valid input meta" |
| 5 | TEXT + 4D NCHW (ko_citrinet 모방) | "Network doesn't have a valid input meta" |

### 추정 원인

- Acuity 6.12의 NPY loader가 일부 numpy 헤더 버전을 pickle로 오인
- inputmeta.yml의 lid/shape이 acuity 내부 그래프와 미세 불일치
- non-image 입력(embedded vector)에 대한 inputmeta 표준 형식 정보 부족
- Acuity 6.12 docs에 비-이미지 input 예제가 거의 없음

## 차선책 — 현재 시스템 충분히 실용

| 측정 | 값 |
|------|-----|
| 디바이스 ONNX CPU | **21~30ms / 추론** |
| 정확도 (491 골든셋) | 93.7% |
| 실시간 응답 | STT 200ms + NLU 30ms + TTS = 약 330ms |

**NB 변환 없이도 실용 가능.** NB 성공 시 latency 21ms → 3ms (7배) 가속이지만 현재로도 사용자 체감 즉각 응답.

## NB 변환 추가 시도할 수 있는 방향

1. **Acuity 6.12 manual 비-이미지 input 예제 찾기**
2. **VeriSilicon/Acuity 팀 직접 문의** — non-image input meta 양식 가이드 요청
3. **GENERATOR 타입** Python yield 함수 작성
4. **CNN body 더 단순화** — 3D 입력을 4D 이미지처럼 변형 (e.g., [1, 32, 768] → [1, 1, 32, 768] 강제)
5. **ONNX 그래프에 Reshape 추가** — input을 acuity 친화적 4D로 변환하는 노드 추가

## 보존 산출물

```
checkpoints/cnn_body_v46.onnx                 (5.91MB) — CNN body
checkpoints/cnn_body_acuity/
  ├─ cnn_body_v46.json                        (15KB) — Acuity 그래프
  ├─ cnn_body_v46.data                        (6MB)  — weights
  ├─ inputmeta.yml                            — 시도 중인 형식
  ├─ dataset.txt                              — 100 calib 경로
  └─ calib/                                   — 100개 .bin/.npy
```

다음 iteration 또는 별도 작업으로 NB 변환 깊이 파고 가능.

## 최종 결론

**T527 디바이스에서 NLU 시스템 완성, NB 변환은 다음 단계.**
- ONNX CPU 추론 21~30ms로 실시간 응답 가능
- NB 변환은 acuity 호환성 디버깅이 더 필요한 작업
