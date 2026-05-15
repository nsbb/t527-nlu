# Source Note — NPU v72 NB 진짜 GT 219 측정 (2026-05-15)

## Source

- 디바이스: T527 데브킷 `51475789d0c64881cd3`
- NB: `/data/local/tmp/cnn_body_v72_int16.nb` (2.74MB, int16 fl=15)
- Activity: `NpuClassifierFullEvalActivity` (25 batch + System.gc() + 120ms sleep)
- 후처리: PostRulesV4
- 임베딩: `token_emb_v46.bin` (94MB, ko-sbert frozen 재사용)
- GT 입력: `/data/local/tmp/npu_eval_refs.json` (gt_known_v2 204 + gt_unknown 15 = **219**)
- 입력 생성: `scripts/eval_npu_vs_onnx.py /tmp/ruel_219_flat.json`
  - ONNX ref = v28+v46 ensemble (`nlu_v28_v46_ensemble.onnx`)

## 측정 결과

| 헤드 | hit/n | % | ONNX baseline |
|---|---|---|---|
| fn   | 205/219 | **93.6%** | 95.4% (Δ -1.8) |
| exec | 210/219 | 95.9% | 98.6% (Δ -2.7) |
| dir  | 207/219 | 94.5% | 98.2% (Δ -3.7) |
| **combo** | **195/219** | **89.0%** | **93.2% (Δ -4.2)** |

ONNX agreement: fn 96.3%, combo 91.3%
Latency: 평균 **11.72ms** / 총 2566ms

## fn 오답 (8건)

- '외출 후 복귀할 때 일괄소등 해제 및 환기시스템 켜 줘' GT=security_mode → NPU=light_control (ONNX=security_mode)
- '볼륨 조절할 수 있어?' GT=home_info → NPU=unknown (ONNX=home_info)
- '볼륨 조절은 어떻게 조절할 수 있어?' GT=home_info → NPU=unknown
- '월패드 볼륨을 최대로 키워줘' GT=home_info → NPU=unknown
- '(시간대별 자동볼륨 시나리오 반영) 월패드 볼륨을 최대로 키워줘' GT=home_info → NPU=unknown
- 'OO (특정 지역명칭 일부만) 날씨는?' GT=weather_query → NPU=unknown (ONNX=unknown, 둘 다 틀림)
- '공기 안 좋아?' GT=weather_query → NPU=vent_control (ONNX=weather_query)
- '거실 조명이 밝을 때 월패드 화면이 어두워졌으면 좋겠어' GT=unknown → NPU=system_meta (ONNX=system_meta, 둘 다 틀림)

## 핵심 결론

1. **이전 NPU 르엘 219 측정값 60.7% 는 자동매핑 GT (`golden_ruel_219.json`, 라벨 38% 오류, 삭제됨) 기준이었음** — 진짜 GT로 측정하니 **89.0%** 로 폭증
2. 서버 ONNX v72 ensemble (93.2%) 대비 -4.2%p — int16 양자화 손실 + 단일 NB (v28 exec/param head 도움 못 받음) 합산 영향
3. fn 오답 8건 중 **5건이 home_info/system_meta 모호 경계** (볼륨/화면 밝기) — production v72 학습 데이터에서도 약점
4. ART JIT crash 회피 (25 batch + GC) 219셋 성공. 491셋은 여전히 crash 가능성

## 다음 단계

- v28 cnn_body int16 NB 변환 + JNI head별 선택 → production v72 ensemble 재현 → 93%대 가능성
- system_meta/home_info 경계 학습 데이터 보강 (v73)
- 491셋 평가 위해 embedding native mmap (heap 압박 회피)

## Linked wiki

- `wiki/projects/t527-npu-integration.md`
- `wiki/models/version-matrix.md`
- `wiki/decisions/ruel-gt-set.md`
