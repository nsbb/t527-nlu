# T527 디바이스 NLU 벤치마크 결과

작성: 2026-05-07

## 환경

- **디바이스**: T527 데브킷 (51475789d0c64881cd3, ARM64)
- **모델**: nlu_v28_v46_ensemble.onnx (105MB, KoELECTRA + CNN 5-head)
- **런타임**: ONNX Runtime Android 1.17.1 (CPU)
- **빌드**: t527_smart_v2 / NluBenchmarkActivity (Kotlin)
- **모델 위치**: /data/data/com.t527.smart_v2/files/ (run-as 복사)

## 결과

| 지표 | 1차 (raw 모델) | 2차 (+ PostRules 25개) |
|------|---------|---------|
| 정확도 | **10/15 (66%)** | **15/15 (100%)** |
| 단일 추론 평균 | 17ms | 20ms |
| 단일 추론 max | 22ms | 30ms |
| 100회 반복 평균 | 17ms | 20ms |
| 초기화 시간 | 952ms | 1017ms |

## 의미

- **모델 자체로는 66%** (서버 raw와 일치, 르엘 패턴은 잘 잡지만 비유/완곡/조사 등 약함)
- **132개 후처리 규칙 중 25개만 포팅해도 100%** (벤치마크 범위 한정)
- **추론 속도**: 서버 CPU 0.64ms vs T527 20ms ≈ 31배 차이 — 임베디드 ARM CPU 정상 범위
- **실시간 응답**: STT(200ms) + NLU(20ms) + AIDL(100ms) ≈ 320ms 응답 가능

## 다음 작업

1. **PostRules 132개 전체 포팅** (현재 25개 → 132개)
2. **NB 변환 시도** (CNN body만 NPU, 임베딩은 CPU lookup)
   - 예상: 20ms → 3ms (학계 Pure CNN 4L 사례 1ms)
3. **t527_smart_v2 본 서비스에 IntentClassifierV46 통합** (STT → NLU → AIDL chain)

## 3차 벤치마크 — 자동 포팅 87 + 수동 25 = 112개 규칙

| 지표 | 1차 raw | 2차 수동 25 | 3차 자동 112 |
|------|---------|-----------|------------|
| 정확도 | 10/15 (66%) | 15/15 (100%) | **14/15 (93%)** ⚠ |
| 단일 평균 | 17ms | 20ms | 33ms |
| 100회 평균 | 17ms | 20ms | 28ms |
| 초기화 | 952ms | 1017ms | 1028ms |

**회귀 발견:** "난방 23도로 해줘" → heat_control/on (기대: set)
- 자동 포팅된 87개 중 일부가 dir=set 케이스를 on으로 덮어씀
- 정교한 조건이 필요한 규칙은 자동 변환 부적합

## 자동 포팅 도구 한계

- 단순 if-pattern-action 패턴: 자동 변환 OK (87개)
- 복잡 패턴 (m=re.search, group(), 변수 사용): 수동 처리 필요 (121개)
- set/value 관련 규칙: 자동 변환 후 검증 필수

## 다음 단계

1. 자동 포팅 87개 중 회귀 유발 규칙 격리 (set 처리 보완)
2. 복잡 121개 중 고임팩트 규칙만 수동 포팅
3. 골든 데이터셋 (test_suite 219 시나리오 변형)으로 자동 검증

## 4차 벤치마크 — 안전 자동 36 + 수동 25 = 61개 규칙 (V3)

| 지표 | V1 raw | V2 (87+25=112) | **V3 (36+25=61, 안전)** |
|------|---------|---------|---------|
| 정확도 | 10/15 (66%) | 14/15 (93%) | **15/15 (100%)** |
| 단일 평균 | 17ms | 33ms | 25ms |
| 100회 평균 | 17ms | 28ms | 25ms |

**V2 회귀 원인:** "해줘" 등 일반 키워드를 단독 매칭으로 무조건 dir=on으로 설정하는 규칙이 set/value 케이스를 덮어씀.

**V3 해결:** 자동 포팅 도구를 **엄격 모드**로 정밀화 — fn/dir 가드(`if preds['fn'] in (...)`)가 있는 규칙만 자동 변환 대상.

## 자동 포팅 도구 v2 한계

```
208개 if 블록 분석:
  - 안전 자동 변환 (fn/dir 가드 有): 36개 (17%)
  - 복잡 (변수, 그룹 캡처, 다중 분기): 139개 (67%)
  - fn 조건 없는 단순 액션: 4개 (skip)
  - exec_type만 변경: ~29개 (skip)
```

복잡한 규칙은 다음 방식으로 보강 권장:
1. 자주 firing되는 규칙 측정 → 상위 N개만 수동 포팅
2. JSON 데이터 형식으로 규칙 추출 + Kotlin 엔진 작성
3. 또는 Python apply_post_rules를 ONNX 그래프로 변환 (어려움)

## 종합 — 디바이스 NLU 상태

```
T527 디바이스 (51475789...):
  모델: nlu_v28_v46_ensemble.onnx (105MB)
  런타임: ONNX Runtime CPU
  Latency: 25ms / req → 40 req/sec
  정확도: 15/15 (100%) ← V3 PostRules 61개 적용
  
실시간 응답 예측:
  STT(200ms) + NLU(25ms) + AIDL(100ms) = 325ms
```

## 5차 — 골든셋 100개 대규모 디바이스 검증

기존 15개 → **99개 골든셋**으로 정확한 측정.

### 디바이스 (T527 + 61 규칙 V3)
```
combo: 91 / 99 = 91.9%
latency_avg: 18ms (1차 25ms 측정시 warmup 직후 → 캐시 안정 후 18ms)
```

### 서버 (Xeon + 132 규칙)
```
fn:    99 / 99 = 100.0%
exec:  97 / 99 = 98.0%
dir:   99 / 99 = 100.0%
combo: 97 / 99 = 98.0%
latency: 0.64ms (서버 CPU)
```

### 차이 분석 — 디바이스가 6건 못 잡는 이유

| 발화 | 서버 | 디바이스 | 원인 |
|------|------|---------|------|
| 간접등 켜줘 | clarify/on | clarify/none | dir 보정 규칙 (자동 포팅 못함, 변수 사용) |
| 오늘날씨 | weather/none | unknown/none | "오늘날씨" STT 변형 규칙 |
| 불좀켜 | clarify/none | direct_respond/none | exec_type 보정 |
| 안방 남방 올려줘 | heat/up | heat/on | "남방→난방" STT + 올려→up |
| 이거 동굴이야? | query_then_judge/on | control_then_confirm/on | exec 보정 |
| 안방 혹시 불 켜줘 | control_then_confirm/none | clarify/none | exec 보정 |

→ 모두 **자동 포팅 안 된 복잡 규칙** (변수 사용, group capture, 다중 분기). 수동 포팅 가능.

### 서버도 못 잡는 2건

```
'어르신이 덥다고 하시네요'  GT: ac/control_then_confirm/on  → ac/direct_respond/on  (exec 차이)
'등줄기가 서늘해'           GT: heat/control_then_confirm/on → heat/direct_respond/on (exec 차이)
```

→ 모델 자체의 한계 (직접/제어 exec 분류 모호). 별도 규칙 추가 필요.

## 종합 — 디바이스 NLU 결론

```
T527 NPU 데브킷 (51475789...) NLU 추론 성능:
  ├─ 모델: 105MB ONNX (CPU 추론)
  ├─ 정확도: 91.9% combo (99개 골든셋)
  │   └─ 부족분 6건은 자동 포팅 못 한 복잡 규칙 (수동 포팅 시 잡을 수 있음)
  ├─ Latency: 18~25ms / 추론
  └─ 실시간성: STT(200ms) + NLU(20ms) + AIDL(100ms) ≈ 320ms 응답 가능
```
