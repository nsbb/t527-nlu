# Source Note — 디바이스 NLU 파이프라인 포팅 인수인계 (2026-05-15)

> 다른 세션에서 디바이스(t527_vad_service)에 Python 파이프라인을 통합 중. 본 노트가 인수인계 단일 진실.

## 1. Production 모델 = v72

- 학습 모델: `cnn_multihead_v72.pt` (4/28, 마지막 학습)
- 서버 ONNX: `checkpoints/nlu_v28_v72_ensemble.onnx` (head별 fn=v72, exec=v28, dir=v72, param=v28, judge=v72)
- 디바이스 NPU NB: `/data/local/tmp/cnn_body_v72_int16.nb` (2.74MB, int16 fl=15, v72 단독)
- 임베딩: `/data/data/<pkg>/files/token_emb_v46.bin` (94MB, ko-sbert frozen, v28~v72 공통)

⚠️ `v73~v78`은 모델 아님 — Python 룰/DST 코드 revision (서버 v78까지 사용)

## 2. 측정값 (진짜 GT 219 + TS 3043, 2026-05-15 실측)

| 평가셋 | 서버 ONNX + Python rule | T527 NPU + Kotlin PostRulesV4 |
|---|---|---|
| 르엘 GT 219 (gt_known_v2 + gt_unknown) | combo **91.3%** / fn 95.0% | combo **89.0%** / fn 93.6% |
| TS 3043 (르엘 + 비유 + STT + 오탈자) | combo **94.0%** / fn 99.0% | combo **91.6%** / fn 98.9% |

NPU 양자화 손실 ~2.4%p + Kotlin 룰 부족 갭. Ensemble 시도 실패 (NPU driver `libVIPlite gcvip_patch_network_inputs` SEGV — 한 process에서 두 NB sequential init 안 받아줌).

⚠️ `golden_ruel_219.json` 자동매핑 GT는 폐기 (2026-05-15, 라벨 38% 오류). 진짜 GT = `data/golden/gt_known_scenarios_v2.json` (204) + `gt_unknown_scenarios.json` (15) = 219.

## 3. Android Studio 프로젝트 ↔ APK 패키지 매핑

| Android Studio 프로젝트 | namespace (Kotlin) | applicationId (APK) | 역할 |
|---|---|---|---|
| `t527_smart_v2` | com.t527.smart_service | com.t527.smart_v2 | NLU 측정/테스트 앱 (NpuEvalSingleActivity 등 측정 Activity 위치) |
| **`t527_vad_service`** | com.t527.awaiasr_2 | **com.t527.vad_service** | **통합 production 앱** (VAD+STT+NLU+AIDL 다 안에) — 포팅 작업 목표 |

두 프로젝트의 `nlu/PostRulesV4.kt` 내용 **동일** (diff empty). NLU 코드는 t527_vad_service가 com/t527/smart_service/nlu/ 폴더로 그대로 보유.

## 4. 옮겨야 할 Python → Kotlin (t527_vad_service)

| Python 파일 (서버) | 규모 | → Kotlin (디바이스, t527_vad_service) | 현재 포팅율 |
|---|---|---|---|
| `scripts/ensemble_inference_with_rules.py` 안 `apply_post_rules()` | **601 if-block / 1981줄** | `nlu/PostRulesV4.kt` 확장 | **9 if (1.5%)** ⚠️ 가장 큰 갭 |
| `scripts/dialogue_state_tracker.py` | 521줄 (slot fill / correction / 10s timeout) | `nlu/DialogueStateTracker.kt` 확장 | 136줄 (26%) |
| `scripts/response_generator_v2.py` | 2418줄 (한국어 응답 + 받침 + 자리표시자) | `integration/ResponseGenerator.kt` 확장 | 162줄 (7%) |
| `test_api.py` (Open-Meteo / RSS / placeholder fill) | ~300줄 | 새 Retrofit 모듈 | 0 |

이미 OK (안 옮겨도 됨):
- `nlu/Preprocess.kt` (296 STT 매핑) ✓
- `nlu/IntentClassifierV46.kt` (NPU 추론, native lookup) ✓
- `nlu/HomeState.kt` (HVAC 재해석) ✓
- `integration/IntentRouter.kt`, `AidlMockClient.kt` (AIDL 라우팅) ✓

## 5. 평가 인프라 (t527_smart_v2 안, t527_vad_service에 동기화 미완)

| Activity | 목적 |
|---|---|
| `NpuClassifierFullEvalActivity` | NPU v72 NB 219 자동 평가 (combo 89.0%) |
| `NpuEvalSingleActivity` | NB path + start_idx + count 인텐트로 받음 (분할 평가용) |
| `NpuEnsembleEvalActivity` | sequential ensemble 시도 (driver 한계로 실패) |

→ 통합 작업 시 NpuEvalSingleActivity 만 vad_service로 옮기면 충분 (TS 3043 91.6% 재현 가능).

## 6. 우선순위 (정확도 효과 큰 순)

1. **PostRulesV4 완전 포팅** (601 if → Kotlin) — NPU 91.6 → ~94% 가능 (서버 수준)
2. DST 풀 포팅 (slot fill, correction, multi-turn) — 멀티턴 안정
3. ResponseGenerator 풀 포팅 — 음성 응답 품질 (받침 처리 + 자리표시자)
4. API Retrofit (Open-Meteo + RSS + placeholder fill) — 날씨/뉴스 응답
5. VoiceAiService 통합 (STT → NLU → AIDL/REST/API 라우팅 → 응답)

## 7. 반복 금지 함정

- `golden_ruel_219.json` 자동매핑 GT vs `gt_known_v2` 진짜 GT 38% 차이 — 자동매핑 폐기 완료, 단일 진실 = 진짜 GT 219.
- NPU에서 두 NB sequential ensemble = `libVIPlite SEGV` → 단일 NB만 사용.
- `data/local/tmp/`는 앱 쓰기 권한 없음 → `filesDir` 사용.
- Kotlin rule이 Python rule의 1.5%만 포팅됨 = NPU 91.6% 정체 원인.
- 디바이스 NB 변환 절차: `wiki/projects/t527-npu-integration.md` + 메모리 axiom (Docker NeMo 23.06 + Acuity 6.12 + VivanteIDE 환경변수 + `VIP9000NANOSI_PLUS_PID0X10000016`).

## Linked

- `wiki/models/version-matrix.md`
- `wiki/projects/t527-npu-integration.md`
- `wiki/decisions/ruel-gt-set.md`
- `src-npu-real-gt-219-20260515.md` (NPU 219 측정)
- `src-v72-nb-conversion-20260515.md` (NB 변환)

## Last updated

- 2026-05-15
