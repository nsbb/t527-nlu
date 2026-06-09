# T527 온디바이스 NLU — CNN 5-Head + v72 Ensemble + Kotlin/NPU 통합

르엘 어퍼하우스 AI 월패드 (T527 NPU) STT 텍스트 → Structured Action → 기기 제어.

## 핵심 지표 (production, 2026-06)

| 지표 | 서버 ONNX | 디바이스 NPU (v72 NB int16) |
|------|:---:|:---:|
| **르엘 GT 219 (gt_known_v2 + gt_unknown, 진짜 GT)** | **combo 93.2% / fn 96.3%** | **combo 89.0% / fn 93.6%** |
| **TS 3043 (르엘 + 비유 + STT + 오탈자 변형)** | combo 94.0% / fn 99.0% | combo 91.6% / fn 98.9% |
| 추론 지연 | 0.32ms (CPU) | 11.7ms (NPU, native lookup) |
| 파일 크기 | 110MB (FP32 ONNX) | 2.74MB (NPU NB int16) + 94MB emb |

> ⚠️ **이전 README 값(TS 97.21%, GT 95.0%)은 iter9 (4/22) 시점 자체 측정**이고 그 후 GT 정의 재정립으로 폐기됨.  
> **진짜 GT는 사람 검증된 `data/golden/gt_known_v2.json + gt_unknown.json` 219개**. 자동매핑 GT(`golden_ruel_219.json`)는 라벨 38% 오류로 2026-05-15 폐기.

→ **서버 production**: `checkpoints/nlu_v28_v72_ensemble.onnx` (110MB)  
→ **디바이스 NPU NB**: `/data/local/tmp/cnn_body_v72_int16.nb` (2.74MB)  
→ **파이프라인**: `scripts/deployment_pipeline_v2.py` (서버) / `t527_vad_service` Android (디바이스)

## 아키텍처

```
사용자 발화 "거실 에어컨 23도로 맞춰줘"
    ↓ STT (Citrinet / Conformer NB)
"거실 에어컨 23도로 맞춰줘"
    ↓ preprocess.py (296 STT 교정 + 한글숫자 변환)
    ↓ 토크나이저 (ko-sbert WordPiece, max_len=32)
input_ids [1, 32]
    ↓ Ensemble ONNX v28+v72 (CNN 5-Head, head별 다른 모델)
       또는 NPU v72 int16 NB (cnn_body 단독, embedding native lookup)
5개 logits (fn/exec/dir/param/judge)
    ↓ argmax + param_type 규칙 보정 + confidence fallback
preds = {fn, exec_type, param_direction, param_type, judge}
    ↓ PostRulesV4 (서버 v78 / 디바이스 Kotlin 일부)
    ↓ Rule slots (room/value 키워드 추출 — 휴리스틱)
    ↓ DST (10초 timeout, 멀티턴 follow-up/correction)
    ↓ HomeState (HVAC 재해석, 상호배제)
resolved = {fn, exec, dir, room, value, judge}
    ↓ ResponseGenerator (한국어 응답, 받침, 자리표시자 치환)
"네, 거실 에어컨 온도를 23도로 설정합니다."
    ↓ IntentRouter → AIDL (월패드 27 enum) / REST_RUEL / EXTERNAL_API / SELF
    ↓ TTS
```

## 5-Head 구조 (AIDL 4축과 1:1 정합)

| Head | 클래스 수 | AIDL 매핑 |
|------|:---:|------|
| **fn** | 20 | `command: remote_access_*` (27 enum) |
| **exec_type** | 5 | `action: status / control` |
| **param_direction** | 9 | `ctrl_action: on/off/up/down/set/open/close/stop/none` |
| **param_type** | 5 | `temperature / brightness / mode / speed` |
| **judge** | 5 | (outdoor_activity / clothing / air_quality / cost_trend) |

→ Multi-head 채택 이유 및 학계 비교: `docs/INDUSTRY_PRACTICES_RESEARCH.md`. AIDL gap: `docs/AIDL_RUEL_GAP_REPORT.md`. RTM: `docs/RTM_REQUIREMENTS_TRACEABILITY.md`.

## 모델 라인업 (4종 버전 시리즈)

### A. 학습 모델 (`checkpoints/cnn_multihead_v*.pt`)

| 버전 | 날짜 | 강점 | GT-219 combo |
|---|---|---|---|
| v28 | 4/19 | TS 100%, GT 직접 시나리오 강함 | 92.7% (NPU raw 측정) |
| v34 | 4/19 | pseudo-labeling 돌파, KE 96.8% | — |
| v46 | 4/20 | mixup 일반화 KE 97.8% (단일 최고) | — |
| **v72** | **4/28** | **비유 3961개 + 간접 20/21** | **production** ⭐ |
| v73~v78 | 4/28 | (모델 아님 — 룰/DST 코드 revision) | — |

### B. Ensemble (head별 모델 선택)

| 조합 | 전략 | GT-219 combo (서버 ONNX + Python rule) |
|---|---|---|
| v28+v46 | (구) 디바이스 CPU 옛 배포 | — |
| **v28+v72** | fn=v72, exec=v28, dir=v72, param=v28, judge=v72 | **91.3%** ⭐ production |

### C. NPU NB (디바이스)

| NB | 양자화 | GT-219 combo (+ Kotlin rule) |
|---|---|---|
| cnn_body_v46.nb | uint8 | 54.5% (폐기) |
| cnn_body_v46_int16.nb | int16 fl=15 | (자동매핑 GT 60.7% — 진짜 GT 미측정) |
| **cnn_body_v72_int16.nb** | int16 fl=15 | **89.0%** ⭐ 현재 NPU 모드 default |

⚠️ NPU ensemble (head별 NB 합성) = **실패**. libVIPlite driver가 한 process에서 두 NB sequential init 안 받음. 단일 NB로 고정.

### D. 룰 revision

- **Python v78** (서버, `scripts/ensemble_inference_with_rules.py:apply_post_rules`) — 601 if-block
- **Kotlin PostRulesV4** (디바이스) — 9 if-block만 포팅됨 (1.5%). **NPU 89% 정체의 가장 큰 원인**

## 빠른 시작

### 서버 ONNX

```bash
# 대화형 테스트 (API 포함, 날씨/뉴스)
python3 test_interactive.py

# Streamlit 웹 UI (멀티턴 + DST + HomeState)
streamlit run scripts/nlu_test_app.py

# CI 회귀 체크 (~0.4s, baseline = 르엘 219)
python3 scripts/ci_quick_check.py

# Test Suite 3043 평가
python3 scripts/run_test_suite.py

# 르엘 GT 219 평가
python3 scripts/eval_v2_ruel_scenarios.py
```

### 디바이스 NPU 측정 (T527 데브킷 `51475789...`)

```bash
# 입력 셋 준비
python3 scripts/eval_npu_vs_onnx.py data/golden/gt_known_scenarios_v2.json /tmp/refs_219
adb push /tmp/refs_219/refs.json /data/local/tmp/npu_eval_refs.json

# Activity 실행 (NB path 인텐트로 받음)
adb shell am start -n com.t527.smart_v2/com.t527.smart_service.NpuEvalSingleActivity \
    --es nb_path /data/local/tmp/cnn_body_v72_int16.nb \
    --es out_name ens_v72.tsv

# 결과 pull + 합성
adb shell "run-as com.t527.smart_v2 cat files/ens_v72.tsv" > /tmp/ens_v72.tsv
```

## 핵심 파일

### 모델 & NB
- `checkpoints/nlu_v28_v72_ensemble.onnx` — **서버 production**
- `checkpoints/cnn_multihead_v72.pt` — PyTorch 원본 (4/28)
- 디바이스 `/data/local/tmp/cnn_body_v72_int16.nb` — **NPU production**
- 디바이스 `/data/data/<pkg>/files/token_emb_v46.bin` (94MB) — ko-sbert frozen, v28~v72 공통

### 스크립트
- `scripts/deployment_pipeline_v2.py` — end-to-end 서버 파이프라인
- `scripts/ensemble_inference_with_rules.py` — ONNX + 601 if rule
- `scripts/dialogue_state_tracker.py` — DST (521줄)
- `scripts/response_generator_v2.py` — 한국어 응답 (2418줄)
- `scripts/preprocess.py` — STT 전처리 (296 매핑)
- `scripts/ci_quick_check.py` / `ci_regression_check.py` — 회귀 체크
- `scripts/eval_npu_vs_onnx.py` — NPU 입력 .bin + ONNX refs 생성

### 데이터
- `data/golden/gt_known_scenarios_v2.json` (204) + `gt_unknown_scenarios.json` (15) — **진짜 production GT 219**
- `data/golden/test_suite.json` (3043) — 르엘 + 비유 + STT + 오탈자 변형
- `data/golden/gt_seeds_integrated.json` — 학습 데이터 시드
- `data/golden/test_ruel.csv` — 회사 원본 csv (응답 평가)
- `data/train/train_final_v72.json` (32,809) — v72 학습 데이터

### 문서 (단일 진실)
- **`knowledge/index.md`** — knowledge wiki entry point (최신 사실 정리)
- `knowledge/wiki/models/version-matrix.md` — 4종 버전 매트릭스 (학습 모델 / Ensemble / Rule / NPU NB)
- `knowledge/wiki/projects/t527-npu-integration.md` — NPU 통합 상태
- `knowledge/wiki/issues/server-device-version-gap.md` — 위치별 모델 gap
- `knowledge/wiki/decisions/ruel-gt-set.md` — 르엘 GT 219 정의
- `knowledge/raw/source-notes/` — 측정/실험 raw 로그
- `docs/HANDOFF_2026_04_28.md` — v72 핸드오프 (옛 상세 문서)
- `docs/VERSION_LOG.md` — v1~v72 실험 히스토리
- `docs/CHANGELOG.md` — 룰 r1~r1990 변경 기록
- `docs/AIDL_RUEL_GAP_REPORT.md` — AIDL 명세 gap + 보강 7건
- `docs/RTM_REQUIREMENTS_TRACEABILITY.md` — 219 시나리오 ↔ NLU ↔ AIDL/REST 매핑
- `docs/INDUSTRY_PRACTICES_RESEARCH.md` — 학계/Google/Alexa 비교 (755줄)
- `docs/T527_NPU_FULL_JOURNEY.md` — NPU NB 변환 + 통합 여정

## 진행 히스토리 (요약)

```
4/13~21  v1~v68    초기 학습 + Test Suite 3043 구축 + 앙상블 v28+v46
4/21~22  iter9     rule + DST 도입, TS 95.76%
4/28     v70       대규모 라벨 수정 → -4%p regression (폐기)
4/28     v71       어두침침 + paraphrase, GT-219 93.6%
4/28     v72       비유 3961개 + 간접 20/21 ⭐ 마지막 학습 모델
4/28     v73~v78   룰/DST 코드 revision (모델 아님)
5/7      v46 NB    NPU int16 변환 + native lookup (Kotlin 통합)
5/13     wiki      knowledge/ 도입 (팀 wiki schema)
5/15     v72 NB    cnn_body v72 NPU 변환 + 진짜 GT 219 측정 (combo 89.0%)
5/15     data정리  자동 생성 시험용 셋 10개 폐기, GT 219 단일화
5/18     TS 3043   NPU + ensemble 시도 (실패, driver 한계)
6/현재   다음작업  PostRulesV4 풀 포팅 (601 if → Kotlin), 디바이스 NPU 94% 목표
```

## 다음 작업 (우선순위)

1. **PostRulesV4 풀 포팅** — Python 601 if → Kotlin `t527_vad_service/.../PostRulesV4.kt`. NPU 91.6 → ~94% 목표.
2. DST 풀 포팅 — `dialogue_state_tracker.py` 521줄 → Kotlin `DialogueStateTracker.kt` 확장.
3. ResponseGenerator 포팅 — 한국어 응답 + 받침 + 자리표시자.
4. API Retrofit 통합 — Open-Meteo / RSS / placeholder fill.
5. v73 학습 (선택) — schedule_manage / volume / home_info↔system_meta 경계 보강.
6. 장기: BIO Slot Tagging (Joint Intent + Slot, JointBERT 패턴, CNN 백본 유지) — 95%+ 노릴 때.

## 디바이스 / 앱

| 앱 | applicationId | 역할 |
|---|---|---|
| `t527_smart_v2` | com.t527.smart_v2 | NLU 측정/테스트 (NpuEvalSingleActivity 등) |
| **`t527_vad_service`** | com.t527.vad_service | **통합 production** (NLU 풀 통합, PostRules/DST/HomeState) |
| `t527_ondevicevoice_service` | com.hdclabs.ondevice.voice | 디바이스 active, ALSA + Conformer STT + 단지서버 REST (NLU 없음) |
| `t527_ondevice_ai_agent` (WSL) | com.t527.wallpad_agent | AI 홈 에이전트 시연대 (KoElectra NLU + TTS + CitriNet) |

## 라이센스

내부 사용 (HDC Labs)

## 개발팀

HDC Labs AI Team — T527 NLU
