# T527 온디바이스 NLU

르엘 어퍼하우스 AI 월패드(T527 NPU)용 한국어 자연어 이해 시스템. STT 텍스트를 구조화된 동작(fn / exec / dir / param / room / value)으로 변환하여 AIDL · REST · 외부 API로 라우팅한다.

## 성능

기준 평가셋 — **르엘 GT 219개** (`data/golden/gt_known_scenarios_v2.json` 204 + `gt_unknown_scenarios.json` 15).

| 환경 | fn | exec | dir | combo | latency |
|---|:---:|:---:|:---:|:---:|:---:|
| 서버 ONNX raw (룰 미적용) | 96.3% | 98.6% | 96.3% | 93.2% | 0.32 ms |
| 서버 ONNX + Python `apply_post_rules` | 95.0% | 98.2% | 95.9% | **91.3%** | 0.32 ms |
| 디바이스 NPU (v72 NB int16) + Kotlin PostRulesV4 | 93.6% | 95.9% | 94.5% | **89.0%** | 11.7 ms |

확장 평가셋 — **3,043개** (르엘 시나리오 + 비유 · STT 오류 · 오탈자 변형, `data/golden/test_suite.json`).

| 환경 | fn | combo |
|---|:---:|:---:|
| 서버 ONNX + Python `apply_post_rules` | 99.0% | **94.0%** |
| 디바이스 NPU (v72 NB) + Python `apply_post_rules` | 98.9% | **91.6%** |

> 디바이스 후처리 룰의 Kotlin 포팅이 진행 중이라 GT 219(Kotlin)와 TS 3043(Python) 측정 시 적용된 룰이 다르다.

## 아키텍처

```
사용자 발화
  ↓ STT (Conformer NB)
  ↓ preprocess.py (STT 정규화 296개 + 한글 숫자 변환)
  ↓ WordPieceTokenizer (ko-sbert, max_len=32)
  ↓ NLU 추론
      서버:    nlu_v28_v72_ensemble.onnx (CNN 5-Head, head별 모델 선택)
      디바이스: cnn_body_v72_int16.nb (NPU) + token_emb_v46.bin (native lookup)
  ↓ 5-Head logits
  ↓ 후처리 룰 (PostRulesV4)
  ↓ slot 추출 (room 키워드 / value 정규식)
  ↓ DST (10초 timeout, 멀티턴 follow-up / correction)
  ↓ HomeState (HVAC 재해석, 상호배제)
  ↓ ResponseGenerator (한국어 응답, 받침, 자리표시자 치환)
  ↓ IntentRouter
      ├── AIDL → 월패드 remote_access_* (27 enum)
      ├── REST → 단지 서버
      ├── EXTERNAL_API → Open-Meteo · RSS · 교통 · 의료
      └── SELF → FAQ / 매뉴얼
```

### 5-Head 출력

| Head | 클래스 | AIDL 매핑 |
|---|:---:|---|
| fn | 20 | `command` (remote_access_*) |
| exec_type | 5 | `action` (status / control) |
| param_direction | 9 | `ctrl_action` (on/off/up/down/set/open/close/stop/none) |
| param_type | 5 | temperature / brightness / mode / speed |
| judge | 5 | outdoor_activity / clothing / air_quality / cost_trend |

## 빠른 시작

### 서버

```bash
# 대화형 (날씨 · 뉴스 API 포함)
python3 test_interactive.py

# Streamlit 웹 UI (멀티턴 · DST · HomeState)
streamlit run scripts/nlu_test_app.py

# 회귀 체크 (~0.4s)
python3 scripts/ci_quick_check.py

# 평가
python3 scripts/eval_v2_ruel_scenarios.py    # 르엘 219
python3 scripts/run_test_suite.py            # 확장 3,043
```

### 디바이스 (T527 NPU)

```bash
# 입력 셋 생성 → 디바이스 push
python3 scripts/eval_npu_vs_onnx.py /tmp/refs.json
adb push /tmp/refs.json /data/local/tmp/npu_eval_refs.json

# 평가 Activity 실행
adb shell am start -n com.t527.smart_v2/com.t527.smart_service.NpuEvalSingleActivity \
    --es nb_path /data/local/tmp/cnn_body_v72_int16.nb \
    --es out_name result.tsv
```

## 디렉토리 구조

```
checkpoints/
  nlu_v28_v72_ensemble.onnx       서버 production (110MB)
  cnn_multihead_v72.pt            PyTorch 원본
  cnn_body_v72_int16.nb           NPU production (2.74MB, int16 fl=15)

scripts/
  deployment_pipeline_v2.py       end-to-end 서버 파이프라인
  ensemble_inference_with_rules.py  ONNX 추론 + 후처리 룰
  dialogue_state_tracker.py       DST
  response_generator_v2.py        한국어 응답 생성
  preprocess.py                   STT 전처리
  ci_quick_check.py               pre-commit 회귀 체크

data/
  golden/                         평가 셋 (GT 219, TS 3043 등)
  train/                          학습 데이터 (train_final_v72.json, 32,809)
  raw/                            회사 원본 (르엘 csv, AIDL xlsx)
  augment/                        증강 소스
  ci/                             CI baseline

knowledge/
  index.md                        wiki entry point
  wiki/                           유지되는 현재 진실
  raw/                            소스 노트 (측정 · 실험 기록)
  schema/                         운영 규칙

docs/
  HANDOFF_2026_04_28.md           v72 종합 핸드오프
  VERSION_LOG.md                  실험 히스토리
  CHANGELOG.md                    룰 revision 기록
  AIDL_RUEL_GAP_REPORT.md         AIDL 명세 gap 분석
  RTM_REQUIREMENTS_TRACEABILITY.md  르엘 시나리오 ↔ NLU ↔ AIDL 매핑
  INDUSTRY_PRACTICES_RESEARCH.md  학계 · 산업 표준 조사
  T527_NPU_FULL_JOURNEY.md        NPU NB 변환 + 통합 기록
```

## 모델 버전

| 버전 | 날짜 | 특징 | 위치 |
|---|---|---|---|
| v28 | 2026-04-19 | GT 직접 시나리오 학습, exec head 안정 | ensemble 일부 |
| v72 | 2026-04-28 | 비유 3,961개 + 간접 표현 강화, 최종 학습 모델 | ensemble · NPU NB |
| v73~v78 | 2026-04-28 | 후처리 룰 · DST 코드 revision (모델 아님) | 서버 |
| v100~v133 | 2026-04-26 ~ | Kotlin 디바이스 룰 revision | 디바이스 |

Ensemble 전략: `fn=v72, exec=v28, dir=v72, param=v28, judge=v72`. 자세히는 `knowledge/wiki/models/version-matrix.md`.

## Android 통합

| 프로젝트 폴더 | applicationId | 역할 |
|---|---|---|
| `t527_vad_service` | `com.hdclabs.ondevice.voice` | 통합 production (음성 캡처 + STT + NLU 풀 스택). Kotlin nlu 패키지 `com.hdclabs.ondevice.voice.nlu` |
| `t527_ondevicevoice_service` | `com.hdclabs.ondevice.voice` | 동일 패키지명의 이전 빌드 (음성 캡처 + STT만, NLU 없음, Java) |
| `t527_smart_v2` | `com.t527.smart_v2` | NLU 측정 · 테스트 전용 Activity |
| `t527_ondevice_ai_agent` (WSL) | `com.t527.wallpad_agent` | AI 홈 에이전트 시연대 |

> `t527_vad_service`와 `t527_ondevicevoice_service`는 동일 applicationId(`com.hdclabs.ondevice.voice`)로 빌드되므로 디바이스에는 마지막 install된 빌드 하나만 존재한다.

## 평가 셋

| 파일 | 개수 | 용도 |
|---|:---:|---|
| `data/raw/ruel_scenarios_final.csv` | 219 | 회사 원본 르엘 시나리오 (사용자 발화 + intent + 기대 응답) |
| `data/golden/gt_known_scenarios_v2.json` | 204 | 위 원본에 사람 검증 multi-head 라벨링 (known intent) |
| `data/golden/gt_unknown_scenarios.json` | 15 | 위 원본에 사람 검증 multi-head 라벨링 (unknown class) |
| `data/golden/test_suite.json` | 3,043 | 219 시나리오 + 비유 · STT · 오탈자 변형 (회귀 체크용) |
| `data/golden/gt_seeds_integrated.json` | — | 학습 데이터 시드 |

성능 측정은 **회사 원본 219개에 사람이 multi-head 라벨링한 `gt_known_scenarios_v2.json` + `gt_unknown_scenarios.json` 219개**를 단일 기준으로 사용한다.

## 회귀 체크

NLU 관련 파일 commit 시 `.git/hooks/pre-commit`이 자동으로 `ci_quick_check.py` 실행. baseline 대비 fn / combo가 -2%p 이상 떨어지면 commit 차단.

```bash
python3 scripts/ci_quick_check.py           # 현재 baseline 대비 체크
python3 scripts/ci_quick_check.py --update  # 의도된 개선 후 baseline 갱신
```

baseline: `data/ci/ci_baseline_quick.json`.

## 개발 로드맵

1. PostRulesV4 풀 포팅 (Python → Kotlin) — 디바이스 NPU 정확도 개선
2. DialogueStateTracker 풀 포팅 — 멀티턴 안정성
3. ResponseGenerator 포팅 — 한국어 응답 품질
4. API Retrofit 통합 — 날씨 · 뉴스 · 교통
5. Joint Intent + Slot Filling (BIO tagging) 도입 — 자유 텍스트 슬롯 학습 기반 추출

## 참고 문서

- 산업 표준 · 학계 비교: `docs/INDUSTRY_PRACTICES_RESEARCH.md`
- AIDL 명세 gap: `docs/AIDL_RUEL_GAP_REPORT.md`
- 시나리오 ↔ NLU ↔ AIDL 매핑: `docs/RTM_REQUIREMENTS_TRACEABILITY.md`
- NPU 통합: `docs/T527_NPU_FULL_JOURNEY.md`
- 모델 한계: `docs/MODEL_LIMITATIONS.md`, `docs/KNOWN_FAILURES.md`

## 라이선스

내부 사용 (HDC Labs).

## 팀

HDC Labs AI Team — T527 NLU.
