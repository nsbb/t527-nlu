# Source Note — NPU TS 3043 측정 + Ensemble 시도 결과 (2026-05-18)

## Source

- 디바이스: T527 데브킷 `51475789d0c64881cd3`
- NB: `/data/local/tmp/cnn_body_v72_int16.nb` + `cnn_body_v28_int16.nb`
- Activity: `NpuEvalSingleActivity` (신규, NB path / start_idx / count 인텐트로 받음)
- 입력: `/data/local/tmp/npu_eval_refs.json` (test_suite.json 3043 + GT 219 둘 다 측정)

## TS 3043 측정 결과 — 통합 평가 (219 르엘 + 비유 + STT 변형)

500개씩 7 batch (첫 batch만 SIGSEGV 79개 후 죽음 → 79-499 missing 구간 100개씩 5 batch로 보충 → 3043 전부 채움).

### T527 NPU v72 NB + Python apply_post_rules
| 헤드 | 정답률 |
|---|---|
| fn   | **98.9%** (3009/3043) |
| exec | 95.6% |
| dir  | 94.8% |
| **combo** | **91.6%** (2787/3043) |
| Latency (per inference) | ~5.3ms |

### 서버 ONNX v72 ensemble + Python apply_post_rules (같은 셋)
| 헤드 | 정답률 |
|---|---|
| fn   | 99.0% |
| exec | 99.5% |
| dir  | 95.2% |
| combo | **94.0%** |

→ NPU vs ONNX gap: combo **-2.4%p** (양자화 손실 일관)

## 르엘 GT 219 측정 결과 (재확인)

| 모드 | combo | fn |
|---|---|---|
| 서버 ONNX raw (룰 미적용) | 93.2% | 96.3% |
| 서버 ONNX + Python rule | 91.3% | 95.0% |
| T527 NPU v72 NB + Kotlin PostRulesV4 | **89.0%** | **93.6%** |

⚠️ Python `apply_post_rules` 적용 시 ONNX raw 93.2 → 91.3로 떨어짐 (특정 케이스에서 rule이 over-correct). Kotlin PostRulesV4는 v133 일부만 포팅되어 더 보수적 → 89.0%로 더 낮음.

## Ensemble 시도 — head별 NB 합성 (v28+v72)

목표: production v72 ensemble 패턴 (fn=v72, exec=v28, dir=v72, param=v28, judge=v72)을 NPU에서 재현하여 93% 회복.

### 시도 1: 한 process에서 두 NB 동시 로드 (`NpuEnsembleEvalActivity` v1)
- 두 IntentClassifierV46 인스턴스 동시 init → **앱 crash** (50/219 진행 후)
- 추정: embedding 94MB × 2 + NB × 2 + heap 압박

### 시도 2: 한 process에서 sequential (Phase 1 v72 → release → Phase 2 v28)
- v72 Phase 1만 진행 중 75/219에서 SIGSEGV
- backtrace: `libVIPlite.so gcvip_patch_network_inputs+100` → NPU 드라이버 단의 SEGV
- 원인: VIPlite driver가 한 process 안에서 `awnn_destroy → awnn_create` 미지원 (global state 누수)

### 시도 3: 두 Activity로 process 분리 (`NpuEvalSingleActivity` v1, v2)
- v72 NB로 한 Activity 실행 → 결과 `ens_v72.tsv` 저장 → process 종료
- force-stop → v28 NB로 새 Activity 실행 → `ens_v28.tsv` 저장
- 호스트(Python `merge_ensemble.py`)에서 head별 합성 + `apply_post_rules`

### Ensemble 합성 결과 (르엘 GT 219)
| 모드 | fn | exec | dir | combo |
|---|---|---|---|---|
| v72 단독 raw | 95.9% | — | — | 90.9% |
| v72 단독 + Python rule | 94.5% | 95.9% | 95.4% | 89.5% |
| v28 단독 raw | 95.9% | — | — | **92.7%** |
| v28 단독 + Python rule | 93.6% | 95.4% | 94.5% | 89.0% |
| **Ensemble raw** (v72.fn + v28.exec + v72.dir) | 95.9% | — | — | **90.9%** |
| **Ensemble + Python rule** | 94.5% | 96.3% | 95.4% | **89.0%** |

## 결론 — Ensemble로 93% 회복 실패

- NPU 양자화 손실이 v28/v72 두 NB 모두에 있어 head 합성으로 메꿈 불가
- v28 raw 92.7% (단독, 룰 없음)이 의외로 가장 높음 — GT 219의 `unknown` 라벨 케이스에서 v28의 보수적 분류가 우연히 매치 (v72는 over-classify로 unknown 7건 놓침)
- rule 적용 시 v28 결과를 v72 기준 룰이 망가뜨려 89%로 하락
- **NPU 천장 = 89~91% 근처. ensemble 실익 없음 확정.**

### v28 vs v72 unique 비교 (fn 기준)
- v28만 맞춤 7건 — 전부 `unknown` 라벨 (v72는 system_meta/news_query/medical/market으로 over-classify)
- v72만 맞춤 7건 — light/ac/weather/traffic/medical 일반화 케이스

→ v28은 학습 데이터 좁아서 unknown 보수 분류 / v72는 비유/간접 일반화로 over-classify. 두 모델 강점이 다른 영역.

## 디바이스/앱 매핑 (2026-05-18 확인)

| Android Studio 프로젝트 | applicationId | 정체 |
|---|---|---|
| t527_smart_v2 | com.t527.smart_v2 | NLU 측정/테스트 (NpuEvalSingleActivity 위치) |
| t527_vad_service | com.t527.vad_service | 통합 production (NLU 코드 포함) |
| t527_ondevicevoice_service | **com.hdclabs.ondevice.voice** | **active 177MB**, ALSA 캡처 + Conformer STT + 단지서버 REST (NLU 없음) |
| t527_vad_service_v3 | com.hdclabs.ondevicevoice | 구버전 (점 위치 다른 패키지) |
| t527_ondevice_ai_agent (WSL) | com.t527.wallpad_agent | 시연대 통합 (KoElectra NLU + TTS + CitriNet) |

## Linked

- `wiki/projects/t527-npu-integration.md`
- `wiki/models/version-matrix.md`
- `src-npu-real-gt-219-20260515.md`
- `src-pipeline-port-handoff-20260515.md`

## Last updated

- 2026-05-18
