# T527 디바이스 NLU 시스템 — 13 iterations 종합 보고서

작성: 2026-05-07

## 최종 상태

### End-to-End 체인 완성 (디바이스 검증됨)

```
사용자 발화
    ↓
[Preprocess] 296 매핑 (STT 정규화)
    ↓
[NLU] KoELECTRA + CNN 5-head ONNX (105MB, 21~30ms)
    ↓
[PostRules] 자동 36 + 수동 28 = 64 규칙
    ↓
[IntentRouter] 4채널 분배 (AIDL/REST_RUEL/EXTERNAL_API/SELF)
    ↓
[AidlMockClient] AIDL 호출 시뮬 (실 IoT 연동 전)
    ↓
[ResponseGenerator] 한국어 응답 + 받침 자동 처리
    ↓
[TtsPlayer] 미리 녹음 MP3 재생 (98개 fallback)
    ↓
사용자 응답
```

### 측정 결과

| 지표 | 값 |
|------|-----|
| **정확도 (491 골든셋)** | 93.5% combo |
| **정확도 (99 골든셋)** | 98.0% combo |
| **NLU 평균 latency** | 21~30ms |
| **End-to-End 응답** | ~330ms (STT+NLU+Router+Response+TTS) |
| **NLU 모델 크기** | 105MB ONNX (CPU) |
| **검증 시나리오** | 13건 + 491개 |

## 아키텍처 — Kotlin 모듈 8개

```
src/main/kotlin/com/t527/smart_service/
├── nlu/
│   ├── Preprocess.kt          (296 매핑 STT 정규화)
│   ├── IntentClassifierV46.kt (ONNX 추론 + 규칙 적용)
│   ├── WordPieceTokenizer.kt  (BERT 토크나이저)
│   ├── PostRules.kt + V3 + V4 (64개 규칙)
├── integration/
│   ├── IntentRouter.kt        (NLU → 채널 라우팅)
│   ├── AidlMockClient.kt      (AIDL 호출 시뮬)
│   ├── ResponseGenerator.kt   (한국어 응답 생성, 받침 처리)
├── audio/
│   └── TtsPlayer.kt           (MP3 TTS 재생)
├── NluBenchmarkActivity.kt    (15개 + 99개 골든셋 검증)
├── IntegrationDemoActivity.kt (13개 End-to-End 데모)
└── InteractiveTestActivity.kt (사용자 입력 테스트)
```

## Iteration 별 진전

| # | 작업 | 결과 |
|---|------|------|
| 1 | 디바이스 첫 NLU 추론 | 17ms, 66.7% (raw) |
| 2 | NB 변환 (CNN body 추출) | 5.91MB ONNX, Acuity import 성공 |
| 3 | 자동 포팅 도구 (V2) | 87 규칙, 회귀 발견 |
| 4 | 엄격 모드 (V3) | 36 규칙 (안전), 100% 회복 |
| 5 | 골든셋 인프라 99개 | 디바이스 자동 검증 |
| 6 | Preprocess 모듈 (V4) | 296 매핑, 94.9% |
| 7 | 보정 3건 추가 (V5) | 98% (서버 동등) |
| 8 | 골든셋 491개 (정직 측정) | 93.5% (over-fit 발견) |
| 9 | PostRules 부작용 진단 | 시뮬-실제 차이, 통합 방향 전환 |
| 10 | AIDL Mock + IntentRouter | 4-channel End-to-End |
| 11 | ResponseGenerator (Kotlin) | 한국어 응답 템플릿 |
| 12 | 받침 처리 + MP3 TTS | 음성 출력 완성 |
| 13 | InteractiveTestActivity | 사용자 입력 테스트 (코드 작성) |

## 중요한 발견

1. **자동 포팅 도구 한계** — fn/dir 가드 없는 단순 액션은 over-eager 매칭으로 부작용. 엄격 모드 필수.
2. **골든셋 크기 중요** — 99개에서 98%였던 게 491개에서 93.5%. over-fit 위험.
3. **디바이스 ARM CPU 한계** — 21~30ms (서버 0.64ms 대비 30배). 그래도 실시간 가능.
4. **Acuity quantize 막힘** — inputmeta.yml 호환성 4가지 시도 실패. NPU 변환은 다음 작업.

## 남은 작업

- [ ] Acuity quantize 디버깅 (inputmeta GENERATOR 또는 다른 방법)
- [ ] NB 변환 성공 시 NPU 추론 (목표 ~3ms)
- [ ] VoiceAiService 본 서비스 통합 (실 STT → NLU 체인)
- [ ] PostRules 정밀화 (디바이스 raw vs 후처리 비교 모드)
- [ ] InteractiveTestActivity PackageManager 캐시 문제 해결

## 산출물 (GitHub)

```
nsbb/t527-nlu @ 12 commits this session:
  - docs/T527_NLU_BENCHMARK_RESULT.md        벤치마크 진화 기록
  - docs/T527_INTEGRATION_DEMO.md            End-to-End 통합
  - docs/T527_RESPONSE_GENERATOR.md          응답 생성기
  - docs/T527_NB_CONVERSION_LOG.md           NB 변환 진행
  - docs/AIDL_RUEL_GAP_REPORT.md             AIoT 갭 분석
  - docs/AIOT_REQUEST_1PAGER.md              AIoT 요청 1쪽
  - docs/RTM_REQUIREMENTS_TRACEABILITY.md    219 매트릭스
  - docs/INDUSTRY_PRACTICES_RESEARCH.md      산업 표준 조사
  - data/golden_test_100.json + _500.json    골든셋
  - scripts/auto_port_rules_to_kotlin.py     자동 포팅 도구
  - android_assets/stt_correction_v2.json    Preprocess 매핑 (15.6KB)
  - 디바이스 코드 (Kotlin 모듈 9개)         t527_smart_v2 프로젝트
```
