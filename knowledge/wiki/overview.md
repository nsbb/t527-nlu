# Overview — t527-nlu

## Current state

T527 월패드용 한국어 NLU 시스템. multi-head CNN (20 fn × 5 exec × 9 dir × 5 param × 5 judge).

**중요: 서버 production과 디바이스 production이 다른 모델을 쓰고 있다.**

| 위치 | 모델 | 정확도 (GT-219 ref) |
|---|---|---|
| 서버 (`deployment_pipeline_v2.py`) | **v28+v72 ensemble** | HANDOFF 주장 94.06% combo (rule 포함, 미검증) |
| 디바이스 (`t527_smart_v2` 앱) | **v28+v46 ensemble** (110MB ONNX) | raw 60.3% combo, +rule TBD |
| 디바이스 NPU (`cnn_body_v46_int16.nb`) | **v46 단독 CNN body** (int16) | NPU+rule 60.7% combo (직접 측정 2026-05-07) |

→ 디바이스가 서버보다 한 세대 뒤처짐. v72 ONNX 디바이스 배포 + v72 NB 변환이 미완 작업.

## Known-good settings

- Tokenizer: ko-sbert WordPiece (`tokenizer/`), maxLen 32
- Preprocess: STT 정규화 296개 매핑 (`scripts/preprocess.py`)
- Post-rules: PostRulesV4 (Kotlin) / `ensemble_inference_with_rules.py` (Python)
- DST: 10s timeout, room/device/confirm/bare/intensity 5패턴
- HomeState: HVAC 재해석 + 상호배제

## Open issues

- 서버↔디바이스 모델 버전 gap (v72 vs v46)
- NPU NB는 v46 단독만 변환됨 — v72 cnn_body NB 변환 필요
- v46_errors_latest.csv (207건 오답)은 v46 기준이라 v72에서 다를 수 있음
- 르엘 219 ground-truth 라벨 중 "뉴스 dir=on/exec=control" 13건은 의심
- a527_wallpad iMapDefine.java 27개 enum 대응 라우팅 미구현

## Related sources

- `raw/source-notes/src-model-version-gap-20260513.md`
- 팀 wiki: `/home/nsbb/travail/claude/T527/wewonnim/t527_llm_wiki/wiki/models/koelectra-nlu.md` (자매 NLU)

## Last updated

- `2026-05-13`
