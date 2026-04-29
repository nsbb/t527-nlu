# 작업 현황 — 대화 압축 대비 상시 갱신

> 이 파일은 대화가 압축되더라도 작업 맥락을 유지하기 위해 매 작업 후 업데이트한다.

---

## 핵심 파일 구조

| 파일 | 역할 | 상태 |
|------|------|------|
| `scripts/ensemble_inference_with_rules.py` | NLU 추론 + 후처리 규칙 (v108) | ✅ 메인 |
| `scripts/preprocess.py` | STT 오인식 정규화 | ✅ 메인 |
| `scripts/deployment_pipeline_v2.py` | NLU 파이프라인 (preprocess→NLU→DST→응답) | ✅ 메인 |
| `scripts/dialogue_state_tracker.py` | 멀티턴 DST (방상속/fn상속/slot filling) | ✅ 메인 |
| `scripts/response_generator_v2.py` | AI기대응답 생성 (OO placeholder 포함) | ✅ 메인 |
| `test_api.py` | **CLI 테스트 앱 + 실제 API 연동** | ✅ 완성 — 건드리지 말 것 |
| `scripts/nlu_test_app.py` | Streamlit 웹 UI (API 통합, DST 인스턴스 분리) | ✅ 완성 |
| `scripts/run_auto_test.py` | 자동 테스트 (221단일 + 33멀티턴) | ✅ 메인 |
| `data/test_suite.json` | TS 3043개 GT | ✅ 메인 |
| `docs/CHANGELOG.md` | 버전별 변경 내역 | ✅ 매 버전 업데이트 |

---

## test_api.py — 완성된 CLI 앱 (핵심!)

**실행:** `python3 test_api.py` (프로젝트 루트에서)

**기능:**
- Open-Meteo API (무료, 키 없음) → 실시간 날씨 (기온/체감/습도/강수/풍속/일출일몰)
- Open-Meteo Air Quality API → 실시간 미세먼지 (PM10/PM2.5)
- 경향신문 RSS → 실시간 뉴스 3건
- `fill_placeholders()` — 응답의 OO/00 자리표시자를 실데이터로 치환
- `weather_response()` — 빨래/운동/외출/옷차림 등 판단형 응답
- `DeviceState` — 디바이스 상태 추적 (`d` 입력으로 확인)
- 멀티턴 DST 활성, Enter×2로 DST 초기화

**위치:** `LAT=37.4837, LON=127.0324, DISTRICT='서초구'` (필요시 수정)

**OO 자리표시자 치환 패턴:**
```
"오늘 OO구 OO동 날씨는 맑고/흐리고/비가 오고/춥고/덥고 최고 00도, 최저 00도이며
 미세먼지는 나쁨/보통/좋음 수준입니다."
→ "오늘 서초구 날씨는 맑고 최고 23.4도, 최저 12.1도이며
   미세먼지(PM10 45.2㎍/m³)는 보통 수준입니다."
```

---

## Streamlit 앱 — 완성 (2026-04-29)

**실행:** `streamlit run scripts/nlu_test_app.py`

**기능:**
- 단일발화 탭: 개별 발화 테스트 + API 응답 (날씨/미세먼지/뉴스)
- 멀티턴 탭: DST 상태 추적, 대화 이력, 홈 상태 표시
- OO placeholder 실데이터 치환 (날씨/미세먼지 API)
- 단일용/멀티턴용 파이프라인 별도 인스턴스 (DST 충돌 방지)

---

## 현재 NLU 성능

| 지표 | 수치 |
|------|------|
| TS 3043 combo | **100.00%** |
| auto_test 단일 | **306/306 (100%)** |
| auto_test 멀티턴 | **60/60 (100%)** |
| KoELECTRA fn | **96.22%** |
| 최신 규칙 버전 | **v128** |

---

## 버전 히스토리 요약 (v96 이후)

| 버전 | 내용 |
|------|------|
| v96 | TS 100% 최초 달성 |
| v97 | 조건부/hearsay/비유 처리 |
| v98 | 꿉꿉/눅눅→vent, 귀가+환기 |
| v99 | 창문잠겼어→door, ventilation preprocess |
| v100 | 꺼줘요 dir버그, 켜져있잖아 불만 |
| v101 | 더위비유 확장, 온실만족→unknown |
| v102 | 완곡 제안형 dir복구 |
| v103 | 냉장고→미지원, 가스냄새→close, 빨리→dir복구 |
| v104 | 요리중→vent, 밖이추운것→weather |
| v104b | 거이→거실 preprocess, 밝혀줘 unknown 누락 |
| v105 | 탁해→vent, 너무세다→down, 꺼도되지→off |
| v106 | v72 오예측교정, 쾌적했으면→ac/on |
| v107 | 부정명령, 과거형보고, 켜놓아줘, 여기불켜줘 |
| v108 | hearsay가족확장, 지시어dir, STT오인식, 모드해제 |
| v109 | 창문닫았어gas교정, 껴줘STT, 의향형, 만족관찰unknown, 수면표현 |
| v110 | 간접화행(문열려있어/이상소리/불켜져있어), 요리패턴확장 |
| v111 | 한증막→ac교정, 약한것같아↑, 어두운것같아↑, 훈훈포근→unknown, 에어캔preprocess |
| v112 | 더시원하게→ac/up, 수사적반어(끄면안되나→off), 이중부정(안끄면안돼→off), 월패드→unknown |
| v113 | DST: AC컨텍스트 온도높여줘→ac/up, 커튼컨텍스트 완전히내려줘→curtain/close |
| v114 | 수사적불평(왜이렇게어둡냐→up, 버티겠어→on), 완곡(더운것같기도→on), STT(커트/벤틸) |
| v115 | 가스잠궈줘→close, 창문잠궈줘→door, 일어났더니춥네→heat, hearsay켜달래→on, 땀삐질삐질→ac/on |
| v116 | 습하네/습해서→vent, 생선구이냄새→vent/on, 미세먼지+창문닫아야겠다→door/close |
| v128 | 집상태알려줘→home_info(v72교정), 에어컨세게틀어줘→up, 세게GT교정 |
| v127 | 동굴비유→light/on, 낮추면안될까→down, 땀이비처럼→ac/on, 안써도될것같아→off, 꺼도될것같아요→off |
| v126 | 비유(얼음창고→heat/on, 북극→ac/down), 의무형(닫아야/올려야), 효과부족(켰는데도더워→up), 어르신hearsay |
| v125 | 송풍해줘(공백없음)→ac/set, 커튼쳐줘→curtain/close(stop오예측교정) |
| v124 | 추위를타→heat/on, 아주많이추워→heat/on, 창문열어도돼요→door/open, 극도로덥습니다→ac/on, 잠깐만켜줘/꺼줘→dir교정 |
| v123 | STT(에어킁→에어컨,보이라→보일러), 창문열려있네→door/close, 불아직도켜져있어→light/off, 창문열어주겠어요→door/open, 환기해주시겠어요→vent/on |
| v122 | 조명밝은것같→down, 에어컨세다싶→down, 꺼줄수있어→off, 미세먼지창문닫아줘curtain오예측→door/close |
| v121 | 이더위에어떻게→ac/on, 쌀쌀한/서늘한형용사형→heat/on, 조명깜빡거려→unknown, 창문닫아도될까요→door/close, 어두워서안보여→light/up |
| v120 | 비/눈오기시작→door/close, 창문바람들어오→close, 창문열어두고외출→close, 건조해→unknown, 해눈부셔→curtain/close, 에어컨끄면안될까→ac/off |
| v119 | 전동커튼 열기 멈춰줘 → stop (open 오예측 교정) |
| v118 | 수사적추위(이추위에어떻게→heat), 냉방병→ac/down, 창문열어볼까→door, 밝혀볼까→up, STT(헌기→환기), DST:식사완료→vent/off, DeviceState파이프라인연동, fill_placeholders슬롯처리(맑고/흐리고/나쁨/보통/좋음) |
| v117 | STT(남바/에어껀), 요리예고형(삼겹살구워먹을건데→vent), 소원형(에어컨있으면얼마나→on) |

---

## 작업 규칙 (반드시 준수)

1. **규칙 추가 후 즉시:** `python3 scripts/ensemble_inference_with_rules.py` → TS 100% 확인
2. **auto_test 통과 후 즉시:** `git add ... && git commit && git push`
3. **test_api.py는 완성본** — 수정할 때는 기존 기능 유지하면서 추가만
4. **이 파일 갱신** — 주요 변경사항마다 WORKING_STATUS.md 업데이트
