# NLU Changelog — 버전별 업데이트 내역

## v15 (현재)
- Hard edge case 수정: "어둡게 해줘"→light, "공기가 안 좋아"→vent, "그만"→off
- "불 켤 수 있어?"→light, "난방 25도로 맞춰"→heat 수정
- 수정 패턴 6/6 정확
- 219 GT: 99.1% | combo: 94.6% | false rejection: 0건
- 데이터: 20,289개

## v14
- 78개 종합 테스트 fn **100%** (78/78), 219 GT **100%**, combo **94.7%**
- v11~v14 반복 개선으로 오분류 패턴 전수 수정:
  - 간접표현: "더워"→ac, "시끄러워"→vent off, "외출할게"→security
  - STT오류: "커턴"→curtain, "남방"→heat
  - 경계: "밖에 더워?"→weather vs "더워"→ac 구분
  - direction: "환기 세게"→up, "에어컨 온도 내려"→down, "에어컨 23도"→set
  - exec: "송풍 해줘"→control(not clarify), "뉴스 틀어줘"→query(not unknown)
- 데이터: 20,247개

## v13
- 확장 테스트 55개 fn 98.2% (54/55)
- "더워"→ac, "시끄러워"→vent, "외출할게"→security, "커턴"→curtain 수정
- "오늘 어때?"→market 오분류 1건 남음
- 219 GT fn: 100%

## v12
- v11 남은 오류 2개 수정 ("환기 세게"→dir:up, "뭐 할 수 있어?"→system_meta)
- 핵심 33개 combo **100%** (33/33)
- combo: 94.7%

## v11
- v10 종합 96개 테스트에서 발견된 9개 fn오류 + 7개 exec오류 + 7개 dir오류 전수 수정
- 난방낮춰→heat, 에어컨온도→ac, 제습→ac, 미세먼지→weather, 우산→weather, 뉴스틀어→news, 볼륨→home, 사용법→system
- 핵심 43개 combo 95.3% (41/43)
- combo: 94.6%

## v10
- HA 영어 583개 한국어 번역 (Home-Assistant-Requests → 규칙 기반 번역)
  - door_control +97, curtain_control +127, vent_control +283 보강
- "불 꺼줘" → light_control 패턴 수정 (light_off 44개 + light_on_room 24개 추가)
- 기본 테스트 14/14 정확
- 데이터: 19,966개 | combo: 94.7%

## v9
- 부족 fn 집중 증강 (elevator 218→615, gas 362→702, door 372→797)
- 전용 표현 추가 (엘리베이터/가스/도어락 자연어 변형)
- 데이터: 22,724개 | 219 GT fn: 99.1% | combo: 94.9%
- 문제: "거실 불 꺼줘" → ac_control 오분류 (light_off 학습 데이터 0개)

## v8
- v7 edge case 수정 시도 (볼륨, 난방온도, 우산, 미세먼지 패턴 추가)
- 데이터: 20,861개 | combo: 94.4%
- 문제: false rejection 발생 ("환기 켜줘"→unknown, "뉴스 틀어줘"→unknown)
- 결론: v7보다 나빠져서 폐기

## v7 (clean)
- **GT 원본 제거 후 재학습** — 학습 데이터에 테스트 원본 포함 문제 수정
- 219 GT 원본을 train에서 빼고 증강 변형만으로 학습
- 데이터: 20,515개 | 219 GT fn: 100% | combo: 94.5% | unknown: 90% | false rej: 0건

## v7
- **고다양성 증강** — 시나리오당 avg 25 → avg 94 유니크
- 변형 규칙 대폭 강화 (동의어, 조사변형, 단어탈락, 복합변형)
- GT 증강 19,237개 + CNN 보충 + unknown 4,818개 = 24,351개
- 데이터: 24,351개 | 219 GT fn: 100% | combo: 94.5% | unknown: 95%

## v6
- targeted hard negatives 추가 (택시, 병원예약, 종목추천 등 미지원 패턴)
- unknown 거부 개선 시도
- 데이터: 10,342개 | 219 GT fn: 97.3% | combo: 86.5% | unknown: 50%
- 문제: hard negative 수 부족 (targeted 39개), unknown 거부율 낮음

## v5
- unknown weight를 0.3으로 낮춤 → 거부율 오히려 하락
- 데이터: 10,342개 | 219 GT fn: 96.8% | combo: 86.3% | unknown: 50%
- 결론: v4보다 나빠져서 폐기

## v4
- **219개 원본 GT 기준 평가 시작** (이전까지 142개 파생 파일 사용)
- unknown class 추가 (fn 19→20)
- unknown weight 0.5로 보정
- 데이터: 10,342개 | 219 GT fn: 96.3% | combo: 86.3% | unknown: 70% | false rej: 0건

## v3
- unknown class 최초 도입 (MASSIVE OOD + HA vacuum/media)
- GT 증강 시나리오당 100개 목표 (실제 avg 25 유니크)
- 데이터: ~10K | combo: 89.5% | unknown: 75%
- 문제: 증강 다양성 부족, clarify 라벨 버그

## v2
- MASSIVE 한국어 5,069개 추가 + judge/clarify 특수 데이터 1,026개
- weather 다운샘플링 (10,005→2,000)
- 데이터: 13,904개 | 르엘142 fn: 95.1% | combo: 89.1%
- 문제: fn|exec 51개 조합 중 35개 목표 미달, 1개 0건

## v1
- **CNN 5-head 최초 학습**
- 기존 CNN 94-intent 데이터 16,879개 → 19 fn multi-head 라벨 재매핑
- 부족 fn 규칙 기반 증강 8,045개
- 데이터: 13,904개 | val fn: 96.5% | combo: 90.8%
- 문제: fn 레벨 밸런싱만 함 (fn|exec 조합 밸런싱 안 함), weather 10,005개 지배

---

## 아키텍처 변경 이력

| 시점 | 아키텍처 | 변경 이유 |
|------|---------|----------|
| 초기 | CNN 94-intent flat | 첫 NLU |
| SAP | TextConformer 3L + 8-head | multi-head 분해 시도 |
| v1~ | **CNN 4L + 5-head** | Conformer→CNN (94%>86%), 8→5 head (불필요한 head 규칙으로) |

## 데이터 소스 이력

| 소스 | 도입 시점 | 용도 |
|------|---------|------|
| kochat_intent.csv | 초기 | 한국어 의도 분류 |
| Amazon MASSIVE ko-KR | v2 | IoT/weather/news/alarm + OOD |
| 르엘 GT 219개 증강 | v3 | 핵심 학습 데이터 |
| Home-Assistant-Requests | v10 | 영어→한국어 번역 (door/curtain/vent/heat/ac) |
| CNN 94-intent 재매핑 | v1 | multi-head 라벨 변환 |

## 주요 버그/실수 기록

| 버전 | 내용 |
|------|------|
| v1 | fn 레벨로만 밸런싱 — fn|exec 조합 불균형 방치 |
| v1 | 142개 파생 파일을 GT로 사용 — 219개 원본이 진짜 GT |
| v3 | GT 원본을 train에 포함한 채 평가 — 100% 허수 |
| v3 | clarify 라벨 버그 — room 추가해도 exec_type 안 바꿈 |
| v1~v6 | "불 꺼줘" 학습 데이터 0개 — 기본 명령 오분류 |
| v8 | 패턴 수정하다 false rejection 발생 — v7보다 나빠짐 |
