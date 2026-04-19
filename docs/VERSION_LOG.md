# NLU Version Log

## [2026-04-19] update | v28-test-suite-3000-and-preprocessor

- **Test Suite 3,043개 달성**, v28 combo 100%.
- STT 전처리 사전 54개 추가 (`scripts/preprocess.py`): 한글숫자 변환(이십삼도→23도) + STT 교정(남방→난방, 에어콘→에어컨 등).
- confidence fallback 추가 (conf<0.5 → unknown).
- param_type 규칙 보정 (open/close→none, judge→none, query/direct→none).
- 알려진 실패 32개 문서화 (`docs/KNOWN_FAILURES.md`).
- ONNX export: `nlu_v28_final.onnx` (99.7MB, 0.32ms CPU).

## [2026-04-19] update | v10-to-v28-iterative-improvement

- v10→v28 반복 개선 (19 iterations).
- v10: "거실 불 꺼줘"→ac_control 오분류 수정 (light_off 학습 데이터 0개 문제).
- v11: 96개 종합 테스트에서 발견된 9개 fn + 7개 exec + 7개 dir 오류 전수 수정.
- v12: "환기 세게"→dir:up, "뭐 할 수 있어?"→system_meta 수정.
- v13: "더워"→ac, "시끄러워"→vent off, "외출할게"→security 수정.
- v14: 78개 종합 fn 100%. 미세먼지/우산/밖에더워/오늘어때 수정.
- v15: "어둡게 해줘"→light, "공기가 안 좋아"→vent, "그만"→off 수정.
- v16: 실사용 92개 테스트 오류 18개 전수 수정 (은은하게/데워줘/블라인드/방범모드 등).
- v17: "불 안 꺼져?"→light query, "환기 안 되나?"→vent query 수정.
- v18: STT 오류 내성 강화 (오늘날씨어때/재습/면시야/뉴쓰/근쳐 등).
- v19: 구어체 수정 (조용히→볼륨, 들어왔어→귀가, 자려고→취침, 히팅→난방). **val combo 94.9% 최고치**.
- v20~v21: judge/param head 수정 (반팔→clothing, 미세먼지→air_quality).
- v22~v27: 실험적 시도 → regression 발생으로 v28에서 안정화.
- v28: **Test Suite 135개 combo 100%**, 띄어쓰기 없는 발화 direction 수정, 감탄형 exec 수정.
- v29~v33: 추가 실험 (통합 재증강, 패턴 추가) → 모두 v28보다 regression → **v28 확정**.
- 교훈: 패치 데이터 추가 시 기존 패턴이 밀림. Test Suite로 regression 즉시 감지하는 것이 핵심.

## [2026-04-18] update | v7-to-v9-data-augmentation

- **v7**: 고다양성 증강 — 시나리오당 avg 25→94 유니크. 변형 규칙 대폭 강화. GT 증강 19,237개 + CNN 보충 + unknown 4,818개 = 24,351개. 219 GT fn 100%, combo 94.5%, unknown 95%.
- **v7 clean**: GT 원본을 train에서 제거 후 재학습. 219 GT fn 100% 유지 확인.
- **v9**: 부족 fn 집중 증강 (elevator 218→615, gas 362→702, door 372→797). 219 GT fn 99.1%, combo 94.9%.

## [2026-04-18] update | v1-to-v6-initial-development

- **v1**: CNN 5-head 최초 학습. CNN 94-intent 16,879개 → 19 fn multi-head 재매핑. val fn 96.5%, combo 90.8%.
- **v2**: MASSIVE 한국어 5,069개 추가 + judge/clarify 특수 데이터 1,026개. 르엘142 fn 95.1%, combo 89.1%.
- **v3**: unknown class 최초 도입. GT 증강 시나리오당 100개 목표 (실제 avg 25). combo 89.5%.
- **v4**: 219개 원본 GT 기준 평가 시작 (이전까지 142개 파생 파일 사용). 219 GT fn 96.3%, combo 86.3%, unknown 70%.
- **v5**: unknown weight 0.3 → 거부율 오히려 하락. 폐기.
- **v6**: targeted hard negatives 추가. 219 GT fn 97.3%, combo 86.5%, unknown 50%.

## [2026-04-17] bootstrap | architecture-decision

- 아키텍처 비교 분석: flat intent vs multi-head vs hybrid.
- 조합 폭발 분석: 136 flat intent (room 제외) vs multi-head fn(19) + exec(5) + dir(9).
- 데이터 효율: multi-head 7~27x 유리 (20K 데이터 기준).
- Google/Alexa Trait 시스템 분석 → multi-head가 동일 구조.
- **결정: CNN 4L + 5-head (fn/exec/direction/param_type/judge) + rule slots**

## [2026-04-16] bootstrap | data-collection

- 르엘 GT 219개 시나리오 파싱 (260330 엑셀 기준 개발O).
- 65개 조합(구분|세부기능|서비스유형) 분석.
- 외부 데이터셋 조사: MASSIVE ko-KR, Home-Assistant-Requests, Fluent Speech Commands, SNIPS.
- HA 영어 583개 한국어 번역 (규칙 기반).
- MASSIVE 비IoT → OOD/unknown 학습용.

## [2026-04-19] update | koelectra-v8-data-merge-prepared

- 다른 세션에서 KoELECTRA v8 학습 데이터 13,540개를 5-head 포맷으로 변환.
- 79개 KoELECTRA label → 20개 fn multi-head 매핑 완료 (0개 스킵).
- 병합: 21,672 → 34,513개 (`data/train_final_v33.json`).
- 부족 fn 대폭 보강: gas_control +725, door_control +564, ac_control +1,975 등.
- 교차검증용 val 1,536개 별도 보존 (`data/koelectra_converted_val.json`).
- 학습 + 평가 대기 중.

## [2026-04-19] experiment | v33-koelectra-merge-regression

- KoELECTRA v8 데이터 12,841개 병합 학습 (총 34,513개 → 실제 20,823개 로드).
- val combo: 94.2% (v28 94.7% 대비 -0.5%p).
- **Test Suite 3,043개: 90.5% (290개 오류)** — v28 100% 대비 심각한 regression.
- 주요 오류: exec_type/direction 라벨 불일치 210건, fn 오류 80건.
- 원인: KoELECTRA 라벨 체계(79 flat intent)와 우리 5-head 라벨 간 변환 시 exec/dir 매핑 부정확.
- **결론: v28 유지. KoELECTRA 데이터는 fn만 선별 활용 필요.**

## [2026-04-19] experiment | v34-pseudo-labeling-breakthrough

- KoELECTRA 13,189개를 pseudo-labeling으로 병합 (fn=KoELECTRA 원본, exec/dir=v28 예측).
- **외부 데이터 fn: 75.5% → 96.8% (+21.3%p)** — 일반화 능력 대폭 향상.
- Test Suite: 100% → 90.6% — trade-off 발생.
- **핵심 발견**: pseudo-labeling이 직접 라벨 매핑보다 훨씬 효과적.
  - v33 (직접 매핑): KoELECTRA fn 변화 없음 + Test Suite 90.5% regression
  - v34 (pseudo-labeling): KoELECTRA fn 96.8% + Test Suite 90.6%
  - 같은 regression이지만 외부 데이터 정확도가 21%p 높음
- 다음 과제: Test Suite regression 복구 — 기존 fix 패턴 + KoELECTRA 데이터 공존 방법 연구
