# 알려진 실패 패턴 (v46 기준, 2026-04-21 업데이트)

v46 모델의 Test Suite 3,043개 중 204개 오류 (93.3%) 분석.
Ensemble v28+v46 적용 시 172개 → 94.3% (일부 오류 해소).

## 오류 분류 (204건)

| 유형 | 건수 | 비율 |
|------|:---:|:---:|
| fn 오류 (분류 잘못) | 62 | 30.4% |
| dir 오류만 (방향 잘못) | 82 | 40.2% |
| exec 오류만 (실행 타입 잘못) | 46 | 22.5% |
| exec + dir 동시 오류 | 14 | 6.9% |

## fn 혼동 Top 패턴 (v46)

| 기대 fn | 예측 fn | 빈도 | 발화 예시 |
|---------|---------|:---:|----------|
| schedule_manage | system_meta | 19x | "알람 맞춰줘", "타이머 설정" |
| home_info | system_meta | 6x | "화면 밝기", "월패드 밝기" |
| unknown | system_meta | 6x | "기능 알려줘" (라벨 모호) |
| weather_query | unknown | 5x | "서울 날씨", "오늘날씨어때" |
| medical_query | unknown | 3x | "근처 신경외과", "약국 위치" |
| unknown | security_mode | 3x | - |
| news_query | unknown | 2x | "국제 뉴스" |

## exec 혼동 Top 패턴

| 기대 | 예측 | 빈도 | 원인 |
|------|------|:---:|------|
| query_then_respond | control_then_confirm | 27x | "~상태", "~확인" 구어체 |
| direct_respond | control_then_confirm | 9x | 단일 단어 발화 ("조명") |
| control_then_confirm | query_then_respond | 7x | 모호한 단축 발화 |
| control_then_confirm | direct_respond | 5x | - |
| clarify | control_then_confirm | 3x | "간접등 켜줘" |

## dir 혼동 Top 패턴

| 기대 | 예측 | 빈도 | 패턴 |
|------|------|:---:|------|
| none | set | 11x | 애매한 값 지정 |
| down | on | 10x | "줄여줘"→on으로 예측 |
| on | none | 10x | 짧은 발화 |
| down | up | 8x | 반대 방향 혼동 |
| on | up | 8x | - |
| down | none | 7x | "줄여"만 있을 때 |
| close | open | 5x | 반대 |
| on | set | 5x | - |
| on | off | 4x | 반대 |

## KoELECTRA val 34건 분석 (실제 정확도 ~98.8%)

v46의 KoELECTRA fn 97.8%가 실제로는 더 높음:

### KoELECTRA 라벨 오류 (16건, 우리 모델이 맞음)
- unknown→system_meta 9건: "기능 알려주세요", "너 이름 뭐야"는 system_meta가 정답
- system_meta→home_info 7건: "소리 키워줘", "볼륨 낮춰줘"는 home_info가 정답

### 진짜 어려운 케이스 (18건)
- "찜질방 같아" → ac/heat 구분 모호
- "손발이 꽁꽁" → heat/light 모호
- 단일 발화로는 문맥 부족, **DST 필요**

## v46/Ensemble이 여전히 틀리는 대표 패턴

### 학습 데이터 경계 밖 (10+ 예시)
| 발화 | 기대 | v46 예측 | 원인 |
|------|------|---------|------|
| 알람 취소해줘 | schedule_manage | system_meta dir:set | "취소"→system 매핑 학습됨 |
| 전화해줘 | unknown | home_info | 전화 기능 학습 없음 |
| 좀 시원하게 | ac_control | heat_control | "시원"이 heat 쪽 분포에 섞임 |
| 전체 꺼 | ac_control | light_control | 전체 끄기 → light default |
| 와이파이 비번 뭐야 | unknown | system_meta | system에 와이파이 연관 |
| 작년보다 추워? | weather_query | energy_query | "작년" → 비교 분석으로 오해 |

### 모호한 단일 표현
| 발화 | 기대 | v46 | 왜 어려운가 |
|------|------|-----|------------|
| 조명 | light_control, direct | light, control | 짧은 발화는 control로 편향 |
| 에어컨 | ac_control, direct | ac, control | 동일 |
| 에어컨 바람 | ac, dir:up? | ac, dir:none | "바람"만으론 방향 불명 |

### STT/구어체 변형
| 발화 | 기대 | v46 | 해결 방법 |
|------|------|-----|---------|
| 남방 꺼쥬 | heat_control, off | heat, on | preprocess에서 "꺼쥬→꺼줘" 추가 필요 |
| 에어컨꺼 | ac, off | ac, on | "에어컨꺼→에어컨 꺼" 분리 (v67 preprocess에 추가됨) |
| 오늘날씨어때 | weather, query | unknown, direct | 띄어쓰기 없는 질의 어려움 |

### DST 필요 (문맥 의존)
| 발화 | 기대 | v46 단독 | 해결 |
|------|------|---------|------|
| 안방도 | (prev context) | light_control, clarify | DST로 이전 fn 상속 |
| 아니 꺼줘 | (prev) | light, control, off | DST 교정 패턴 |
| 응 | (prev) | unknown, direct | DST 확인 패턴 |

## 우리가 해결한 패턴 (역사적)

### v28 이전 (해결됨)
- "거실 불 꺼줘" → v10에서 수정 (light_off 학습 데이터 0→44개)
- "환기 세게" → v12에서 수정 (dir:up 누락)
- 한글 숫자 "이십삼도" → v18 preprocess로 해결
- STT "미세문지" → v18 preprocess로 해결

### v29-v33에서 시도했으나 regression
- 패치 형태 수정 → 분포 왜곡 → v28에서 안정화

## 해결 가이드

### 1. 예측 맞지만 지원 안 됨 (UNSUPPORTED_ACTIONS)
`sap_inference_v2.py`에서 키워드 매칭하여 "죄송합니다" 응답

### 2. 낮은 confidence
conf < 0.5 → unknown으로 fallback → 서버 처리

### 3. 문맥 의존 (DST)
멀티턴 상태 추적 (10초 timeout)
- room follow-up: "안방도"
- device follow-up: "난방도"
- correction: "아니 꺼줘"
- confirm: "응"

### 4. 실제 사용 로그 수집 필요
현재 데이터는 증강 기반. 실 사용 로그 피드백 루프로 개선 가능.

## 참고 문서

- `docs/CHANGELOG.md` — 버전별 변화
- `docs/VERSION_LOG.md` — 실험별 상세 결과
- `docs/MODEL_LIMITATIONS.md` — 구조적 한계 9가지
- `docs/DEPLOYMENT_GUIDE.md` — 배포/통합 가이드
