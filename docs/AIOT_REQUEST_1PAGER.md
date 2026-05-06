# AIDL 보강 요청 — 르엘 음성 NLU ↔ 실 기기 제어 정합성 (1-Pager)

**To:** AIoT 팀 / 기획팀  
**From:** AI팀 (NLU)  
**Date:** 2026-05-06  
**첨부:** `AIDL_RUEL_GAP_REPORT.md` (상세), `RTM_REQUIREMENTS_TRACEABILITY.csv` (전체 219 매핑), `INDUSTRY_PRACTICES_RESEARCH.md` (산업 표준 근거)

---

## TL;DR

> **르엘 음성 시나리오 219개 = GT** 기준으로 분석한 결과, 현재 AIDL 21 command 중 12개는 매칭되나 **갭 7건 (28 시나리오, 약 13%)** 이 미커버 상태입니다. AIDL **보강 3건 + 신규 4건** 으로 음성 NLU의 모든 시나리오를 실 기기 제어로 연결할 수 있습니다.

---

## 1. 전체 시나리오 219개 처리 채널 분포

| 채널 | 시나리오 | 상태 |
|------|---------|------|
| ✅ AIDL 매핑됨 (기존 AIDL 21개로 OK) | 52 (24%) | 변경 불필요 |
| 🔴 **AIDL 보강·신규 필요** | **28 (13%)** | **본 요청 대상** |
| 🔵 르엘 REST API | 26 (12%) | 이미 정의됨 |
| 🟢 외부 API (날씨/뉴스/교통/...) | 86 (39%) | NLU에서 처리 |
| ⚪ 자체 처리 (FAQ/매뉴얼/시간/알람) | 26 (12%) | 월패드 앱 내부 |
| ❓ 미분류 | 1 (door_open) | 명세 확인 필요 |

---

## 2. AIoT 보강 요청 7건

### 🟠 기존 command 확장 (3건)

| # | AIDL command | 변경 내용 | 영향 시나리오 | 우선순위 |
|---|---|---|---|---|
| 1 | `remote_access_doorlock` | **control(open) action 추가** (현재 status만) | 1 ("현관문 열어줘") | 🔥 High |
| 2 | `remote_access_curtain` | **ctrl_action: stop 추가** | 1 ("커튼 멈춰줘") | 🔥 High |
| 3 | `remote_access_mode` | **return_time 필드 추가** (외출 복귀 시간) | 3 (security_return_set) | 🟧 Medium |

### 🔴 새 command 신규 (4건)

| # | 새 AIDL command | 용도 | 영향 시나리오 | 우선순위 |
|---|---|---|---|---|
| 4 | `remote_access_schedule` | **모든 기기 예약 통합** (난방/AC/환기/커튼/조명/밝기) | 11 | 🔥 High |
| 5 | `remote_access_volume` + `remote_access_brightness` | 월패드 자체 (볼륨/화면 밝기) | 2 | 🟧 Medium |
| 6 | `remote_access_energy_alert` + `remote_access_energy_goal` | 에너지 알림 on/off + 목표 설정 | 4 (3 ↔ 1 분리) | 🟨 Low |
| 7 | `remote_access_emergency` | 비상 시 복합 제어 (가스 close + 도어락 + 알림) | 3 | 🟨 Low (안전 관련) |

→ **합계 7개 변경 → 28개 시나리오 미커버 해결.**

---

## 3. 르엘 시나리오에 없는 AIDL command (양방향 검토 요청)

AIDL에 정의되어 있지만 르엘 시나리오 0개:

| AIDL command | 음성 시나리오 추가 검토 |
|---|---|
| `remote_access_cooktop` / `_hybridcooktop` | 권장 ("쿡탑 꺼줘") |
| `remote_access_sensor` | 권장 ("공기질 어때?") |
| `remote_access_curtain2` / `_louver` | 검토 필요 |
| `remote_access_electric` (콘센트) | 검토 필요 |
| `remote_access_multivent` (각실 환기) | 검토 필요 |
| `remote_access_smartlight` | 시스템 내부 전용? |
| `remote_get_home_devices` | 시스템 내부 전용? |

→ 기획팀 협의 후 음성 시나리오 추가 또는 AIDL에서 제거 결정.

---

## 4. 일정 제안

| Week | 액션 | 담당 |
|------|------|------|
| Week 1 (현재) | 본 요청서 검토, 회의 | AIoT + 기획 + AI |
| Week 1 | AIoT 회신 (어떤 항목 수용/반려/일정) | AIoT |
| Week 1~2 | NLU 모델 데이터 증강·재학습 (블로킹 안 됨) | AI |
| Week 2~3 | AIDL 변경 사항 구현 | AIoT |
| Week 3 | NLU + AIDL Mock 통합 테스트 | AI |
| Week 4 | T527 디바이스 실 통합 + 시나리오 219개 검증 | 전사 |

---

## 5. 협의가 필요한 결정사항

1. **AIoT**: 7건 중 어느 것 수용? 어느 것 반려? 일정?
2. **기획**: AIDL 미사용 9개 중 음성 시나리오 추가할 것은?
3. **공통**: `door_open` (통화 카테고리, AIDL doorlock과 다른지) 명세 확인.

---

## 6. 첨부 문서

1. **`AIDL_RUEL_GAP_REPORT.md`** — 7건 각각의 NLU 5-head ↔ AIDL 매핑 상세 (368줄)
2. **`RTM_REQUIREMENTS_TRACEABILITY.csv`** — 219개 시나리오 1:1 매핑 매트릭스
3. **`INDUSTRY_PRACTICES_RESEARCH.md`** — Google/Amazon/학계 사례, RTM 표준, 회귀 테스트 (755줄)

---

## 회의 안건 (30분)

1. (5분) 현황 — 219 시나리오 분포, 갭 7건
2. (15분) 7건 각각 검토 — AIoT 의견
3. (5분) 양방향 검토 — AIDL 미사용 9개
4. (5분) 다음 액션 — 일정 합의

---

**문의:** AI팀 NLU 담당  
**GitHub:** `nsbb/t527-nlu` (`docs/` 폴더)
