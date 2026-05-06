# AIDL ↔ 르엘 시나리오 갭 분석 및 AIDL 보강 요청서

> **르엘 시나리오 219개 = GT (Ground Truth)** 기준  
> Multi-Head NLU (5 heads) 출력을 AIDL 호출로 변환할 때, 현재 AIDL 정의로 커버 불가능한 항목과 보강 요청 사항.

작성일: 2026-05-06

---

## 1. 배경

### 우리 NLU 아키텍처 (Multi-Head)

```
사용자 발화
    ↓
KoELECTRA + CNN 5-head 분류
    ↓
┌─────────┬──────────┬───────────┬──────────┬────────┐
│   fn    │exec_type │param_dir  │param_type│ judge  │
│  (20)   │   (5)    │   (9)     │   (5)    │  (5)   │
└─────────┴──────────┴───────────┴──────────┴────────┘
    ↓
[Intent Router]
    ↓
AIDL 호출 (또는 REST/외부 API)
```

**5 head 출력 라벨:**
- **fn**: light/heat/ac/vent/gas/door/curtain/elevator/security/schedule/weather/news/traffic/energy/home_info/system_meta/market/medical/vehicle/unknown
- **exec_type**: query_then_respond / control_then_confirm / query_then_judge / direct_respond / clarify
- **param_direction**: none / up / down / set / on / off / open / close / stop
- **param_type**: none / temperature / brightness / mode / speed
- **judge**: none / outdoor_activity / clothing / air_quality / cost_trend

### 핵심 원칙

> 르엘 시나리오는 **변경 불가**. AIDL이 르엘에 맞춰 확장되어야 함.

---

## 2. 르엘 ↔ AIDL 좌우 비교 (있는 것 / 없는 것)

### 2-1. 카테고리별 좌우 비교 표

| 카테고리 | 르엘 시나리오 (intent / 발화 수) | AIDL command | 갭 |
|---|---|---|---|
| **조명 (Light)** | light_on / off / query / dim / brighten / brightness_up/down / batch_off / schedule (15개 시나리오) | livinglight, light, smartlight, dimming_livinglight, light_batch_control (5종) | ⚠ schedule 신규 필요 |
| **에어컨 (AC)** | ac_on / off / query / temp / mode / mode_noroom / mode_schedule / wind / wind_up / wind_down / temperature_up/down / exception (13개) | sysaircon (1종, 모든 동작 통합) | ⚠ schedule(mode_schedule) 신규 필요 |
| **난방 (Heating)** | heat_on / off / query / set_temp / temperature_up/down / outmode / schedule_set/query/cancel / heating_up (10개) | temper (1종) | ⚠ outmode 필드 + schedule 신규 필요 |
| **환기 (Vent)** | ventilation_on / off / query / mode / speed_up/down / schedule_set/query / exception (10개) | ventil + multivent (2종) | ⚠ schedule 신규 필요 |
| **가스 (Gas)** | gas_close / query (4개) | gas (1종) | ✅ 충분 |
| **도어락 (Doorlock)** | doorlock_query / doorlock_open (2개) | doorlock (status only) | ⚠ control(open) action 누락 |
| **커튼 (Curtain)** | curtain_open / close / query / stop / schedule (8개) | curtain, curtain2, louver (3종) | ⚠ stop action + schedule 신규 필요 |
| **방범 (Security)** | security_query / activate / return_set (8개) | mode (1종) | ⚠ return_set 보강/신규 필요 |
| **에너지 (Energy)** | usage_query / goal_set / alert_on/off (10개) | (없음) | ⚠ alert/goal 신규 필요 + REST 사용 |
| **월패드 자체** | volume_set / brightness_set / brightness_schedule (3개) | (없음) | ⚠ volume/brightness 신규 필요 |
| **비상 (Emergency)** | emergency (3개) | (없음) | ⚠ emergency 신규 필요 |
| **상태 조회** | home_status_query (1개) | get_home_total_device (1종) | ✅ 충분 |
| **인덕션/쿡탑** | (없음) | cooktop / hybridcooktop (2종) | 🔵 음성 시나리오 추가 검토 |
| **콘센트** | (없음) | electric (1종) | 🔵 음성 시나리오 추가 검토 |
| **공기질 센서** | dust_query (3개, 외부 API로 처리) | sensor (1종) | 🔵 센서 직접 조회 시나리오 검토 |
| **단지 부가서비스** | elevator / cars / ev / announcements (13개) | (영역 외) | 🟡 르엘 REST API로 처리 |
| **외부 정보** | weather / news / traffic / fuel / stock / medical (94개) | (영역 외) | 🟢 외부 API |
| **자체 처리** | manual / faq / time / alarm (26개) | (영역 외) | 🟢 월패드 앱 내부 |

**범례:**
- ✅ AIDL 충분 / 변경 불필요
- ⚠ AIDL 보강·신규 필요 (르엘 시나리오 커버 위해)
- 🔵 르엘에 시나리오 없음 (음성 시나리오 추가 검토)
- 🟡 AIDL 영역 외, 르엘 REST 사용
- 🟢 AIDL 영역 외, 외부 API

### 2-2. 정량 요약

```
르엘 219 시나리오 처리 채널 분포:
├─ AIDL 처리:     80개 (37%)
│   ├─ 기존 AIDL로 OK:    52개
│   ├─ AIDL 보강 필요:    2개
│   └─ AIDL 신규 필요:    26개
├─ 르엘 REST API:  26개 (12%)
├─ 외부 API:        83개 (38%)
├─ 자체 처리:       26개 (12%)
├─ 분류 모호:        1개 (door_open)
└─ 통화:            1개

AIDL 21 command 활용 현황:
├─ 르엘 시나리오에서 사용:  12개 (57%)
└─ 사용 안 됨:               9개 (43%)
    cooktop, hybridcooktop, curtain2, louver, electric,
    multivent, sensor, smartlight, get_home_devices
```

---

## 3. AIDL 보강·추가 요청 — 7건

각 항목에 **NLU 5 head 출력 → AIDL 호출 매핑**도 함께 명시.

### 3-1. 🔴 보강 #1: `remote_access_doorlock` control 액션 추가

**현재 AIDL:** status만 정의 (조회 전용)  
**르엘 시나리오:** "현관문 열어줘", "도어락 열어줘"

**Multi-Head 출력 예시:**
```json
{"fn": "door_control", "exec_type": "control_then_confirm",
 "param_direction": "open", "param_type": "none"}
```

**요청 AIDL:**
```json
{
  "command": "remote_access_doorlock",
  "action": "control",
  "dev_num": "1",
  "unit_num": "doorlock1",
  "ctrl_action": "open"
}
```

---

### 3-2. 🔴 보강 #2: `remote_access_curtain` stop 액션 추가

**현재 AIDL:** ctrl_action open/close만  
**르엘 시나리오:** "커튼 열기 멈춰줘"

**Multi-Head 출력:**
```json
{"fn": "curtain_control", "param_direction": "stop"}
```

**요청 AIDL:**
```json
{
  "command": "remote_access_curtain",
  "action": "control",
  "ctrl_action": "stop"
}
```

---

### 3-3. 🔴 신규 #1: `remote_access_schedule_*` (예약 통합)

**대상 르엘 시나리오 (8개):**
- heating_schedule_query / set / cancel
- ac_mode_schedule
- ventilation_schedule_query / set
- curtain_schedule
- light_schedule

**Multi-Head 출력:**
```json
{"fn": "schedule_manage", "exec_type": "control_then_confirm",
 "param_direction": "set", "param_type": "mode"}
```

**제안 AIDL (통합 schedule command 1개):**
```json
{
  "command": "remote_access_schedule",
  "action": "control",
  "device": "temper",          // 대상 기기 (temper/sysaircon/ventil/curtain/light/brightness)
  "schedule_id": "1",
  "operation": "create",       // create / update / delete / list
  "start_time": "10:00",
  "end_time": "16:00",
  "days": ["mon","tue","wed","thu","fri"],
  "params": {                  // 기기별 제어 파라미터
    "ctrl_action": "outmode",
    "set_temp": "20"
  }
}
```

---

### 3-4. 🔴 신규 #2: `remote_access_volume`, `remote_access_brightness`

**대상 르엘 시나리오 (3개):** system_volume_set, system_brightness_set, system_brightness_schedule

**Multi-Head 출력:**
```json
{"fn": "system_meta", "exec_type": "control_then_confirm",
 "param_direction": "up", "param_type": "brightness"}
```

**제안 AIDL:**
```json
// 볼륨
{"command": "remote_access_volume", "action": "control",
 "ctrl_action": "set", "level": "10"}

// 밝기
{"command": "remote_access_brightness", "action": "control",
 "ctrl_action": "set", "level": "30"}

// 밝기 예약은 schedule command 사용 (3-3)
```

---

### 3-5. 🔴 신규 #3: `remote_access_energy_alert`, `remote_access_energy_goal`

**대상 르엘 시나리오 (4개):** energy_alert_on/off, energy_goal_set

**Multi-Head 출력:**
```json
{"fn": "energy_query", "exec_type": "control_then_confirm",
 "param_direction": "on" or "off" or "set"}
```

**제안 AIDL:**
```json
// 알림
{"command": "remote_access_energy_alert", "action": "control",
 "ctrl_action": "on", "threshold": "500"}

// 목표 설정
{"command": "remote_access_energy_goal", "action": "control",
 "category": "electric", "goal_kwh": "300"}
```

---

### 3-6. 🔴 신규 #4: `remote_access_emergency` (복합 비상 제어)

**대상 르엘 시나리오 (3개):** emergency ("화재야", "비상상황", "침입자")

**Multi-Head 출력:**
```json
{"fn": "security_mode", "exec_type": "control_then_confirm",
 "param_direction": "on", "judge": "none"}
```

**제안 AIDL:**
```json
{
  "command": "remote_access_emergency",
  "action": "control",
  "emergency_type": "fire" | "intrusion" | "panic" | "general",
  "auto_actions": [
    "gas_close",
    "doorlock_lock",
    "notify_security",
    "light_all_on"
  ]
}
```

내부적으로 여러 AIDL 호출을 chain 실행. 응답에는 각 동작 결과 array 반환.

---

### 3-7. 🔴 보강 #3: `remote_access_mode` return_time 필드 (또는 schedule_mode)

**대상 르엘 시나리오 (3개):** security_return_set ("외출 설정 복귀 시간")

**옵션 A — 기존 mode에 필드 추가:**
```json
{"command": "remote_access_mode", "action": "control",
 "ctrl_action": "away", "return_time": "18:00"}
```

**옵션 B — schedule command 활용 (3-3과 통합):**
```json
{"command": "remote_access_schedule", "device": "mode",
 "operation": "create", "start_time": "09:00", "end_time": "18:00",
 "params": {"ctrl_action": "away"}}
```

권장: **옵션 B (schedule 통합)**. 일관성 ↑.

---

## 4. Multi-Head NLU 모델 학습 영향

### 4-1. 신규 fn 추가 필요 여부

현재 fn 20종 중 다음 매핑 사용:
| AIDL 추가 항목 | 사용 fn | 학습 데이터 |
|---|---|---|
| schedule_* | `schedule_manage` ✅ 이미 존재 | 르엘 시나리오 14개 + 변형 합성 |
| volume / brightness | `system_meta` ✅ | 르엘 3개 + 변형 |
| energy_alert / goal | `energy_query` ✅ | 르엘 4개 + 변형 |
| emergency | `security_mode` 또는 신규 `emergency` | 신규 fn 카테고리 검토 필요 |
| doorlock / curtain stop | 기존 `door_control` / `curtain_control` ✅ | 변형만 추가 |

**emergency만 신규 fn 카테고리 추가 검토 필요** (현재는 `security_mode/on`으로 처리하지만 의미가 다름).

### 4-2. param_direction에 추가 필요한 값

현재 9개로 충분하지만, **schedule operation**은 별도 필드로 처리해야 함 (param_direction에 set만 사용 + 추가 필드).

### 4-3. param_type 추가 검토

현재 5종(temperature/brightness/mode/speed). 다음 새 필요:
- `level` (볼륨/밝기 단계)
- `time` (예약 시각)
- `enum` (operation_mode 종류)

→ 모델 재학습 시 보강 권장.

---

## 5. 구현 우선순위

| 순위 | 항목 | 이유 | 난이도 |
|------|------|------|--------|
| 🔥 1 | doorlock control(open) | 단순 보강, 즉시 가능 | 낮음 |
| 🔥 2 | curtain stop | 단순 보강 | 낮음 |
| 🟧 3 | volume / brightness | 사용자 빈번 요청 | 중간 |
| 🟧 4 | schedule_* 통합 | 시나리오 14개 커버 | 높음 (설계 필요) |
| 🟨 5 | energy_alert / goal | 4 시나리오 | 중간 |
| 🟨 6 | emergency | 안전 관련 | 높음 (복합 동작) |
| 🟦 7 | mode return_time | schedule과 통합 가능 | 낮음 |

---

## 6. 결론 및 다음 단계

### AIDL 측 변경 (기획팀 요청)
- **보강 2건** (doorlock/curtain) — 즉시 처리 가능
- **신규 5건** (schedule, volume/brightness, energy, emergency, return_time)
- 처리 후 **르엘 시나리오 80개 (37%) 모두 AIDL 호출 가능**

### 르엘 측 검토 (선택)
- AIDL의 9개 미사용 command 중 일부는 음성 시나리오 추가 검토 가치 있음
  - 우선: cooktop, sensor (공기질 직접 조회)
  - 차순: curtain2, electric

### NLU 모델 측 작업
- **emergency** 처리: security_mode/on로 처리하거나 신규 fn 카테고리 추가
- **param_type 확장**: level/time/enum 추가 학습
- **schedule_manage** intent 변형 데이터 합성 후 재학습 (현재 데이터 소량)

### 통합 테스트
- AIDL Mock Client 작성 → end-to-end 테스트 (디바이스 없이 검증)
- 실 디바이스(T527 + 월패드 서버) 통합 시 회귀 검증

---

## 부록: 현재 NLU 모델 라벨 전체 목록

```
fn (20):
  light_control, heat_control, ac_control, vent_control, gas_control,
  door_control, curtain_control, elevator_call, security_mode,
  schedule_manage, weather_query, news_query, traffic_query,
  energy_query, home_info, system_meta, market_query, medical_query,
  vehicle_manage, unknown

exec_type (5):
  query_then_respond, control_then_confirm, query_then_judge,
  direct_respond, clarify

param_direction (9):
  none, up, down, set, on, off, open, close, stop

param_type (5):
  none, temperature, brightness, mode, speed

judge (5):
  none, outdoor_activity, clothing, air_quality, cost_trend
```
