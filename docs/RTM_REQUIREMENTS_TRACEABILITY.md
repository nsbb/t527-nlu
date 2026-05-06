# RTM (Requirements Traceability Matrix) — 르엘 시나리오 ↔ NLU ↔ AIDL/REST

> **시나리오 219개 1:1 매핑.** 각 시나리오가 NLU의 어떤 fn으로 분류되고, 어떤 AIDL/REST/외부 API로 처리되는지 추적.  
> CSV 파일: `docs/RTM_REQUIREMENTS_TRACEABILITY.csv` (전체 219행, 엑셀에서 열기 권장)

작성일: 2026-05-06

---

## 1. 상태 범례

| 기호 | 의미 | 시나리오 수 |
|------|------|-----------|
| ✅ | AIDL 매핑 완료 (변경 불필요) | 52 |
| 🟠 | AIDL 보강 요청 (기존 command 확장) | 5 |
| 🔴 | AIDL 신규 요청 (새 command) | 21 |
| 🔵 | 르엘 REST API (이미 정의됨) | 26 |
| 🟢 | 외부 API (Open-Meteo, RSS, 등) | 86 |
| 🟡 | 외부 서비스 (필터 주문 등) | 2 |
| ⚪ | 자체 처리 (FAQ/매뉴얼/시간/알람) | 26 |
| ❓ | 미분류 (명세 확인 필요) | 1 |
| **합계** | | **219** |

---

## 2. 카테고리별 RTM 요약

### 🎛 제어 (63 시나리오, 36 intent)

| intent | 시나리오 수 | NLU fn | AIDL command | 상태 |
|---|---|---|---|---|
| ac_query | 5 | ac_control | remote_access_sysaircon (status) | ✅ |
| ventilation_query | 5 | vent_control | remote_access_ventil (status) | ✅ |
| light_on | 4 | light_control | remote_access_livinglight or _light | ✅ |
| ac_wind | 3 | ac_control | remote_access_sysaircon (wind) | ✅ |
| ac_exception | 3 | ac_control | 고장 안내 (자체) | ⚪ |
| light_query | 2 | light_control | remote_access_livinglight (status) | ✅ |
| **light_schedule** | **2** | **schedule_manage** | **remote_access_schedule(light)** | **🔴 신규** |
| heating_on | 2 | heat_control | remote_access_temper | ✅ |
| heating_up | 2 | heat_control | remote_access_temper | ✅ |
| ac_temp | 2 | ac_control | remote_access_sysaircon (set_temp) | ✅ |
| ac_mode | 2 | ac_control | remote_access_sysaircon (operation_mode) | ✅ |
| ventilation_mode | 2 | vent_control | remote_access_ventil (mode) | ✅ |
| **ventilation_exception** | **2** | **vent_control** | **필터 주문 → 외부** | **🟡 외부** |
| gas_query | 2 | gas_control | remote_access_gas (status) | ✅ |
| gas_close | 2 | gas_control | remote_access_gas | ✅ |
| curtain_open | 2 | curtain_control | remote_access_curtain | ✅ |
| **curtain_schedule** | **2** | **schedule_manage** | **remote_access_schedule(curtain)** | **🔴 신규** |
| light_dim | 1 | light_control | remote_access_dimming_livinglight | ✅ |
| light_brighten | 1 | light_control | remote_access_dimming_livinglight | ✅ |
| heating_query | 1 | heat_control | remote_access_temper (status) | ✅ |
| **heating_schedule_query** | **1** | **schedule_manage** | **remote_access_schedule(temper)** | **🔴 신규** |
| **heating_schedule_cancel** | **1** | **schedule_manage** | **remote_access_schedule(temper)** | **🔴 신규** |
| **heating_schedule_set** | **1** | **schedule_manage** | **remote_access_schedule(temper)** | **🔴 신규** |
| ac_on | 1 | ac_control | remote_access_sysaircon | ✅ |
| ac_off | 1 | ac_control | remote_access_sysaircon | ✅ |
| **ac_mode_schedule** | **1** | **schedule_manage** | **remote_access_schedule(sysaircon)** | **🔴 신규** |
| ac_mode_noroom | 1 | ac_control | remote_access_sysaircon | ✅ |
| ventilation_on | 1 | vent_control | remote_access_ventil | ✅ |
| ventilation_off | 1 | vent_control | remote_access_ventil | ✅ |
| **ventilation_schedule_query** | **1** | **schedule_manage** | **remote_access_schedule(ventil)** | **🔴 신규** |
| **ventilation_schedule_set** | **1** | **schedule_manage** | **remote_access_schedule(ventil)** | **🔴 신규** |
| doorlock_query | 1 | door_control | remote_access_doorlock (status) | ✅ |
| **doorlock_open** | **1** | **door_control** | **remote_access_doorlock (control 추가)** | **🟠 보강#1** |
| curtain_query | 1 | curtain_control | remote_access_curtain (status) | ✅ |
| curtain_close | 1 | curtain_control | remote_access_curtain | ✅ |
| **curtain_stop** | **1** | **curtain_control** | **remote_access_curtain (stop 추가)** | **🟠 보강#2** |

### ⚙ 설정 (22 시나리오, 8 intent)

| intent | 시나리오 수 | NLU fn | AIDL/방법 | 상태 |
|---|---|---|---|---|
| system_faq | 10 | system_meta | 내부 FAQ | ⚪ |
| alarm_query | 4 | home_info | 월패드 자체 알람 | ⚪ |
| system_exception | 2 | system_meta | 시스템 안내 | ⚪ |
| password_change | 2 | system_meta | 월패드 시스템 설정 | 🔵 REST |
| **system_volume_set** | **1** | **system_meta** | **remote_access_volume** | **🔴 신규** |
| **system_brightness_set** | **1** | **system_meta** | **remote_access_brightness** | **🔴 신규** |
| **system_brightness_schedule** | **1** | **schedule_manage** | **remote_access_schedule(brightness)** | **🔴 신규** |
| alarm_set | 1 | home_info | 월패드 자체 알람 | ⚪ |

### 💡 에너지 (10 시나리오, 4 intent)

| intent | 시나리오 수 | NLU fn | AIDL/REST | 상태 |
|---|---|---|---|---|
| energy_usage_query | 5 | energy_query | /v2/ai/meter | 🔵 REST |
| **energy_goal_set** | **3** | **energy_query** | **remote_access_energy_goal** | **🔴 신규** |
| **energy_alert_on** | **1** | **energy_query** | **remote_access_energy_alert** | **🔴 신규** |
| **energy_alert_off** | **1** | **energy_query** | **remote_access_energy_alert** | **🔴 신규** |

### 🛡 방범 (8 시나리오, 3 intent)

| intent | 시나리오 수 | NLU fn | AIDL | 상태 |
|---|---|---|---|---|
| security_query | 3 | security_mode | remote_access_mode (status) | ✅ |
| **security_return_set** | **3** | **security_mode** | **remote_access_mode (return_time 필드)** | **🟠 보강#3** |
| security_activate | 2 | security_mode | remote_access_mode | ✅ |

### 🌐 부가 (94 시나리오, 23 intent)

| intent | 시나리오 수 | 채널 | 상태 |
|---|---|---|---|
| news_play | 13 | RSS | 🟢 외부 |
| weather_info | 12 | Open-Meteo | 🟢 외부 |
| medical_search | 10 | 병원 API | 🟢 외부 |
| traffic_route_query | 7 | 경로 API | 🟢 외부 |
| weather_activity | 4 | Open-Meteo + judge | 🟢 외부 |
| traffic_bus_query | 4 | 버스 API | 🟢 외부 |
| fuel_price_query | 4 | 유가 API | 🟢 외부 |
| stock_index_query | 4 | 지수 API | 🟢 외부 |
| **emergency** | **3** | **AIDL** | **🔴 신규-emergency** |
| complex_info | 3 | 단지 정보 (REST) | 🔵 |
| dust_query | 3 | Open-Meteo Air | 🟢 외부 |
| weather_clothing | 3 | Open-Meteo | 🟢 외부 |
| medical_exception | 3 | 증상→병원 매핑 | 🟢 외부 |
| fuel_exception | 3 | 유가 할인 | 🟢 외부 |
| stock_price_query | 3 | 종목 가격 | 🟢 외부 |
| stock_exception | 3 | 추천 | 🟢 외부 |
| news_info | 2 | RSS 메타 | 🟢 외부 |
| weather_exception | 2 | 강수 정밀 | 🟢 외부 |
| medical_hours | 2 | 진료시간 | 🟢 외부 |
| fuel_station_search | 2 | 주유소 | 🟢 외부 |
| community_query | 2 | 단지 시설 | 🔵 REST |
| news_exception | 1 | RSS 카테고리 | 🟢 외부 |
| traffic_exception | 1 | 택시 호출 | 🟢 외부 |

### 🚗 더보기 (13 시나리오, 8 intent)

| intent | 시나리오 수 | REST 엔드포인트 | 상태 |
|---|---|---|---|
| ev_charging | 4 | /v2/ai/eleccar | 🔵 REST |
| notice_query | 2 | /v2/ai/announcements | 🔵 REST |
| visitor_parking_register | 2 | /v2/ai/cars/guest | 🔵 REST |
| elevator_query | 1 | 엘리베이터 위치 | 🔵 REST |
| elevator_call | 1 | /v2/ai/elevator/{dir} | 🔵 REST |
| car_history_query | 1 | /v2/ai/cars/parkinfo | 🔵 REST |
| car_history_delete | 1 | /v2/ai/cars/parkinfo | 🔵 REST |
| visitor_parking_query | 1 | /v2/ai/cars/guest | 🔵 REST |

### 📋 메인 (8 시나리오, 8 intent)

| intent | 시나리오 수 | 처리 | 상태 |
|---|---|---|---|
| home_status_query | 1 | remote_get_home_total_device | ✅ |
| notification_query | 1 | /v2/ai/announcements | 🔵 REST |
| time_query | 1 | 월패드 자체 | ⚪ |
| manual_capability | 1 | 내부 FAQ | ⚪ |
| manual_usage | 1 | 내부 FAQ | ⚪ |
| manual_creator | 1 | 내부 FAQ | ⚪ |
| manual_name | 1 | 내부 FAQ | ⚪ |
| manual_unsupported | 1 | 내부 FAQ | ⚪ |

### 📞 통화 (1 시나리오)

| intent | 시나리오 수 | 처리 | 상태 |
|---|---|---|---|
| door_open | 1 | ?? (AIDL doorlock과 다른가?) | ❓ 미분류 |

---

## 3. 갭 핵심 정리 — AIoT 보강 7건

| # | 변경 내용 | 영향 시나리오 | intent |
|---|---|---|---|
| 1 | 🟠 doorlock control(open) 추가 | 1 | doorlock_open |
| 2 | 🟠 curtain stop ctrl_action 추가 | 1 | curtain_stop |
| 3 | 🟠 mode return_time 필드 추가 | 3 | security_return_set |
| 4 | 🔴 schedule 통합 command 신규 | 11 | heating_schedule_×3, ac_mode_schedule, ventilation_schedule_×2, curtain_schedule, light_schedule, system_brightness_schedule |
| 5 | 🔴 volume + brightness 신규 | 2 | system_volume_set, system_brightness_set |
| 6 | 🔴 energy_alert + energy_goal 신규 | 5 | energy_alert_on/off, energy_goal_set |
| 7 | 🔴 emergency 신규 | 3 | emergency |

**합계 28 시나리오 = 13%**가 AIoT 보강 7건에 의존.

---

## 4. AIDL 미사용 9개 — 음성 시나리오 추가 검토

| AIDL command | 르엘 시나리오 | 권장 |
|---|---|---|
| `remote_access_cooktop` | 0 | 추가 ("쿡탑 꺼졌어?") |
| `remote_access_hybridcooktop` | 0 | 추가 |
| `remote_access_sensor` | 0 (간접) | 추가 ("공기질 어때?") |
| `remote_access_curtain2` | 0 | 검토 |
| `remote_access_louver` | 0 | 검토 |
| `remote_access_electric` | 0 | 검토 ("콘센트 꺼") |
| `remote_access_multivent` | 0 | 검토 ("안방 환기") |
| `remote_access_smartlight` | 0 | 시스템 전용? |
| `remote_get_home_devices` | 0 | 시스템 전용? |

---

## 5. 사용 방법

### 갭 추적 (AIoT 회신 시 업데이트)

각 항목 상태:
```
🔴 신규-schedule  → 🟡 검토중 (AIoT 회신 받음)
                  → 🟠 구현중
                  → ✅ 완료
```

### 새 시나리오 추가 시

기획팀이 시나리오 추가하면:
1. 새 행을 RTM CSV에 추가
2. NLU fn 매핑 결정
3. AIDL/REST/외부 채널 결정
4. 상태 표시

### CSV로 보기

```
docs/RTM_REQUIREMENTS_TRACEABILITY.csv
열: 번호 | 구분 | intent | 발화 | NLU fn | AIDL/REST | 채널 | 상태
```

엑셀에서 열어서 필터/정렬/색상 처리 가능.

---

## 부록: 채널별 시나리오 분포 시각화

```
AIDL (80개, 37%)        ████████████░░░░░░░░░░░░░░░░░░░░░░  
  ├─ ✅ 매핑됨 52         █████████░░░░░░░░░░░░░░░░░░░░░░░░░░  
  ├─ 🟠 보강 5            █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  
  └─ 🔴 신규 21           ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  
🟢 외부 API   86개, 39%  █████████████░░░░░░░░░░░░░░░░░░░░░░  
🔵 르엘 REST  26개, 12%  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  
⚪ 자체 처리  26개, 12%  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  
🟡 외부 서비스 2개         ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  
❓ 미분류     1개          ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  
```
