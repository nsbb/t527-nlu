# a527_wallpad 원격 제어 API 매핑 (2026-05-08)

## 출처

- 레포: `https://bitbucket.org/hdclabs/a527_wallpad`
- 로컬: `/home/nsbb/claude/a527_wallpad/`
- 핵심 파일: **`wallpadserver/src/main/java/com/hdclabs/wallpadserver/declare/iMapDefine.java:132-162`**

## 원격 제어 enum 전체 (iMapDefine.java)

`mIdx` 정수 ID를 통해 wallpad 서버에 명령 전달. NLU에서 fn 분류 결과를 이 ID로 매핑해야 AIDL 호출 가능.

| ID | enum 상수 | 식별자 (mContent) | 비고 |
|---|---|---|---|
| 39 | REMOTE_REGISTER_NEW_DEVICE | remote_register_new_device | 디바이스 등록 |
| 40 | REMOTE_REGISTER_NEW_DEVICE_AUTH_NUMBER | remote_register_new_device_auth_number | 등록 인증 |
| 41 | REMOTE_DELETE_DEVICE | remote_delete_device | 디바이스 삭제 |
| 42 | **REMOTE_GET_HOME_DEVICES** | remote_get_home_devices | 가구 디바이스 조회 |
| 43 | REMOTE_CHECK_REGISTER_DEVICE | remote_check_register_device | 등록 확인 |
| 44 | **REMOTE_GET_HOME_TOTAL_DEVICE** | remote_get_home_total_device | 전체 디바이스 조회 |
| 45 | REMOTE_ENTIRE_ACCESS_LIGHT_CONTROL | remote_entire_access_light_control | 전체 조명 일괄 |
| 46 | REMOTE_DEVICE_EVENT | remote_device_event | 디바이스 이벤트 |
| **47** | **REMOTE_ACCESS_GAS** | remote_access_gas | 가스 |
| **48** | **REMOTE_ACCESS_SMARTLIGHT** | remote_access_smartlight | 스마트조명 |
| **49** | **REMOTE_ACCESS_LIVINGLIGHT** | remote_access_livinglight | 거실조명 |
| **50** | **REMOTE_ACCESS_DIMMING_LIVINGLIGHT** | remote_access_dimming_livinglight | 디밍 거실조명 |
| **51** | **REMOTE_ACCESS_ELECTRIC** | remote_access_electric | 전기/콘센트 |
| **52** | **REMOTE_ACCESS_DOORLOCK** | remote_access_doorlock | 도어락 |
| **53** | **REMOTE_ACCESS_TEMPER** | remote_access_temper | 난방 (temperature controller) |
| **54** | **REMOTE_ACCESS_MODE** | remote_access_mode | 모드 (외출/재택/취침 등) |
| **55** | **REMOTE_ACCESS_VENT** | remote_access_ventil | 환기 |
| **56** | **REMOTE_ACCESS_MULTIVENT** | remote_access_multivent | 다중환기 |
| **57** | **REMOTE_ACCESS_CURTAIN** | remote_access_curtain | 커튼 |
| **58** | **REMOTE_ACCESS_LIGHT** | remote_access_light | 일반 조명 |
| **59** | **REMOTE_ACCESS_LIGHT_BATCH_CONTROL** | remote_access_light_batch_control | 조명 일괄제어 |
| **60** | **REMOTE_ACCESS_ONEPASS** | remote_access_onepass | 원패스 (출입통제) |
| **61** | **REMOTE_ACCESS_COOKTOP** | remote_access_cooktop | 쿡탑 |
| **62** | **REMOTE_ACCESS_HYBRIDCOOKTOP** | remote_access_hybridcooktop | 하이브리드 쿡탑 |
| **63** | **REMOTE_ACCESS_SYSTEM_AIRCON** | remote_access_sysaircon | 시스템 에어컨 |
| **71** | **REMOTE_ACCESS_LOUVER** | remote_access_louver | 루버 (블라인드) |
| **72** | **REMOTE_ACCESS_SHADING** | **remote_access_curtain2** | 차양/롤스크린 (식별자만 curtain2) |

## NLU fn 클래스 → wallpad 원격 제어 ID 매핑

NLU 모델은 20개 `fn` 클래스로 분류. 각 fn이 어떤 wallpad 원격 제어로 가는지:

| NLU fn | wallpad enum (ID) | 비고 |
|---|---|---|
| `light_control` | 사용자 발화에 따라 분기 | |
| └ "거실 불" | REMOTE_ACCESS_LIVINGLIGHT (49) 또는 DIMMING(50) | 디밍 가능 여부에 따라 |
| └ "조명 전체" / "전등 다" | REMOTE_ACCESS_LIGHT_BATCH_CONTROL (59) | |
| └ 기타 일반 등 | REMOTE_ACCESS_LIGHT (58) | |
| └ "스마트 조명" | REMOTE_ACCESS_SMARTLIGHT (48) | |
| `heat_control` | REMOTE_ACCESS_TEMPER (53) | 난방기/온도 제어 |
| `ac_control` | REMOTE_ACCESS_SYSTEM_AIRCON (63) | 시스템 에어컨 |
| `vent_control` | REMOTE_ACCESS_VENT (55) 또는 MULTIVENT (56) | 일반/다중환기 |
| `gas_control` | REMOTE_ACCESS_GAS (47) | |
| `door_control` | REMOTE_ACCESS_DOORLOCK (52) 또는 ONEPASS (60) | |
| `curtain_control` | REMOTE_ACCESS_CURTAIN (57) / LOUVER (71) / SHADING (72) | curtain/blind/shade에 따라 분기 |
| `elevator_call` | (별도 API) | 엘리베이터는 wallpad 원격 X — 다른 채널 |
| `security_mode` | REMOTE_ACCESS_MODE (54) | 외출/재택/취침/방범 |
| `home_info` | REMOTE_GET_HOME_DEVICES (42) / REMOTE_GET_HOME_TOTAL_DEVICE (44) | 상태 조회 |
| `energy_query` | REMOTE_ACCESS_ELECTRIC (51) | 전기 사용량 |
| `weather_query` / `news_query` / `traffic_query` | (외부 API) | 외부 채널 |
| `schedule_manage` | (별도) | |
| `system_meta` | (시스템 자체 응답) | |

## 사용자 나열 vs 실제 발견 비교

| 사용자가 댄 함수 | 발견 | 매핑 |
|---|---|---|
| remote_get_home_devices | ✅ | ID 42 |
| remote_get_home_total_device | ✅ | ID 44 |
| remote_access_livinglight | ✅ | ID 49 |
| remote_access_light | ✅ | ID 58 |
| remote_access_smartlight | ✅ | ID 48 |
| remote_access_dimming_livinglight | ✅ | ID 50 |
| remote_access_light_batch_control | ✅ | ID 59 |
| remote_access_gas | ✅ | ID 47 |
| remote_access_doorlock | ✅ | ID 52 |
| remote_access_temper | ✅ | ID 53 |
| remote_access_mode | ✅ | ID 54 |
| **remote_access_sensor** | **❌** | iMapDefine에 없음 — 별도 API |
| remote_access_electric | ✅ | ID 51 |
| remote_access_ventil | ✅ | ID 55 |
| remote_access_multivent | ✅ | ID 56 |
| remote_access_sysaircon | ✅ | ID 63 |
| remote_access_curtain | ✅ | ID 57 |
| **remote_access_curtain2** | **✅** | **ID 72 (REMOTE_ACCESS_SHADING enum)** |
| remote_access_louver | ✅ | ID 71 |
| remote_access_hybridcooktop | ✅ | ID 62 |

**추가 발견 (사용자 목록 외):**
- remote_access_cooktop (61) — 일반 쿡탑
- remote_access_onepass (60) — 출입통제
- remote_entire_access_light_control (45) — 전체 조명 일괄
- remote_register_new_device / remote_delete_device / remote_check_register_device

## sensor가 enum에 없는 이유

`grep` 결과 `iMapDefine.java`엔 `remote_access_sensor` 매핑 없음. 그러나 코드엔 센서 사용 플래그 존재:
- `ServerMainService.java:677`: `isEnvMultiSensorUsage` (환경 다중센서 사용 여부)
- 환경/대기질 센서는 이 enum 체계 밖에서 처리됨 — 별도 API 채널로 추정

## 참고 파일

| 파일 | 역할 |
|---|---|
| `wallpadserver/.../declare/iMapDefine.java` | 모든 enum 정의 (132~162줄) |
| `wallpadserver/.../ServerMainService.java` | 메인 서비스 |
| `wallpadux/.../MainActivity.java:368` | UI 측 REMOTE_ACCESS_LIGHT (값 360) — 별도 핸들러 ID. iMapDefine과 다름! |
| `WallpadAIDL/src/main/aidl/.../IWallpadServer.aidl` | AIDL 인터페이스 (스켈레톤만) |

## 다음 단계

1. **NLU IntentRouter 매핑 갱신** — `t527_smart_v2/.../IntentRouter.kt`의 AIDL 채널이 위 enum ID 정확히 사용하는지 검증
2. **AidlMockClient 보완** — 27개 enum 모두 시뮬레이션
3. **light_control 분기 룰** — "거실"이면 LIVINGLIGHT(49), "전체"면 BATCH(59), 일반이면 LIGHT(58)
4. **curtain 분기 룰** — "커튼"=CURTAIN(57), "블라인드/루버"=LOUVER(71), "차양/롤스크린"=SHADING(72)
5. **sensor 처리 채널 별도 조사** — wallpad 서버 sensor API 위치 파악
