# T527 Multi-turn DST (Dialogue State Tracker)

## 개요

Python `scripts/dialogue_state_tracker.py` (521줄) 을 Kotlin으로 단순화 포팅.
T527 디바이스에서 멀티턴 대화 컨텍스트를 추적해 fn/exec/dir 자동 상속.

## 처리 패턴

| 패턴 | 예시 | 동작 |
|---|---|---|
| Confirm | "응", "그래" | 이전 턴 fn/exec/dir 그대로 반복 |
| Room follow-up | "안방도", "거실" | prev_fn 상속 + room 변경 |
| Device follow-up | "에어컨도", "난방도" | fn 새 device, exec/dir 상속 |
| Bare 발화 | "더", "조금만 켜줘" | fn 상속 + dir 추론 |
| 강도 only | "세게 틀어줘" | prev_fn 상속, exec=control_then_confirm |

Timeout: 10초 후 prev 상태 만료.

## 디바이스 검증 결과 (T527 51475789d0c64881cd3, 2026-04-29)

7-turn 시나리오 — `MultiTurnDemoActivity`:

```
[T1] 거실 불 켜줘      → light_control/on   → 네, 조명을 켭니다.
[T2] 안방도            → light_control/on   → 네, 조명을 켭니다.       (room follow-up 상속)
[T3] 에어컨도 켜줘     → ac_control/on      → 네, 에어컨을 켭니다.     (device follow-up)
[T4] 더 시원하게       → ac_control/up      → 네, 에어컨을 켭니다.     (bare → fn 상속, dir=up)
[T5] 환기도 켜줘       → vent_control/on    → 네, 환기를 켭니다.       (device follow-up)
[T6] 주방 가스 잠가줘  → gas_control/close  → 네, 가스밸브를 닫습니다.
[T7] 응                → gas_control/close  → 네, 가스밸브를 닫습니다. (confirm 반복)
```

7/7 성공. 모든 follow-up 발화에서 fn/exec/dir이 의도대로 상속됨.

## 파일

- `t527_smart_v2/app/src/main/kotlin/com/t527/smart_service/nlu/DialogueStateTracker.kt` (137줄)
- `t527_smart_v2/app/src/main/kotlin/com/t527/smart_service/MultiTurnDemoActivity.kt` (158줄)
- `AndroidManifest.xml` — `MultiTurnDemoActivity` 등록

## 실행

```bash
adb -s 51475789d0c64881cd3 shell am start -n com.t527.smart_v2/com.t527.smart_service.MultiTurnDemoActivity
adb -s 51475789d0c64881cd3 logcat -s MultiTurn:I
```
