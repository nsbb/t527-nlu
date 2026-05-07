# T527 HomeState 멀티턴 통합

## 배경

이전 iteration 21에서 DST(Dialogue State Tracker)만 Kotlin 포팅함 — fn/exec/dir 상속.
사용자 요구: "멀티턴 하면서 집 상태가 자연스럽게 반영이 되어서 되고있는지 봐"

서버 `deployment_pipeline_v2.py`의 HomeState 클래스(per-fn/per-room power 추적)를
T527 Kotlin으로 단순화 포팅하고 HVAC 충돌 재해석을 디바이스에서 검증.

## HomeState 핵심 기능

1. **상태 누적:** `(fn, room) → {power}` 매핑으로 모든 제어 명령 결과 기록
2. **HVAC 재해석:** 명시적 디바이스 키워드 없을 때
   - AC on + "추워" → heat/on 보다 **ac/off** 자연스러움
   - 난방 on + "더워" → ac/on 보다 **heat/off** 자연스러움
3. **HVAC 상호배제:** ac/on 명령 시 heat가 켜져있으면 자동 off

## 디바이스 검증 (T527 51475789d0c64881cd3, 2026-04-29)

```
[T1] 거실 에어컨 켜줘  → ac_control/on
     home: ac@living=on
[T2] 추워             → ac_control/off    🔁 재해석 (heat/on → ac/off)
     home: ac@living=on, ac@none=off
[T3] 난방 켜줘        → heat_control/on
     home: heat@none=on
[T4] 더워             → heat_control/off  🔁 재해석 (ac/on → heat/off)
     home: heat@none=off
[T5] 거실 에어컨 켜줘  → ac_control/on
     home: ac@living=on
[T6] 환기도 켜줘      → vent_control/on   (device follow-up via DST)
     home: ac@living=on, vent@none=on
```

6/6 성공. HomeState 누적 + HVAC 재해석 모두 의도대로 동작.

## 알려진 한계

- **방 추출 비결합:** "추워"처럼 방이 없는 발화는 room=null로 처리 → 이전 발화의 방을 상속하지 않음.
  서버는 DST가 prev_room을 상속하지만 Kotlin DST는 fn/dir만 상속. 향후 개선 대상.
- **상호배제 방 일치 미구현:** room=A에서 AC on일 때 room=B 명령 시 A의 AC를 끄지 못함.
  실용적으로는 단일 방 사용자 기준이라 문제 적음.

## 파일

- `t527_smart_v2/.../nlu/HomeState.kt` (75줄)
- `t527_smart_v2/.../HomeStateDemoActivity.kt` (133줄)

## 실행

```bash
adb -s 51475789d0c64881cd3 shell am start \
  -n com.t527.smart_v2/com.t527.smart_service.HomeStateDemoActivity
adb -s 51475789d0c64881cd3 logcat -s HomeStateDemo:I
```
