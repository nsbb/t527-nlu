# T527 Response Generator (Kotlin)

작성: 2026-05-07

## 개요

Python `scripts/response_generator_v2.py` (2418줄)의 핵심 응답 템플릿을 Kotlin으로 포팅.
NLU 결과 + AIDL 응답 → 사용자 한국어 응답 문자열 생성.

## 응답 템플릿 (요약)

### 제어 (control_then_confirm)
```
on:    "네, [방] [기기]을(를) 켭니다."
off:   "네, [방] [기기]을(를) 끕니다."
open:  "네, [방] [기기]을(를) 엽니다."
close: "네, [방] [기기]을(를) 닫습니다."
up:    "네, [방] [조명/난방/에어컨/환기] [강도|온도|풍량] 올립니다."
down:  (대칭)
set:   "네, [방] [기기]을(를) [N도/단계]로 설정합니다."
```

### 조회 (query_then_respond)
```
"현재 [방] [기기] 상태를 확인합니다."
```

### 직접 응답 (direct_respond)
```
weather: "날씨 정보를 확인합니다."
news:    "뉴스를 들려드릴게요."
unknown: "지원하지 않는 기능입니다."
```

### 판단형 (query_then_judge) / Clarify
```
clarify: "어떤 [기기]를 말씀하시는지 알려주시겠어요?"
judge:   "오늘 상황을 고려할 때 적절해 보입니다."
```

## AIDL 응답 우선

AidlMockClient (또는 실 AIDL) 응답의 `unit_status`가 사용자 의도와 다를 수 있음 (예: 일괄소등 on 요청 → off 응답). 이때 응답은 **실 status 기준**:

```kotlin
ResponseGenerator.generate(
    intent = nluResult,
    room = "living",
    aidlStatus = "off"  // AIDL이 실제 적용한 상태
)
```

→ "네, 거실 조명을 끕니다." (요청과 다른 결과 반영)

## End-to-End 통합

```
사용자: "거실 불 켜줘"
  ↓
Preprocess → "거실 불 켜줘"
  ↓
NLU → light_control / control_then_confirm / on
  ↓
IntentRouter → AIDL remote_access_livinglight (action=control, ctrl_action=on)
  ↓
AidlMockClient → {unit_status: "on"}
  ↓
ResponseGenerator → "네, 거실 조명을 켭니다."
  ↓ (TTS 또는 화면)
사용자에게 응답
```

## 산출물

```
src/main/kotlin/com/t527/smart_service/integration/
  └─ ResponseGenerator.kt    (~140줄, 핵심 템플릿)
```

## 한계 / 다음

- 현재 templates 단순 (날씨/뉴스 본문은 외부 API 응답 합쳐야)
- TTS 연동 안 됨 (응답 문자열만 생성)
- 복합 명령 (다중 fn) 처리 미지원
- emergency / hearsay 등 특수 케이스 단순화

다음 단계:
1. TTS Player 연동 (`assets/tts/` 미리 녹음된 MP3 또는 Android TTS API)
2. VoiceAiService에 NLU+IntentRouter+ResponseGenerator 체인 통합
3. 외부 API (날씨/뉴스) 호출 모듈 추가
