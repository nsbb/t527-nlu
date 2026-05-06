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

## 한국어 받침 처리 (2026-05-07 추가)

```kotlin
private fun josa(word: String, withFinal: String, withoutFinal: String): String {
    val last = word.last()
    if (last !in '가'..'힣') return withoutFinal
    val code = last.code - 0xAC00
    val finalConsonant = code % 28
    return if (finalConsonant == 0) withoutFinal else withFinal
}
```

적용 결과:
```
조명 + 을(받침 ㅇ) → 조명을
에어컨 + 을(받침 ㄴ) → 에어컨을
환기 + 를(받침 X) → 환기를
커튼 + 을(받침 ㄴ) → 커튼을
보안 모드 + 를(받침 X) → 보안 모드를
```

## TTS 출력 (MP3 + Android TTS)

T527 디바이스에는 Android TextToSpeech 엔진 없음. 대신 `assets/tts/` 미리 녹음된 98개 MP3 사용:

```
tts/light_control_on.mp3   ← "조명을 켭니다."
tts/ac_control_on.mp3      ← "에어컨을 켭니다."
tts/vent_control_on.mp3    ← "환기를 켭니다."
tts/curtain_control_close.mp3 ← "커튼을 닫습니다."
... 총 98개
```

`mapIntentToTtsKey(fn, dir)` 매핑으로 intent → ttsKey → MP3 파일 재생.
파일 없으면 텍스트만 출력 (fallback).

## End-to-End 음성 응답 체인

```
사용자 발화
  ↓
Preprocess → NLU → PostRules → IntentRouter → AIDL Mock
  ↓
ResponseGenerator (한국어 텍스트 + 받침 처리)
  ↓
TtsPlayer (assets/tts/MP3 재생)
  ↓
사용자 음성 응답
```

13개 시나리오 검증:
- MP3 재생: 4건 (조명/에어컨/환기/커튼)
- Fallback (텍스트만): 9건 (해당 mp3 없는 fn/dir 조합)
