# T527 End-to-End Integration Demo

작성: 2026-05-07

## 구조

```
사용자 발화
    ↓
[Preprocess]  STT 정규화 (남방→난방, 오늘날씨→오늘 날씨, 296 매핑)
    ↓
[NLU IntentClassifierV46]  KoELECTRA + CNN 5-head + PostRulesV4
    ↓
[IntentRouter]  fn → 채널 매핑
    ├─ AIDL (실 기기 제어)
    ├─ REST_RUEL (르엘 단지서버 API)
    ├─ EXTERNAL_API (날씨/뉴스/교통)
    └─ SELF (FAQ/매뉴얼/홈상태)
    ↓
[AidlMockClient] (또는 실 AIDL 호출)
    ↓
응답 (unit_status)
```

## 디바이스 결과 (13개 발화)

| # | 발화 | 채널 | NLU latency |
|---|------|------|-------------|
| 0 | 거실 불 켜줘 | AIDL | 27ms |
| 1 | 에어컨 23도로 해줘 | AIDL | 33ms |
| 2 | 주방 환기 켜줘 | AIDL | 32ms |
| 3 | 안방 난방 올려줘 | AIDL | 39ms |
| 4 | 가스 잠가줘 | AIDL | 28ms |
| 5 | 커튼 닫아줘 | AIDL | 27ms |
| 6 | 외출모드 실행해 줘 | AIDL | 26ms |
| 7 | 오늘 날씨 어때? | EXTERNAL_API | 29ms |
| 8 | 엘리베이터 불러줘 | REST_RUEL | 29ms |
| 9 | 에너지 사용량 어때? | REST_RUEL | 28ms |
| 10 | 우리집 상태 어때? | SELF | 28ms |
| 11 | 황사 때문에 창문 닫아야 해 | AIDL | 29ms |
| 12 | 어르신이 덥다고 하시네요 | AIDL | 26ms |

**평균 NLU: 29ms / 추론**  
**채널 분배: AIDL 9건, REST 2건, External 1건, SELF 1건**

## 산출물

```
src/main/kotlin/com/t527/smart_service/integration/
  ├─ IntentRouter.kt        NLU intent → 채널 라우팅 + AIDL JSON 생성
  └─ AidlMockClient.kt      AIDL 호출 시뮬레이션 (실 IoT 연동 전)

src/main/kotlin/com/t527/smart_service/
  └─ IntegrationDemoActivity.kt   13개 발화 자동 데모 + 통계 출력
```

## 실 AIDL 연동 시 변경점

`AidlMockClient.call(request)` → 실 AIDL bind:

```kotlin
class RealAidlClient(context: Context) {
    private var binder: IWallpadRemoteAccess? = null
    
    fun bind() {
        Intent("com.bestin.wallpadserver.AIDL_REMOTE_ACCESS").also { intent ->
            intent.setPackage("com.bestin.wallpadserver")
            context.bindService(intent, conn, Context.BIND_AUTO_CREATE)
        }
    }
    
    fun call(request: JSONObject): JSONObject {
        val responseStr = binder!!.requestRemoteAccess(request.toString())
        return JSONObject(responseStr)
    }
}
```

기존 IntegrationDemoActivity는 그대로 두고 `AidlMockClient` → `RealAidlClient`만 교체.

## 다음 단계

1. **실 AIDL 연동** (bestin WallpadServer 패키지명 확인 필요)
2. **TTS 응답 생성** — IntentClassifierV46 결과 + AIDL unit_status → 한국어 응답
3. **VoiceAiService에 통합** — STT 결과 → IntentClassifierV46 → IntentRouter → AIDL → TTS
