# Android JNI 통합 가이드 — 코드 예시

**목적**: ONNX 모델 + 후처리 rules + DST를 Android (Kotlin/Java)로 포팅.

## 전체 아키텍처

```
User Voice → STT → NluService (Kotlin)
                      │
                      ├── Preprocessor.kt (STT_CORRECTION dict)
                      ├── BertTokenizer.kt
                      ├── OnnxRunner.kt (OnnxRuntime Android)
                      ├── PostProcRules.kt (14 rules)
                      ├── DialogueStateTracker.kt
                      └── ResponseGenerator.kt
```

## 1. Preprocessor.kt

### 데이터
- `STT_CORRECTION`: 254 entries Map<String, String>
- JSON 파일로 assets/에 저장, 앱 시작 시 load

```kotlin
class Preprocessor(context: Context) {
    private val sttCorrection: Map<String, String>
    private val koreanNums: Map<String, Int>

    init {
        sttCorrection = loadJsonMap(context, "stt_correction.json")
        // 한글 숫자 dict
        koreanNums = mapOf(
            "하나" to 1, "둘" to 2, ..., "십" to 10,
            "이십" to 20, "삼십" to 30, ...
        )
    }

    fun preprocess(text: String): String {
        // 1. Strip non-printable
        var t = text.filter { it.isPrintable() || it == ' ' }

        // 2. Whitespace normalize
        t = t.replace(Regex("\\s+"), " ").trim()

        // 2-2. Single-char space collapse
        val tokens = t.split(" ")
        val singleCount = tokens.count { it.length == 1 && it[0] in '가'..'힣' }
        if (singleCount >= 4 && singleCount.toDouble() / tokens.size >= 0.7) {
            t = tokens.joinToString("")
        }

        // 3. STT correction 2-pass (long first)
        val sorted = sttCorrection.entries.sortedByDescending { it.key.length }
        for (pass in 0..1) {
            for ((wrong, correct) in sorted) {
                t = t.replace(wrong, correct)
            }
        }

        // 4. Korean number conversion
        t = convertKoreanNumbers(t)

        // 5. Strip leading 잠깐/얼른
        t = t.replace(Regex("^(잠깐|얼른)\\s+"), "")

        return t
    }

    private fun convertKoreanNumbers(text: String): String {
        // "이십삼도" → "23도"
        val tensPattern = Regex("(이십|삼십|사십|십)([일이삼사오육칠팔구])?도")
        return tensPattern.replace(text) { m ->
            val tens = mapOf("십" to 10, "이십" to 20, "삼십" to 30, "사십" to 40)[m.groupValues[1]] ?: 10
            val ones = if (m.groupValues[2].isNotEmpty())
                mapOf("일" to 1, "이" to 2, ...)[m.groupValues[2]] ?: 0 else 0
            "${tens + ones}도"
        }
    }
}
```

### assets/stt_correction.json 생성
```bash
python3 -c "
import json
from scripts.preprocess import STT_CORRECTION
json.dump(STT_CORRECTION, open('assets/stt_correction.json', 'w'), ensure_ascii=False, indent=2)
"
```

## 2. BertTokenizer.kt

기본 구조 (ko-sbert vocab 포팅):

```kotlin
class BertTokenizer(context: Context) {
    private val vocab: Map<String, Int>
    private val maxLen = 32

    init {
        // assets/vocab.txt 로드
        vocab = context.assets.open("vocab.txt").bufferedReader().useLines { lines ->
            lines.mapIndexed { idx, token -> token to idx }.toMap()
        }
    }

    fun tokenize(text: String): LongArray {
        val tokens = mutableListOf<Int>()
        tokens.add(vocab["[CLS]"] ?: 2)

        // Basic whitespace + subword tokenization
        for (word in text.split(" ")) {
            tokens.addAll(subwordTokenize(word))
        }
        tokens.add(vocab["[SEP]"] ?: 3)

        // Pad/truncate to maxLen
        while (tokens.size < maxLen) tokens.add(0)
        if (tokens.size > maxLen) tokens.subList(maxLen, tokens.size).clear()

        return tokens.map { it.toLong() }.toLongArray()
    }

    private fun subwordTokenize(word: String): List<Int> {
        // WordPiece: greedy longest-match
        // ... (standard BERT tokenizer impl)
    }
}
```

**참고**: HuggingFace `transformers` Java port 또는 `sentencepiece-jni` 사용 가능.

## 3. OnnxRunner.kt

```kotlin
class OnnxRunner(context: Context) {
    private val env: OrtEnvironment
    private val session: OrtSession

    init {
        env = OrtEnvironment.getEnvironment()
        val modelBytes = context.assets.open("nlu_v28_v46_ensemble.onnx").readBytes()
        session = env.createSession(modelBytes)
    }

    fun run(inputIds: LongArray): Map<String, Int> {
        val inputTensor = OnnxTensor.createTensor(env, arrayOf(inputIds))
        val inputs = mapOf("input_ids" to inputTensor)
        val outputs = session.run(inputs)

        // 5 heads
        val fnLogits = (outputs[0].value as Array<FloatArray>)[0]
        val execLogits = (outputs[1].value as Array<FloatArray>)[0]
        val dirLogits = (outputs[2].value as Array<FloatArray>)[0]
        val paramLogits = (outputs[3].value as Array<FloatArray>)[0]
        val judgeLogits = (outputs[4].value as Array<FloatArray>)[0]

        return mapOf(
            "fn" to argmax(fnLogits),
            "exec" to argmax(execLogits),
            "dir" to argmax(dirLogits),
            "param" to argmax(paramLogits),
            "judge" to argmax(judgeLogits),
        )
    }
}
```

**Gradle 의존성**:
```gradle
dependencies {
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.17.1'
}
```

## 4. PostProcRules.kt

핵심 14 rules. Python → Kotlin 직역:

```kotlin
data class Preds(
    var fn: String,
    var exec: String,
    var dir: String,
    var param: String,
    var judge: String
)

object PostProcRules {
    val ROOM_REGEX = Regex("(거실|안방|침실|주방|부엌|작은방|아이방|서재|현관)")

    fun apply(preds: Preds, text: String): Preds {
        // Rule 1: param_type basics
        if (preds.dir in listOf("open", "close", "stop")) preds.param = "none"
        if (preds.judge != "none") preds.param = "none"
        if (preds.exec in listOf("query_then_respond", "direct_respond", "clarify", "query_then_judge"))
            preds.param = "none"

        // Rule 2: 밝게 → up, 어둡게 → down
        if (Regex("밝게").containsMatchIn(text) && preds.dir == "down") {
            preds.dir = "up"; preds.param = "brightness"
        }
        if (Regex("어둡게").containsMatchIn(text) && preds.dir in listOf("up", "on")) {
            preds.dir = "down"; preds.param = "brightness"
        }

        // Rule 3: 엘리베이터 호출 패턴
        if (Regex("엘리베이터|엘베|승강기|리프트").containsMatchIn(text)) {
            if (Regex("호출|불러|올라\\s*와|내려\\s*와|오라고|와\\s*줘").containsMatchIn(text)) {
                preds.exec = "control_then_confirm"
                if (preds.dir == "none") preds.dir = "on"
            }
        }

        // Rule 4: N모드로 → set (ac/heat/vent only)
        if (Regex("(냉방|제습|송풍|자동|취침|외출)\\s*모드").containsMatchIn(text)
            && preds.fn in listOf("ac_control", "heat_control", "vent_control")) {
            preds.dir = "set"; preds.param = "mode"
        }

        // Rule 5: 알람/모닝콜 → schedule_manage (device 없을 때만)
        val hasDevice = Regex("조명|불|램프|난방|에어컨|환기|가스|도어|커튼|공기청정|블라인드").containsMatchIn(text)
        if (!hasDevice && Regex("알람|모닝콜").containsMatchIn(text)
            && preds.fn in listOf("system_meta", "home_info", "unknown")) {
            preds.fn = "schedule_manage"
        }

        // Rule 6: OOD keywords → unknown
        val oodKeywords = listOf("네비게이션", "비행기", "크루즈", "수면 기록", "길 안내")
        if (oodKeywords.any { it in text }) {
            preds.fn = "unknown"; preds.exec = "direct_respond"; preds.dir = "none"; preds.param = "none"
        }

        // Rule 7: 전화 entity-aware
        if ("전화" in text && preds.fn == "home_info") {
            val entities = listOf("관리사무소", "관리실", "경비", "이웃", "주민", "같은 동",
                                   "다른 집", "분리수거", "공동", "놓친")
            if (!entities.any { it in text }) {
                preds.fn = "unknown"; preds.exec = "direct_respond"; preds.dir = "none"
            }
        }

        // Rule 8: unknown → weather/news/medical recovery
        if (preds.fn == "unknown") {
            when {
                Regex("날씨|기온|비\\s*와|더울까|추울까|맑|흐림").containsMatchIn(text) -> {
                    preds.fn = "weather_query"; preds.exec = "query_then_respond"
                }
                Regex("뉴스|브리핑|속보").containsMatchIn(text) -> {
                    preds.fn = "news_query"; preds.exec = "query_then_respond"
                }
                Regex("병원|의원|약국|신경외과|내과|외과|안과|치과|한의원").containsMatchIn(text) -> {
                    preds.fn = "medical_query"; preds.exec = "query_then_respond"
                }
            }
        }

        // Rule 9: {room}{device} 좀 {verb} 어순 → CTC
        if (preds.exec == "clarify" && preds.fn == "light_control") {
            val pattern = Regex("(거실|안방|침실|주방|부엌|작은방|아이방|서재|현관)(?:에|은|의)?\\s+(불|조명|등|라이트)\\s+좀\\s+(켜|꺼|끄)")
            if (pattern.containsMatchIn(text)) {
                preds.exec = "control_then_confirm"
                if (preds.dir == "none") {
                    preds.dir = if ("켜" in text) "on" else "off"
                }
            }
        }

        // Rule 10: Curtain open → up/close/stop 수정
        if (preds.fn == "curtain_control") {
            if ("올려" in text && preds.dir in listOf("stop", "none", "open")) preds.dir = "up"
            if ("블라인드" in text && "내려" in text && preds.dir in listOf("down", "none", "open"))
                preds.dir = "close"
            if ("블라인드" in text && !Regex("올려|내려|열어|닫아|멈춰|스톱").containsMatchIn(text)
                && preds.dir == "open") preds.dir = "stop"
            if (Regex("닫아|닫기").containsMatchIn(text) && preds.dir == "open") preds.dir = "close"
        }

        // Rule 11: heat CTC + none → on
        if (preds.fn == "heat_control" && preds.exec == "control_then_confirm" && preds.dir == "none") {
            preds.dir = "on"
        }

        // Rule 12: 화면/음량/알림 → home_info (capability 제외)
        val capability = Regex("어떻게|할\\s*수\\s*있").containsMatchIn(text)
        if (preds.fn == "system_meta" && !capability) {
            if (Regex("화면\\s*밝기|월패드\\s*밝기|음량").containsMatchIn(text)) preds.fn = "home_info"
            else if ("알림" in text && !Regex("사용량|긴급|에너지").containsMatchIn(text)) preds.fn = "home_info"
        }

        // Rule 13: 공기청정 → vent_control
        if (Regex("공기청정|공기\\s*정화|공기\\s*청정").containsMatchIn(text)) {
            if (preds.fn in listOf("weather_query", "unknown", "home_info")) {
                preds.fn = "vent_control"
                if (preds.exec == "direct_respond") preds.exec = "control_then_confirm"
            }
        }

        // Rule 14: 덥/춥 semantic override
        if (preds.fn == "heat_control" && Regex("덥다|더워|덥네|더운").containsMatchIn(text)
            && !Regex("난방|보일러|온돌").containsMatchIn(text)) {
            preds.fn = "ac_control"
            if (preds.dir in listOf("none", "up")) preds.dir = "on"
        }

        return preds
    }
}
```

## 5. DialogueStateTracker.kt

```kotlin
class DialogueStateTracker(val timeoutMs: Long = 10_000) {
    data class State(
        var fn: String? = null,
        var exec: String? = null,
        var dir: String? = null,
        var room: String? = null,
        var value: Pair<String, Int>? = null,
        var text: String? = null,
        var time: Long = 0,
    )

    private val state = State()
    private val history = mutableListOf<Map<String, Any?>>()

    fun update(nlu: Preds, room: String, text: String): Map<String, Any?> {
        val active = System.currentTimeMillis() - state.time < timeoutMs
        var fn = nlu.fn
        var exec = nlu.exec
        var dir = nlu.dir

        if (active) {
            val followup = getFollowupType(text, fn)
            when (followup) {
                "room" -> {
                    fn = state.fn ?: fn
                    exec = state.exec ?: exec
                    dir = state.dir ?: dir
                }
                "device" -> {
                    exec = state.exec ?: exec
                    if (state.dir in listOf("on", "off", "open", "close"))
                        dir = state.dir ?: dir
                    else if (dir == "none")
                        dir = state.dir ?: dir
                }
                "confirm" -> {
                    fn = state.fn ?: fn
                    // ... (see Python version)
                }
                // correction, there_too, slot_fill ...
            }
        }

        state.fn = fn; state.exec = exec; state.dir = dir
        state.room = if (room != "none") room else state.room
        state.text = text
        state.time = System.currentTimeMillis()

        return mapOf("fn" to fn, "exec_type" to exec, "param_direction" to dir,
                     "room" to (state.room ?: "none"), "value" to state.value)
    }

    fun reset() {
        state.fn = null; state.exec = null; state.dir = null
        state.room = null; state.value = null
        state.text = null; state.time = 0
        history.clear()
    }
}
```

## 6. 전체 통합 NluService.kt

```kotlin
class NluService(context: Context) {
    private val preprocessor = Preprocessor(context)
    private val tokenizer = BertTokenizer(context)
    private val onnxRunner = OnnxRunner(context)
    private val dst = DialogueStateTracker()
    private val responseGen = ResponseGenerator()

    fun process(text: String): NluResult {
        val pp = preprocessor.preprocess(text)
        val ids = tokenizer.tokenize(pp)
        val logits = onnxRunner.run(ids)
        val preds = Preds(
            fn = FN_LABELS[logits["fn"]!!],
            exec = EXEC_LABELS[logits["exec"]!!],
            dir = DIR_LABELS[logits["dir"]!!],
            param = PARAM_LABELS[logits["param"]!!],
            judge = JUDGE_LABELS[logits["judge"]!!]
        )
        val ruled = PostProcRules.apply(preds, pp)
        val room = extractRoom(pp)
        val resolved = dst.update(ruled, room, pp)
        val response = responseGen.generate(resolved, pp)
        return NluResult(...)
    }
}
```

## 자산 파일 (앱 assets/)

| 파일 | 크기 | 설명 |
|------|:---:|------|
| `nlu_v28_v46_ensemble.onnx` | 105MB | 모델 (LFS에서 복사) |
| `vocab.txt` | 500KB | ko-sbert vocab |
| `stt_correction.json` | 30KB | 254 preprocess entries |
| `fn_labels.json` | 1KB | fn idx → name |
| `exec_labels.json` | 0.5KB | exec idx → name |
| `dir_labels.json` | 0.5KB | dir idx → name |

**총 assets**: ~106MB (주로 ONNX)

## T527 NPU 옵션

ONNX CPU로 0.67ms 이미 빠름. NPU 변환은 선택:

```
ONNX 105MB → Acuity pegasus → NB 70MB (int8 추정)
CPU 0.67ms → NPU 0.3ms (추정, 2x 빠름)
```

NPU 변환 시:
1. ONNX → Acuity import (`pegasus import onnx ...`)
2. uint8 quantize (calibration data 필요)
3. NB export (network_binary.nb)

참고: `/home/nsbb/travail/claude/T527/CLAUDE.md` 의 STT 변환 절차 기반.

## Testing on Android

```kotlin
// Unit test
val service = NluService(context)
val r = service.process("거실 불 켜줘")
assertEquals("light_control", r.fn)
assertEquals("living", r.room)
assertEquals("on", r.dir)

// Integration
val r2 = service.process("안방도")
assertEquals("light_control", r2.fn)  // DST inherits
assertEquals("bedroom_main", r2.room)
```
