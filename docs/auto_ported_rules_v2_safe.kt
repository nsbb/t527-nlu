// 자동 포팅 v2 (엄격) — fn/dir 조건 있는 단순 규칙만
// 미지원 기기 (볼륨/선풍기 등) → unknown
if (Regex("볼륨|볼름|볼음|선풍기|음악|노래|유튜브").containsMatchIn(text)) {
    if (p.fn in setOf("home_info", "system_meta", "unknown", "energy_query", "news_query")) {
        p.fn = "unknown"
        p.execType = "direct_respond"
        p.direction = "none"
        p.paramType = "none"
    }
}

// "올려도 될까요/올려도 돼요" → dir=up (모델이 될까=set 오분류)
if (Regex("올려도\\s*(?:될까|돼요|돼|되나요)").containsMatchIn(text)) {
    if (p.fn in setOf("heat_control", "ac_control", "light_control", "param_direction", "set")) {
        p.direction = "up"
    }
}

// continuous: judgment 질문 (X해도 돼/괜찮아?) → weather_query (fn 상관없이)
if (Regex("타도\\s*돼\\?|괜찮아\\??$|나가도\\s*돼|세차해도|운동해도|소풍").containsMatchIn(text)) {
    if (p.fn in setOf("market_query", "traffic_query", "unknown")) {
        p.fn = "weather_query"
        p.execType = "query_then_judge"
        p.judge = "outdoor_activity"
    }
}

// v129: "좀 더 환해졌으면/밝아졌으면" → light/up (비교급 소원 = 현재보다 밝게)
if (Regex("(?:좀\\s*더|더\\s*좀|훨씬)\\s*(?:환해졌으면|밝아졌으면|환해지면|밝아지면)").containsMatchIn(text)) {
    if (p.fn in setOf("light_control", "unknown")) {
        p.fn = "light_control"
        p.direction = "up"
    }
}

// v129: 히터/보일러 켜놨더니 더워 → heat_control/off (과열 불만)
if (Regex("(?:히터|보일러|난방|라디에이터)\\s*(?:\\S+\\s*){0,3}(?:켜놨더니|틀었더니|켜놓고\\s*잤더니|틀어놓고\\s*잤더니)\\s*(?:더워|덥다|덥네|더운데)").containsMatchIn(text)) {
    if (p.fn in setOf("ac_control", "heat_control", "unknown")) {
        p.fn = "heat_control"
        p.direction = "off"
    }
}

// v129: 외출할 때/나갈 때 잠가줘 → security_mode/on (방범 활성화)
if (Regex("(?:외출|나갈|나가면서|나갈\\s*때|외출\\s*할\\s*때|나갈\\s*때)\\s*(?:\\S+\\s*)?잠가(?:줘|요|주세요|주겠어)").containsMatchIn(text)) {
    if (p.fn in setOf("security_mode", "door_control", "unknown")) {
        p.fn = "security_mode"
        p.direction = "on"
    }
}

// v86: "켜볼까요/켜볼까/해볼까" → dir=on when device fn but dir=none
if (Regex("켜볼까|틀어볼까|켜볼게").containsMatchIn(text)) {
    if (p.fn in setOf("param_direction", "none")) {
        p.direction = "on"
    }
}

// v93: "끌까요/끌게요/꺼볼까요" — 끄기 청유/제안 → dir=off
if (Regex("(?:끌까요?|끌게요?|꺼볼까요?|꺼드릴까요?)").containsMatchIn(text)) {
    if (p.fn in setOf("light_control", "ac_control", "heat_control", "vent_control")) {
        p.direction = "off"
    }
}

// v94: 끄지 마 → dir=on (부정+끄다 = 켜놔야 함, TS dir=off는 오류)
if (Regex("끄지\\s*(?:마|말|말아줘|마세요)").containsMatchIn(text)) {
    if (p.fn in setOf("light_control", "ac_control", "heat_control", "vent_control")) {
        p.direction = "on"
    }
}

// v102: 완곡 제안형 "끄는 게 어때/켜는 게 어때" → dir 복구  "난방 끄는 게 어때?" = 난방 끄자는 제안 → heat/off
if (Regex("끄\\s*(?:는\\s*게?|면\\s*어때|면\\s*어떨까)").containsMatchIn(text)) {
    if (p.fn in setOf("param_direction", "none")) {
        p.direction = "off"
    }
}

if (Regex("켜\\s*(?:는\\s*게?|면\\s*어때|면\\s*어떨까)").containsMatchIn(text)) {
    if (p.fn in setOf("param_direction", "none")) {
        p.direction = "on"
    }
}

// v105: "꺼도 되지/꺼도 돼/꺼도 되나요" → dir=off (허락 형식 꺼줘)
if (Regex("꺼\\s*도\\s*(?:되지|되나|돼|될까)").containsMatchIn(text)) {
    if (p.fn in setOf("param_direction", "none")) {
        p.direction = "off"
    }
}

// v107: 부정 명령 "켜지 말아줘/켜지 마" → dir=off (don't turn on = keep off / turn off)
if (Regex("켜\\s*(?:지\\s*)?(?:말|마)\\s*(?:줘|요|세요|아줘|아요)?").containsMatchIn(text)) {
    if (p.fn in setOf("param_direction", "on", "none")) {
        p.execType = "control_then_confirm"
        p.direction = "off"
    }
}

// v107: 과거형 보고 "껐어/꺼졌어" → exec=query_then_respond (과거 행위 보고, 명령 아님)  dir을 off로: 껐다고 했으니 현재 o
if (Regex("껐어|껐는데|껐습니다|끄고\\s*왔어").containsMatchIn(text)) {
    if (p.fn in setOf("param_direction", "on", "none")) {
        p.execType = "query_then_respond"
        p.direction = "off"
    }
}

// v107: "켜놓아줘/켜놓아" → dir=on (contracted "켜놔"가 원형보다 잘 인식됨)
if (Regex("켜\\s*놓[아아줘주아]\\s*줘?").containsMatchIn(text)) {
    if (p.fn in setOf("param_direction", "none")) {
        p.execType = "control_then_confirm"
        p.direction = "on"
    }
}

// v109: "불/조명 껴줘" STT 오인식 → light_control/on ("껴" = "켜" 변형)
if (Regex("(?:불|조명|전등)\\s*껴\\s*줘").containsMatchIn(text)) {
    if (p.fn in setOf("weather_query", "unknown", "home_info")) {
        p.fn = "light_control"
        p.execType = "control_then_confirm"
        p.direction = "on"
    }
}

// v109: 자기 의향형 "~해야겠어/~야겠어" + 기기 → control 의도  "에어컨 꺼야겠어" = 에어컨 끌게(의향) → dir=off (이미 모델이 잘 처
if (Regex("(?:켜|틀어)\\s*야겠어").containsMatchIn(text)) {
    if (p.fn in setOf("param_direction", "none")) {
        p.execType = "control_then_confirm"
        p.direction = "on"
    }
}

if (Regex("(?:꺼|끄)\\s*야겠어").containsMatchIn(text)) {
    if (p.fn in setOf("param_direction", "none")) {
        p.execType = "control_then_confirm"
        p.direction = "off"
    }
}

// v109: 수면/취침 시간 표현 → unknown (기기 명령 아님)
if (Regex("잘\\s*(?:시간|게요|게|거야|래|까)").containsMatchIn(text)) {
    if (p.fn in setOf("param_direction", "none")) {
        p.fn = "unknown"
        p.execType = "direct_respond"
    }
}

// v112: "더 시원하게/따뜻하게/밝게/어둡게 해줘" → dir=up/down  비교급 "더" + 상태형용사 = 현재보다 강화 요청
if (Regex("(?:더|좀\\s*더)\\s*시원하게\\s*(?:해줘|해주세요|해봐|틀어줘)").containsMatchIn(text)) {
    if (p.fn in setOf("ac_control", "vent_control", "param_direction", "on", "none")) {
        p.direction = "up"
    }
}

if (Regex("(?:더|좀\\s*더)\\s*따뜻하게\\s*(?:해줘|해주세요|해봐)").containsMatchIn(text)) {
    if (p.fn in setOf("heat_control", "ac_control", "param_direction", "on", "none")) {
        p.direction = "up"
    }
}

// v112: 수사적 반어 요청 "끄면 안 되나요?/켜면 안 될까요?" → dir=off/on
if (Regex("(?:끄|꺼)\\s*(?:면\\s*안\\s*(?:되나요?|될까요?|돼요?)|도\\s*안\\s*돼\\??|면\\s*될까요?)").containsMatchIn(text)) {
    if (p.fn in setOf("param_direction", "none")) {
        p.direction = "off"
        p.execType = "control_then_confirm"
    }
}

if (Regex("(?:켜|켜도)\\s*(?:면\\s*안\\s*(?:될까요?|되나요?)|도\\s*안\\s*돼\\??|면\\s*될까요?)").containsMatchIn(text)) {
    if (p.fn in setOf("param_direction", "none")) {
        p.direction = "on"
        p.execType = "control_then_confirm"
    }
}

// v122: 꺼줄 수 있어/있어요 → dir=off (공손한 끄기 요청)
if (Regex("꺼줄\\s*수\\s*있").containsMatchIn(text)) {
    if (p.fn in setOf("param_direction", "off")) {
        p.direction = "off"
        p.execType = "control_then_confirm"
    }
}

// v124: 추위를 타다 → heat_control/on (추위 민감성 표현)
if (Regex("추위를?\\s*(?:많이\\s*)?타|추위에?\\s*(?:많이\\s*)?약하|추위에?\\s*민감").containsMatchIn(text)) {
    if (p.fn in setOf("energy_query", "unknown", "weather_query", "home_info", "ac_control", "heat_control")) {
        p.fn = "heat_control"
        p.execType = "control_then_confirm"
        p.direction = "on"
    }
}

// v124: 아주/정말 많이 추워요 → heat_control/on (energy_query 오예측 교정)
if (Regex("(?:아주|정말|너무|되게)\\s*많이\\s*추워|많이\\s*춥(?:다|네|죠)").containsMatchIn(text)) {
    if (p.fn in setOf("energy_query", "unknown", "weather_query")) {
        p.fn = "heat_control"
        p.execType = "control_then_confirm"
        p.direction = "on"
    }
}

// v124: 잠깐만 + 켜줘/꺼줘 → dir 교정 (시간 접두어로 dir=none 오예측)
if (Regex("잠깐만\\s*(?:\\S+\\s*)?켜줘").containsMatchIn(text)) {
    if (p.fn in setOf("param_direction", "none")) {
        p.direction = "on"
    }
}

if (Regex("잠깐만\\s*(?:\\S+\\s*)?꺼줘").containsMatchIn(text)) {
    if (p.fn in setOf("param_direction", "none")) {
        p.direction = "off"
    }
}

// v126: 난방/보일러 올려야 → heat_control/up (의무형 상향)
if (Regex("(?:난방|보일러|히터)\\s*(?:\\S+\\s*){0,2}올려야|온도\\s*(?:를\\s*)?(?:좀\\s*)?올려야\\s*할\\s*것\\s*같").containsMatchIn(text)) {
    if (p.fn in setOf("heat_control", "unknown")) {
        p.fn = "heat_control"
        p.direction = "up"
    }
}

// v126: 보일러/난방/라디에이터 틀었는데도 추워 → heat_control/up (효과 부족)
if (Regex("(?:난방|보일러|히터|라디에이터)\\s*(?:\\S+\\s*){0,3}(?:켰는데도?|틀었는데도?|켜놨는데도?|틀어놨는데도?)\\s*(?:아직도?\\s*)?(?:추워|춥다|추운데)").containsMatchIn(text)) {
    if (p.fn in setOf("heat_control", "energy_query", "unknown")) {
        p.fn = "heat_control"
        p.direction = "up"
    }
}

// v130: 환기/냉방 가동 부탁드립니다 → dir=on (가동=켜다)
if (Regex("(?:환기|냉방|에어컨|공기청정)\\s*(?:좀\\s*)?가동\\s*(?:해|시켜|부탁|드려)").containsMatchIn(text)) {
    if (p.fn in setOf("vent_control", "ac_control", "param_direction", "none", "off")) {
        p.direction = "on"
    }
}

// v128: "OO 상태 알려줘/확인해줘" → home_info (v72: light_control 오예측 교정)
if (Regex("(?:거실|주방|침실|안방|욕실|집|실내|전체)?\\s*(?:전체\\s*)?상태\\s*(?:알려|확인|보여|말해)").containsMatchIn(text)) {
    if (p.fn in setOf("light_control", "unknown", "system_meta")) {
        p.fn = "home_info"
        p.direction = "none"
    }
}

// v128: 에어컨/기기 세게 틀어줘 → dir=up (강도 높이기)
if (Regex("(?:에어컨|냉방|난방|보일러|히터|선풍기|환풍기)\\s*(?:\\S+\\s*){0,2}(?:세게|강하게|빠르게|강풍으로|최대로)\\s*(?:틀|켜|해)").containsMatchIn(text)) {
    if (p.fn in setOf("ac_control", "heat_control", "vent_control", "param_direction", "on")) {
        p.direction = "up"
    }
}

// 앞뒤 순서가 바뀐 경우도 커버: "세게 틀어줄래" with ac context in text
if (Regex("(?:세게|강하게|강풍으로|최대로)\\s*(?:틀|켜)(?:어줄래|어줘|어줄게|어줄까|려줘)").containsMatchIn(text)) {
    if (p.fn in setOf("ac_control", "heat_control", "vent_control", "param_direction", "on")) {
        p.direction = "up"
    }
}

// v127: 에어컨/난방 이제 안 써도/쓰지 않아도 될 것 같아 → dir=off (완곡한 불필요 표현)
if (Regex("(?:에어컨|냉방|난방|보일러|히터|공기청정기)\\s*(?:[은는이가]?\\s*)?(?:이제\\s*)?(?:안\\s*써도|쓰지\\s*않아도|사용\\s*안\\s*해도)\\s*될\\s*것\\s*같").containsMatchIn(text)) {
    if (p.fn in setOf("param_direction", "on", "none")) {
        p.direction = "off"
    }
}

// v127: 이제 꺼도/껐다가도 될 것 같아요 → dir=off (완곡한 끄기 표현)
if (Regex("(?:이제\\s*)?(?:꺼도|끄고\\s*싶은데|꺼도\\s*좋을\\s*것\\s*같)\\s*될\\s*것\\s*같").containsMatchIn(text)) {
    if (p.fn in setOf("param_direction", "on")) {
        p.direction = "off"
    }
}
