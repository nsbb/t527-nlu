// 자동 생성된 PostRules — apply_post_rules의 단순 규칙들
// 출처: scripts/ensemble_inference_with_rules.py
// 자동 변환: scripts/auto_port_rules_to_kotlin.py

// 한기가 → heat_control/on (preprocess에서 한기→환기 제거했지만 모델이 미학습)
if (Regex("한기가\\s*(?:도네|느껴|돌아|나|왔|든)").containsMatchIn(text)) {
    p.fn = "heat_control"
    p.execType = "control_then_confirm"
    p.direction = "on"
}

// continuous: system_meta → unknown (특정 OOD 키워드)
if (Regex("와이파이\\s*비번|영어로\\s*뭐|업데이트$|^일정$").containsMatchIn(text)) {
    p.fn = "unknown"
    p.execType = "direct_respond"
    p.direction = "none"
}

// v85: 가스 누출 의심 표현 → gas_control/close
if (Regex("가스\\s*(?:새|새는|새고|냄새|누출|샌다|샌)").containsMatchIn(text)) {
    p.fn = "gas_control"
    p.execType = "control_then_confirm"
    p.direction = "close"
}

// 추위 비유 → dir=none 교정
if (Regex("얼어\\s*죽|냉동실|냉장고\\s*같|시베리아|이글루|덜덜\\s*떨|이가\\s*딱딱|한기|몸이\\s*꽁").containsMatchIn(text)) {
    p.direction = "on"
}

// "쌀쌀해/서늘해" = feeling cold → heat_control/on (ac_control 오분류 교정)  v84: "서늘해지게 해줘" (소원형 = 시원해지고 싶어) 제외 — ac_control이 맞음  v121: 형용사형(쌀쌀한/서늘한) 추가
if (Regex("쌀쌀(?:해|하다|하네|하지|해요|한)|서늘(?:해(?!지게|져|지도록)|하다|하네|하죠|한(?!지게|져|지도록))|으슬으슬|추들추들").containsMatchIn(text)) {
    if (p.fn in setOf("ac_control", "weather_query", "unknown", "home_info", "heat_control") && p.direction in setOf("none")) {
        p.fn = "heat_control"
        p.execType = "control_then_confirm"
        p.direction = "on"
    }
}

// "끈적끈적/끈끈해/축축해/땀이 나" = sticky/sweaty hot → ac_control/on
if (Regex("끈적끈적|끈끈(?:해|하다|하네)|축축(?:해|하다)|땀이\\s*(?:나|났어|나네)").containsMatchIn(text)) {
    if (p.fn in setOf("heat_control", "unknown", "home_info", "weather_query", "ac_control") && p.direction in setOf("none", "off")) {
        p.fn = "ac_control"
        p.execType = "control_then_confirm"
        p.direction = "on"
    }
}

// v83: "잠가야겠다/잠가야지/잠그자" → door_control/close
if (Regex("(?:문|도어락?)\\s*(?:이나|을|은)?\\s*(?:잠가야겠|잠가야지|잠그자|잠가야|잠가볼)").containsMatchIn(text)) {
    p.fn = "door_control"
    p.execType = "control_then_confirm"
    p.direction = "close"
}

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

// continuous: 비상 상황 키워드 (가스 냄새/타는 냄새 등) → security_mode emergency
if (Regex("가스\\s*냄새|타는\\s*냄새|연기\\s*(?:나|난|올)|불\\s*(?:났|붙)|침입|도둑").containsMatchIn(text)) {
    p.fn = "security_mode"
    p.execType = "control_then_confirm"
    p.direction = "on"
}

// continuous: judgment 질문 (X해도 돼/괜찮아?) → weather_query (fn 상관없이)
if (Regex("타도\\s*돼\\?|괜찮아\\??$|나가도\\s*돼|세차해도|운동해도|소풍").containsMatchIn(text)) {
    if (p.fn in setOf("market_query", "traffic_query", "unknown")) {
        p.fn = "weather_query"
        p.execType = "query_then_judge"
        p.judge = "outdoor_activity"
    }
}

// continuous: ac_control CTC + "해줘"/동작 없음 → on
if (Regex("해줘|해\\s*줘|틀어|가동|작동").containsMatchIn(text)) {
    p.direction = "on"
}

// vent_control CTC + 해줘/틀어 → on
if (Regex("해줘|해\\s*줘|틀어|가동").containsMatchIn(text)) {
    p.direction = "on"
}

// continuous: vent_control "환풍 모드" / "환기 시스템" / "공기 순환" direct → query (좁게)
if (Regex("^환풍\\s*모드$|환기\\s*시스템|공기\\s*순환").containsMatchIn(text)) {
    p.execType = "query_then_respond"
}

// continuous: market_query + company name → query (TS inconsistent)
if (Regex("(?:LG|삼성|현대|카카오|기아|네이버|KB|SK|포스코)\\s*주가").containsMatchIn(text)) {
    p.execType = "query_then_respond"
}

// curtain_control dir=stop 오예측 교정 (위로/올려 → up)
if (Regex("위로|위\\s*로|올려|올리").containsMatchIn(text)) {
    p.direction = "up"
}

// v129: "좀 더 환해졌으면/밝아졌으면" → light/up (비교급 소원 = 현재보다 밝게)
if (Regex("(?:좀\\s*더|더\\s*좀|훨씬)\\s*(?:환해졌으면|밝아졌으면|환해지면|밝아지면)").containsMatchIn(text)) {
    if (p.fn in setOf("light_control", "unknown")) {
        p.fn = "light_control"
        p.direction = "up"
    }
}

// v129: 등줄기/온몸이 서늘해 → heat_control/on (한기 신체감각)
if (Regex("(?:등줄기|등골|온몸|몸)\\s*(?:이|가|도)?\\s*(?:서늘해|서늘하|오싹|으슬으슬|떨려)").containsMatchIn(text)) {
    if (p.fn in setOf("vent_control", "ac_control", "unknown", "home_info") && p.direction in setOf("none", "off")) {
        p.fn = "heat_control"
        p.direction = "on"
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

// v89→v90: 기기 상태 조회 "돌아가고 있어/작동 중이야" → exec=query, dir=none (fn은 유지)  "켜져/꺼져/잠겨/열려/닫혀" 패턴은 v77 rule이 이미 처리 — 중복 방지
if (Regex("(?:돌아가고|작동\\s*중이야|돌고\\s*있어)\\s*(?:있어|있나|있나요|있어요|있죠)?\\??").containsMatchIn(text)) {
    p.execType = "query_then_respond"
    p.direction = "none"
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

// v95: 꺼도 될까요/끄면 될까요 → dir=off (허락 요청도 실질적 의도는 off)  v95b: 'on'도 포함 — v72 모델이 on으로 출력하는 경우 대응
if (Regex("꺼\\s*도\\s*될까|끄\\s*면\\s*될까|끄\\s*도\\s*될까").containsMatchIn(text)) {
    if (p.fn in setOf("light_control", "ac_control", "heat_control", "vent_control") && p.direction in setOf("none", "set", "on")) {
        p.direction = "off"
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

// v105: "기기가 너무 세다/강하다" → dir=down, "너무 약하다" → dir=up
if (Regex("너무\\s*(?:세다|강하다|강해|세네|강하네|센\\s*것\\s*같)").containsMatchIn(text)) {
    p.direction = "down"
}

if (Regex("너무\\s*(?:약해|약하다|약하네|약한\\s*것\\s*같)").containsMatchIn(text)) {
    p.direction = "up"
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

// v107: 과거형 보고 "껐어/꺼졌어" → exec=query_then_respond (과거 행위 보고, 명령 아님)  dir을 off로: 껐다고 했으니 현재 off 상태 확인
if (Regex("껐어|껐는데|껐습니다|끄고\\s*왔어").containsMatchIn(text)) {
    if (p.fn in setOf("param_direction", "on", "none")) {
        p.execType = "query_then_respond"
        p.direction = "off"
    }
}

// "저절로 꺼졌어/혼자 꺼졌어" → 상태 진술 (status event, not a command)
if (Regex("(?:저절로|혼자|자동으로|갑자기)\\s*(?:꺼졌|켜졌|꺼진|켜진)").containsMatchIn(text)) {
    p.execType = "query_then_judge"
}

// v107: "켜놓아줘/켜놓아" → dir=on (contracted "켜놔"가 원형보다 잘 인식됨)
if (Regex("켜\\s*놓[아아줘주아]\\s*줘?").containsMatchIn(text)) {
    if (p.fn in setOf("param_direction", "none")) {
        p.execType = "control_then_confirm"
        p.direction = "on"
    }
}

// v108: "저것도/그것도 꺼줘" 지시어 + 꺼 → dir=off 강제 (fn 무관)
if (Regex("(?:저것도|그것도|저거도|그거도)\\s*(?:꺼|끄)").containsMatchIn(text)) {
    p.direction = "off"
}

// "저것도/그것도 켜줘" → dir=on
if (Regex("(?:저것도|그것도|저거도|그거도)\\s*(?:켜|틀어)").containsMatchIn(text)) {
    p.direction = "on"
}

// v109: "불/조명 껴줘" STT 오인식 → light_control/on ("껴" = "켜" 변형)
if (Regex("(?:불|조명|전등)\\s*껴\\s*줘").containsMatchIn(text)) {
    if (p.fn in setOf("weather_query", "unknown", "home_info")) {
        p.fn = "light_control"
        p.execType = "control_then_confirm"
        p.direction = "on"
    }
}

// v131: 날씨 좋은데/맑은데 + 환기 → vent_control/on (날씨-환기 연계)
if (Regex("날씨\\s*(?:좋은데|맑은데|화창한데|좋으니|맑으니)\\s*(?:\\S+\\s*){0,2}환기").containsMatchIn(text)) {
    if (p.fn in setOf("weather_query", "unknown", "home_info") && p.direction in setOf("none", "off")) {
        p.fn = "vent_control"
        p.direction = "on"
    }
}

// v131: 환기시키고/환기하고 청소/요리 → vent_control/on (선행 환기 선언)
if (Regex("환기\\s*(?:시키고|하고|좀\\s*시키고)\\s*(?:청소|요리|조리|닦)").containsMatchIn(text)) {
    if (p.fn in setOf("unknown", "home_info") && p.direction in setOf("none", "off")) {
        p.fn = "vent_control"
        p.direction = "on"
    }
}

// v132: "환기도 좀 해줘" → vent_control/on (조사 '도' 영향으로 weather_query 오예측 교정)
if (Regex("환기도?\\s*(?:좀\\s*)?(?:해|시켜|돌려|틀어)\\s*줘").containsMatchIn(text)) {
    if (p.fn in setOf("weather_query", "unknown") && p.direction in setOf("none", "off")) {
        p.fn = "vent_control"
        p.execType = "control_then_confirm"
        p.direction = "on"
    }
}

// v109: 자기 의향형 "~해야겠어/~야겠어" + 기기 → control 의도  "에어컨 꺼야겠어" = 에어컨 끌게(의향) → dir=off (이미 모델이 잘 처리하지만 none인 경우 보완)
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

// v112: 이중 부정 강조 "안 끄면 안 돼?" → dir=off, "안 켜도 안 돼?" → dir=on
if (Regex("안\\s*(?:끄|꺼)\\s*(?:면|도)\\s*안\\s*돼").containsMatchIn(text)) {
    p.direction = "off"
    p.execType = "control_then_confirm"
}

if (Regex("안\\s*켜\\s*(?:면|도)\\s*안\\s*돼").containsMatchIn(text)) {
    p.direction = "on"
    p.execType = "control_then_confirm"
}

// v112: 월패드 자체 제어 요청 → unknown (self-referential, 미지원)
if (Regex("월패드\\s*(?:좀|를|은|이)?\\s*(?:꺼|껐|끄|켜|켰)").containsMatchIn(text)) {
    p.fn = "unknown"
    p.execType = "direct_respond"
    p.direction = "none"
}

// v110: 기기 + "이상한 소리/소리 이상해" → unknown (고장 보고, 제어 명령 아님)
if (Regex("이상한\\s*소리|소리\\s*이상|소음이|잡음이").containsMatchIn(text)) {
    p.fn = "unknown"
    p.execType = "direct_respond"
    p.direction = "none"
}

// v119: 커튼 + 멈춰줘/스톱/정지 → dir=stop (열기/닫기 명사 오인식 방지)  "전동커튼 열기 멈춰줘" → open 오예측 교정
if (Regex("멈춰\\s*(?:줘|주세요|요)?|스톱|정지\\s*(?:해줘)?").containsMatchIn(text)) {
    p.direction = "stop"
}

// v120: 창문으로 바람이 들어오다 → door_control/close
if (Regex("창문\\s*(?:으로|바람).*바람.*들어오|창문.*바람.*(?:들어오|새)").containsMatchIn(text)) {
    p.fn = "door_control"
    p.execType = "control_then_confirm"
    p.direction = "close"
}

// v120: 창문 열어두고 외출/나갔어/나왔어 → door_control/close (잊고 나간 경우)
if (Regex("창문\\s*(?:을\\s*)?열어\\s*두고\\s*(?:나갔|외출|나왔)").containsMatchIn(text)) {
    p.fn = "door_control"
    p.execType = "control_then_confirm"
    p.direction = "close"
}

// v120: 해/햇볕이 너무 눈부셔 → curtain_control/close (빛 차단 암시)
if (Regex("(?:해|햇볕|햇빛)\\s*(?:가|이)?\\s*(?:너무\\s*)?눈부|눈부신?\\s*(?:해|햇볕|햇빛)").containsMatchIn(text)) {
    p.fn = "curtain_control"
    p.execType = "control_then_confirm"
    p.direction = "close"
}

// v120: "에어컨 끄면 안 될까요" 완곡 요청 → ac_control/off
if (Regex("에어컨\\s*(?:좀\\s*)?끄면\\s*안\\s*될까").containsMatchIn(text)) {
    p.fn = "ac_control"
    p.execType = "control_then_confirm"
    p.direction = "off"
}

// v121: 수사적 더위 "이 더위에 어떻게 사나/버텨/살아" → ac_control/on
if (Regex("이\\s*더위에\\s*(?:어떻게|어케)").containsMatchIn(text)) {
    p.fn = "ac_control"
    p.execType = "control_then_confirm"
    p.direction = "on"
}

// v121: 조명/전등 깜빡거리다 → unknown (고장 리포트)
if (Regex("(?:조명|전등|불)\\s*(?:이|가)?\\s*(?:깜빡|깜박)(?:거려|이는|인다|거린다|이다)").containsMatchIn(text)) {
    p.fn = "unknown"
    p.execType = "direct_respond"
    p.direction = "none"
}

// v121: 창문 닫아도 될까요/돼요 → door_control/close (완곡 닫기 요청)
if (Regex("창문\\s*(?:좀\\s*)?닫아도\\s*(?:될까|돼요?|괜찮)").containsMatchIn(text)) {
    p.fn = "door_control"
    p.execType = "control_then_confirm"
    p.direction = "close"
}

// v122: 꺼줄 수 있어/있어요 → dir=off (공손한 끄기 요청)
if (Regex("꺼줄\\s*수\\s*있").containsMatchIn(text)) {
    if (p.fn in setOf("param_direction", "off")) {
        p.direction = "off"
        p.execType = "control_then_confirm"
    }
}

// v123: 창문 좀 열어주겠어요? → door_control/open (정중 의문형)
if (Regex("창문\\s*(?:좀\\s*)?열어\\s*주겠어요").containsMatchIn(text)) {
    p.fn = "door_control"
    p.execType = "control_then_confirm"
    p.direction = "open"
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

// v124: 창문 열어도 돼요? → door_control/open (창문 열기 완곡 요청)
if (Regex("창문\\s*(?:좀\\s*)?열어도\\s*돼요?").containsMatchIn(text)) {
    p.fn = "door_control"
    p.execType = "control_then_confirm"
    p.direction = "open"
}

// v124: 극도로/아주 심하게 덥습니다 → ac_control/on (heat 오예측 + set 교정)
if (Regex("(?:극도로|아주\\s*심하게?|정말\\s*너무)\\s*덥").containsMatchIn(text)) {
    p.fn = "ac_control"
    p.execType = "control_then_confirm"
    p.direction = "on"
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

// v126: 방/집이 얼음창고/냉동실 같아 → heat_control/on (한기 비유)
if (Regex("(?:방|집|여기|실내|거실|침실)\\s*(?:이\\s*)?(?:얼음창고|냉동창고|냉동실|냉동고|냉장고)\\s*같").containsMatchIn(text)) {
    if (p.fn in setOf("unknown", "home_info", "ac_control", "heat_control") && p.direction in setOf("none", "off")) {
        p.fn = "heat_control"
        p.direction = "on"
    }
}

// v126: 창문 닫아야 할 것 같다 → door_control/close (완곡 의무형)
if (Regex("창문\\s*(?:좀\\s*)?(?:을\\s*)?닫아야\\s*(?:할\\s*것\\s*같|겠|하나)").containsMatchIn(text)) {
    p.fn = "door_control"
    p.direction = "close"
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

// v130: 소등 → light_control/off (소등=끄다 직접 표현)
if (Regex("소등\\s*(?:해|시켜|부탁|드려|해줘|해주세요)").containsMatchIn(text)) {
    p.fn = "light_control"
    p.direction = "off"
    p.execType = "control_then_confirm"
}

// v130: 창문 개방 → door_control/open (개방=열다 직접 표현)
if (Regex("창문\\s*(?:을\\s*)?개방\\s*(?:해|시켜|부탁|드려|해줘|해주세요)").containsMatchIn(text)) {
    p.fn = "door_control"
    p.direction = "open"
    p.execType = "control_then_confirm"
}

// v130: 점화/가동 + 보일러/난방 → heat_control/on
if (Regex("(?:보일러|난방|히터)\\s*(?:\\S+\\s*)?(?:점화|가동|작동)\\s*(?:해|시켜|부탁)").containsMatchIn(text)) {
    if (p.fn in setOf("heat_control", "unknown") && p.direction in setOf("none", "off")) {
        p.fn = "heat_control"
        p.direction = "on"
    }
}

// v130: 환기/냉방 가동 부탁드립니다 → dir=on (가동=켜다)
if (Regex("(?:환기|냉방|에어컨|공기청정)\\s*(?:좀\\s*)?가동\\s*(?:해|시켜|부탁|드려)").containsMatchIn(text)) {
    if (p.fn in setOf("vent_control", "ac_control", "param_direction", "none", "off")) {
        p.direction = "on"
    }
}

// v130: 더위/열기로 녹아내릴 것 같아 → ac_control/on (극단적 더위 비유)
if (Regex("(?:더위|열기)\\s*(?:에|로|때문에)?\\s*(?:녹아내릴|녹아버릴|녹아|녹을)\\s*(?:것\\s*)?(?:같|것)").containsMatchIn(text)) {
    if (p.fn in setOf("heat_control", "unknown", "weather_query", "ac_control") && p.direction in setOf("none", "off")) {
        p.fn = "ac_control"
        p.direction = "on"
    }
}

// v130: 답답해/갑갑해 미치겠어/죽겠어 → vent_control/on (밀폐 답답함)
if (Regex("(?:답답해|갑갑해|숨막혀)\\s*(?:미치겠어|죽겠어|못\\s*살겠어|살겠어)").containsMatchIn(text)) {
    if (p.fn in setOf("weather_query", "unknown", "home_info") && p.direction in setOf("none", "off")) {
        p.fn = "vent_control"
        p.direction = "on"
    }
}

if (Regex("^열어줘\\s*창문").containsMatchIn(text)) {
    p.fn = "door_control"
    p.direction = "open"
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

// v127: 동굴/지하실 비유 → light_control/on (어두운 공간 비유 = 불 켜기)
if (Regex("동굴\\s*(?:같|이야|이네|이냐|야|인가|이에요)|(?:방|집|여기|실내)\\s*(?:이\\s*)?(?:지하실|감방|감옥)\\s*(?:같|이야|이네)").containsMatchIn(text)) {
    if (p.fn in setOf("unknown", "light_control", "home_info") && p.direction in setOf("none", "off")) {
        p.fn = "light_control"
        p.direction = "on"
    }
}

// v127: 땀이 비/폭포처럼 → ac_control/on (땀 비유 = 더워서)
if (Regex("땀이\\s*(?:비|폭포|강|빗물)\\s*(?:같이|처럼|마냥)?\\s*(?:흘러|나|쏟아|줄줄)").containsMatchIn(text)) {
    if (p.fn in setOf("weather_query", "unknown", "home_info") && p.direction in setOf("none", "off")) {
        p.fn = "ac_control"
        p.direction = "on"
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

// v126: 어르신/노인 hearsay + 덥다/춥다 → ac_control/on 또는 heat_control/on
if (Regex("(?:어르신|노인|할머니|할아버지|어머니|아버지|부모님|노부모|엄마|아빠)\\s*(?:이|가|께서)?\\s*(?:덥다고|더우시다고|더워하시|더워한다고|너무\\s*덥다)").containsMatchIn(text)) {
    if (p.fn in setOf("unknown", "home_info", "weather_query", "heat_control", "ac_control") && p.direction in setOf("none", "off")) {
        p.fn = "ac_control"
        p.direction = "on"
    }
}

if (Regex("(?:어르신|노인|할머니|할아버지|어머니|아버지|부모님|노부모|엄마|아빠)\\s*(?:이|가|께서)?\\s*(?:춥다고|추우시다고|추워하시|추워한다고|너무\\s*춥다)").containsMatchIn(text)) {
    if (p.fn in setOf("unknown", "home_info", "weather_query", "ac_control", "heat_control") && p.direction in setOf("none", "off")) {
        p.fn = "heat_control"
        p.direction = "on"
    }
}
