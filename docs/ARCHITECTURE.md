# Architecture — NLU 배포 파이프라인 상세

**대상**: Android JNI 포팅 개발자, 신규 참여자  
**최종 업데이트**: 2026-04-22 (iter9)

## 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                   │
│  User Voice ─── STT ─── "거실 불 좀 켜줘"                        │
│                                 │                                 │
│                                 ▼                                 │
│                        ┌────────────────┐                         │
│                        │  preprocess    │                         │
│                        │  (STT 교정)     │                         │
│                        └────────┬───────┘                         │
│                                 │                                 │
│                         "거실 불 좀 켜줘"                          │
│                                 ▼                                 │
│                        ┌────────────────┐                         │
│                        │  Tokenizer     │                         │
│                        │  (ko-sbert)    │                         │
│                        └────────┬───────┘                         │
│                                 │                                 │
│                          input_ids [1,32]                         │
│                                 ▼                                 │
│                        ┌────────────────┐                         │
│                        │ Ensemble ONNX  │                         │
│                        │  (v28 + v46)   │                         │
│                        │  105MB FP32     │                         │
│                        └────────┬───────┘                         │
│                                 │                                 │
│                     5 logits (fn/exec/dir/param/judge)             │
│                                 ▼                                 │
│                        ┌────────────────┐                         │
│                        │ apply_post_    │                         │
│                        │ rules (10개)   │                         │
│                        └────────┬───────┘                         │
│                                 │                                 │
│                           preds (5 heads)                         │
│                                 ▼                                 │
│                        ┌────────────────┐                         │
│                        │ extract_rooms  │                         │
│                        │ (정규식)        │                         │
│                        └────────┬───────┘                         │
│                                 │                                 │
│                     preds + rooms [list]                          │
│                                 ▼                                 │
│                        ┌────────────────┐                         │
│                        │ DST update     │                         │
│                        │ (10s timeout)  │                         │
│                        └────────┬───────┘                         │
│                                 │                                 │
│              final = {fn, exec, dir, room, value}                  │
│                                 ▼                                 │
│                        ┌────────────────┐                         │
│                        │ generate_      │                         │
│                        │ response       │                         │
│                        └────────┬───────┘                         │
│                                 │                                 │
│                          "네, 거실 조명을 켰습니다."                │
│                                 │                                 │
│                                 ▼                                 │
│                               TTS                                 │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 구성 요소 상세

### 1. Preprocess (`scripts/preprocess.py`)

**역할**: STT 출력의 오류/형식 정제

**크기**: 210+ 교정 패턴 + 한글숫자 변환

**동작**:
1. 비인쇄 문자 제거
2. 공백 정리 (`re.sub(r'\s+', ' ', text).strip()`)
3. STT_CORRECTION 사전 2-pass 적용 (긴 패턴 먼저)
4. `kr_num_to_arabic()` 한글숫자 → 아라비아 숫자
5. 불필요 부사 제거 ("잠깐", "얼른")

**Android 포팅**:
- `HashMap<String, String> STT_CORRECTION`
- regex `Pattern.compile()` 사용
- 사전 크기 ~8KB, 메모리 부담 없음

### 2. Tokenizer (ko-sbert-sts)

**모델**: `jhgan/ko-sbert-sts`  
**타입**: BertTokenizer (Huggingface)  
**vocab**: 32,000  
**max_len**: 32  
**특수**: [CLS], [SEP], [PAD], [UNK]

**Android 포팅**:
- Java/Kotlin BertTokenizer 포팅 필요
- vocab.txt, tokenizer_config.json 파일 assets에 포함
- 참고: `transformers` Java port 또는 직접 구현

### 3. Ensemble ONNX (`checkpoints/nlu_v28_v46_ensemble.onnx`)

**구조**:
- v28 branch: CNN 5-Head (24M trainable, 1M emb shared)
- v46 branch: CNN 5-Head (same)
- Strategy B 결합: fn=v46, exec/dir/param=v28, judge=v46

**Input**: `input_ids [1, 32]` int64  
**Output**: 5개 logits
- fn [1, 20]
- exec_type [1, 5]
- param_direction [1, 9]
- param_type [1, 5]
- judge [1, 5]

**Android 포팅**:
- ONNX Runtime Android (libonnxruntime.so)
- CPU provider (0.5ms 추론)
- GPU/NPU는 별도 NB 변환 필요

### 4. Post-processing Rules (`scripts/ensemble_inference_with_rules.py`)

**10개 rule (iter8/9)**:

```python
def apply_post_rules(preds, text):
    # 1. param_type 기본 규칙 (dir open/close/stop → param_type=none 등)
    # 2. 밝게 → up, 어둡게 → down
    # 3. 엘리베이터 호출/불러/와줘 → CTC
    # 4. N모드로 → set (ac/heat/vent 한정)
    # 5. 알람/모닝콜 + device 없음 → schedule_manage
    # 6. OOD keywords → unknown
    # 7. 전화 + entity 없음 → unknown
    # 8. unknown → 날씨/뉴스/의료 keyword 복구
    # 9. {room}{device} 좀 {verb} 어순 → CTC
    # 10. curtain pred=open → up/close/stop 수정
    # 11. 블라인드 내려 → close
    # 12. 블라인드만 no-action → stop
    # 13. heat CTC + none → on
    # 14. 화면/음량/알림 → home_info (capability query 제외)
    return preds
```

**Android 포팅**:
- Kotlin `fun applyPostRules(preds: Preds, text: String): Preds`
- regex는 `Pattern.compile("...")` 로 변환
- 각 rule 단위 테스트 권장

### 5. Room Extraction (`scripts/deployment_pipeline.py:extract_rooms`)

**매핑**:
```
거실/리빙 → living
안방/주침실 → bedroom_main
침실/작은방/아이방/서재 → bedroom_sub
주방/부엌 → kitchen
현관/외부 → external
욕실/화장실 → none  # (지원 안 함)
전체/모든 → all
```

**다중 room**: `extract_rooms()` 리스트 반환 ("거실과 안방" → [living, bedroom_main])

### 6. Dialogue State Tracker (`scripts/dialogue_state_tracker.py`)

**상태**:
- prev_fn, prev_exec, prev_dir, prev_room
- prev_value (temperature/time/percent/level/enum)
- prev_time (timeout 체크)
- history[5] (최근 5턴)

**Follow-up 패턴**:
- `room`: "안방도", "거실" (이전 fn/exec/dir 상속, room만 변경)
- `device`: "난방도", "에어컨도" (fn은 NLU 예측, exec/dir 상속)
- `confirm`: "응", "해줘" (이전 턴 전체 반복)
- `correction`: "아니 ~", "아 역시 ~", "다시 ~" (action만 변경)
- `there_too`: "거기도", "여기도" (이전 room 재사용)

**Slot filling (iter9)**:
- "25도 설정" → value=(temperature, 25) 저장
- "더 올려" → value=(temperature, 26) 자동 추론
- "조금 내려" → value=(temperature, 25) 자동 추론

**Timeout**: 10초 (마지막 턴에서 10초 이상 경과 시 상태 초기화)

### 7. Response Generation

**템플릿**:
- 조사 처리 (조명/난방/에어컨 → 을, 가스 밸브/엘리베이터 → 를)
- value가 있으면 "X도로 설정했습니다" 등 구체화
- query_then_respond는 정적 템플릿 (날씨/뉴스 등 예시 응답)
- direct_respond는 fn별 capability 응답

## 성능 특성

### Latency Breakdown (0.67ms total)

| 단계 | 시간 |
|------|:---:|
| preprocess | ~0.02ms |
| tokenize | ~0.05ms |
| ONNX inference | ~0.45ms (ensemble) |
| apply_rules | ~0.05ms |
| room extract | ~0.02ms |
| DST update | ~0.02ms |
| response gen | ~0.06ms |

### 메모리

| 구성 | 크기 |
|------|:---:|
| ONNX weights | 105MB FP32 |
| Tokenizer vocab | 500KB |
| Preprocess dict | 8KB |
| DST state | <1KB |
| **총계** | ~106MB |

**NPU 포팅 시**: ONNX → NB 변환으로 ~50% 압축 예상 (int8 quantization)

## 입력/출력 데이터 형식

### Input (사용자 발화)
```python
text = "거실 불 좀 켜줘"  # UTF-8 Korean string
```

### Output (Action Struct)
```python
{
  "raw": "거실 불 좀 켜줘",
  "preprocessed": "거실 불 좀 켜줘",
  "fn": "light_control",
  "exec_type": "control_then_confirm",
  "param_direction": "on",
  "room": "living",
  "rooms": ["living"],
  "value": None,
  "dst_applied": False,
  "response": "네, 거실 조명을 켰습니다."
}
```

## Thread Safety

- ONNX session: thread-safe (multiple sessions run in parallel)
- Tokenizer: thread-safe
- DST: **NOT thread-safe** — 세션마다 새 인스턴스 필요 (사용자별)
- Preprocess: functional (thread-safe)

## 확장 포인트

### 새로운 rule 추가
1. `apply_post_rules()`에 조건 추가
2. `regression_test_iter9.py`에 assertion 추가
3. TS + KE 테스트 → regression 없으면 적용

### 새로운 device 추가
1. `ROOM_MAP` 업데이트 (room만 추가 시)
2. `FN_LABELS` 업데이트 (새 fn 시, 재학습 필요)
3. 응답 템플릿 추가

### 새로운 STT 패턴
1. `STT_CORRECTION` dict에 추가
2. 2-pass preprocess로 자동 적용
