# API Usage — DeploymentPipeline 사용 가이드

## Python API

### 초기화

```python
from scripts.deployment_pipeline import DeploymentPipeline

# 기본 초기화
pipeline = DeploymentPipeline()

# 명시적 경로
pipeline = DeploymentPipeline(
    onnx_path='checkpoints/nlu_v28_v46_ensemble.onnx',
    tokenizer_path='tokenizer/',
    timeout=10  # DST timeout 초
)
```

### 기본 사용

```python
# 단일 발화 처리
result = pipeline.process("거실 불 켜줘")

# 결과 구조
{
    "raw": "거실 불 켜줘",
    "preprocessed": "거실 불 켜줘",
    "fn": "light_control",              # 20 classes
    "exec_type": "control_then_confirm", # 5 classes
    "param_direction": "on",             # 9 classes
    "room": "living",                    # primary room
    "rooms": ["living"],                 # all detected rooms
    "value": None,                       # (type, number) or None
    "dst_applied": False,                # DST가 수정했는지
    "response": "네, 거실 조명을 켰습니다."
}
```

### 멀티턴 대화

```python
# 세션 시작
pipeline.reset_dst()

# Turn 1
r1 = pipeline.process("거실 불 켜줘")

# Turn 2 — "안방도" → DST가 이전 fn=light_control 상속
r2 = pipeline.process("안방도")
# r2["fn"] = "light_control"
# r2["room"] = "bedroom_main"
# r2["dst_applied"] = True

# Turn 3 — "아니 꺼줘" → DST가 correction 인식
r3 = pipeline.process("아니 꺼줘")
# r3["param_direction"] = "off"

# 세션 종료 (or 10초 timeout 자동)
pipeline.reset_dst()
```

### Slot Filling

```python
pipeline.reset_dst()

r1 = pipeline.process("거실 난방 25도로 맞춰줘")
# r1["value"] = ("temperature", 25)

r2 = pipeline.process("더 올려줘")
# r2["value"] = ("temperature", 26)  # +1 자동 추론
# r2["response"] = "네, 거실 난방을 26도로 설정했습니다."

r3 = pipeline.process("조금 내려줘")
# r3["value"] = ("temperature", 25)
```

### 타이머 설정

```python
r = pipeline.process("30분 후 에어컨 꺼줘")
# r["value"] = ("minute", 30)
# r["param_direction"] = "off"
# r["response"] = "네, 에어컨을 30분 뒤에 껐습니다."
```

### 미지원 기능

```python
r = pipeline.process("택시 불러줘")
# r["fn"] = "unknown"
# r["response"] = "해당 요청은 서버에서 처리합니다."

r = pipeline.process("비밀번호 추천해줘")
# r["fn"] = "system_meta"
# 응답 생성기가 UNSUPPORTED_ACTIONS 체크 → "죄송합니다. 해당 기능은 지원하지 않습니다."
```

## Head 값 레퍼런스

### fn (20 classes)
| 값 | 의미 |
|----|------|
| `light_control` | 조명 |
| `heat_control` | 난방/보일러 |
| `ac_control` | 에어컨 |
| `vent_control` | 환기 |
| `gas_control` | 가스 밸브 |
| `door_control` | 도어락/현관 |
| `curtain_control` | 전동커튼/블라인드 |
| `elevator_call` | 엘리베이터 호출 |
| `security_mode` | 외출/재택 모드 |
| `schedule_manage` | 알람/예약/모닝콜 |
| `weather_query` | 날씨/기온/비/미세먼지 |
| `news_query` | 뉴스/브리핑 |
| `traffic_query` | 교통 정보 |
| `energy_query` | 에너지 사용량 |
| `home_info` | 시간/알림/볼륨/공지 |
| `market_query` | 주식/유가 |
| `medical_query` | 병원/약국 |
| `vehicle_manage` | 전기차/차량 |
| `system_meta` | 시스템 정보/capability |
| `unknown` | 미지원/이해 불가 |

### exec_type (5 classes)
| 값 | 의미 | 예시 |
|----|------|------|
| `control_then_confirm` | 제어 + 확인 응답 | "네, 거실 조명을 켰습니다." |
| `query_then_respond` | 조회 + 정보 응답 | "현재 난방은 25도입니다." |
| `query_then_judge` | 조회 + 판단 응답 | "외출하기에 무리가 없습니다." |
| `direct_respond` | 직접 응답 (조회/제어 아님) | "저는 HDC랩스 AI입니다." |
| `clarify` | 재질의 | "어떤 공간의 기기를 제어할지 말씀해주세요." |

### param_direction (9 classes)
| 값 | 의미 | 예시 |
|----|------|------|
| `on` | 켜기 | "켜줘", "작동", "가동" |
| `off` | 끄기 | "꺼줘", "끄자" |
| `open` | 열기 | "열어줘" |
| `close` | 닫기 | "닫아줘", "잠가줘" |
| `up` | 올리기 | "올려줘", "높여", "세게" |
| `down` | 내리기 | "낮춰줘", "줄여" |
| `set` | 설정 | "N도로", "N분으로" |
| `stop` | 중지 | "멈춰", "중단" |
| `none` | 미지정 (query 등) | - |

### param_type (5 classes)
| 값 | 의미 |
|----|------|
| `temperature` | 온도 |
| `brightness` | 밝기 |
| `mode` | 모드 (냉방/제습/수면) |
| `speed` | 풍량/볼륨 강도 |
| `none` | 해당 없음 |

### judge (5 classes)
| 값 | 의미 |
|----|------|
| `outdoor_activity` | 외출/야외 활동 판단 |
| `clothing` | 옷차림 추천 |
| `air_quality` | 공기질/환기 판단 |
| `cost_trend` | 가격 추세 |
| `none` | 해당 없음 |

## 에러 처리

### 짧은/빈 입력
```python
r = pipeline.process("")
# r["fn"] = "unknown"

r = pipeline.process("???")
# fn은 모델 예측, 응답은 "해당 요청은 서버에서 처리합니다."
```

### 네트워크/IO 오류
DeploymentPipeline은 순수 in-process (no I/O). 외부 의존성은 초기화 시에만 (ONNX 로드).

## Performance Tips

1. **단일 초기화 유지**: DeploymentPipeline은 무거움 (ONNX 105MB 로드). 앱 라이프사이클 동안 한 번만 생성.
2. **멀티유저**: 동일 pipeline 공유 가능. DST는 세션별로 별도 인스턴스 권장.
3. **Latency**: 0.67ms/query CPU. NPU 포팅 시 <0.3ms 예상.

## 테스트

```bash
# 전체 regression 검증
python3 scripts/regression_test_iter9.py

# 종합 데모
python3 scripts/demo_comprehensive.py

# Test Suite 정확도 측정
python3 scripts/ensemble_inference_with_rules.py
```
