# v2 End-to-End Pipeline — AI기대응답 생성

## 개요

v1 파이프라인(multi-head까지)을 백업하고 새 v2를 구축.
v2는 multi-head 결과 + value를 받아 **르엘 AI기대응답 스타일 문장**까지 생성.

## 파일

| 파일 | 역할 |
|------|------|
| `scripts/deployment_pipeline_v1_baseline.py` | v1 백업 (수정 금지) |
| `checkpoints/nlu_v28_v46_ensemble_v1_baseline.onnx` | v1 모델 백업 |
| `scripts/response_generator_v2.py` | 응답 문장 생성기 (새) |
| `scripts/deployment_pipeline_v2.py` | end-to-end 파이프라인 v2 |
| `scripts/eval_v2_similarity.py` | 219 시나리오 similarity 평가 |
| `scripts/eval_v2_ruel_scenarios.py` | 카테고리별 응답 샘플 |
| `data/eval_v2_ruel_results.csv` | 전체 219 결과 |

## 성능 (219 르엘 시나리오 대비)

| 지표 | 값 |
|------|-----|
| **Exact match** | **128/219 (58.4%)** |
| **High sim (≥0.5)** | 67/219 (30.6%) |
| **Usable (exact + high)** | **195/219 (89.0%)** |
| Medium sim (≥0.3) | 10/219 (4.6%) |
| Low sim (<0.3) | 14/219 (6.4%) |

남은 low sim은 대부분:
- 비상 3건 — TS expected 공란 (action만 트리거) → unfixable sim
- 같은 utterance 2번 라벨 (1대/여러대 버스) → 둘 중 하나만 일치
- 특정 placeholder 차이 ("2km 거리" vs "반경 3km")

## 핵심 아키텍처

```
user utterance
  ↓
preprocess (STT 교정, 사투리 정규화)
  ↓
ensemble ONNX inference (v28+v46, CPU)
  ↓
post-processing rules (~40개)
  ↓
room extraction
  ↓
DST (slot filling, correction, bare verb)
  ↓
response_generator_v2 (multi-head → AI기대응답)
  ├── Emergency 최우선
  ├── 예약 취소 재분류
  ├── 공간명 없는 CTC → clarify
  ├── 특수 패턴 매칭 (~80개 SPECIFIC_PATTERNS)
  └── exec_type 분기
      ├── CTC → control_response (기기별 템플릿)
      ├── query → query_response (상태값 placeholder)
      ├── judge → judge_response (옷차림/세차/외출 판단)
      ├── clarify → clarify_response (fn별 명확화)
      ├── direct → direct_response (짧은 응답)
      └── unknown → unknown_response (조립)
```

## v2 우위점 (vs Single Intent)

**Single intent**: intent가 틀리면 이상한 응답 or 무조건 unknown.

**v2**: 부분적으로 일치하는 헤드만 있어도 합리적 응답 조립.

예시:
- "거실 뭐 좀 켜" → fn=light_control(clarify) + room=living → "월패드로는 거실과 주방 조명 조절이 가능해요"
- "안방 그거 꺼줘" → fn inherit + room=bedroom_main + dir=off → "안방 난방을 끕니다"
- "너무 답답해" → fn=vent_control + dir=on → "실내 환기시스템을 켰습니다"
- Unknown이어도 device/verb keyword hint 있으면 조립 응답

## 응답 템플릿 예시

| 발화 | 응답 |
|------|------|
| 지금 집 상태 어때? | 현재 각실 조명과 난방이 켜져 있고 실내 온도는 00도 입니다. |
| 거실 불 켜줘 | 네, 거실 조명을 켭니다. |
| 안방 에어컨 23도로 | 네, 안방 에어컨 온도를 23도로 설정합니다. |
| 전체 난방 좀 올려줘 | 네, 전체 난방을 22도에서 24도로 설정합니다. |
| 가스 밸브 잠금해 | 네, 가스 밸브를 잠금 처리하였습니다. |
| 거실 전동커튼 열어줘 | 네, 거실 전동커튼을 열고 있습니다. |
| 오늘 날씨 어때? | 오늘 OO구 OO동 날씨는 맑고 최고 22도, 최저 12도... |
| 30분 후에 난방 꺼 | 네, 30분 뒤에 난방을 끄겠습니다. |
| 가스 냄새 나 | ⚠️ 비상 경보를 울렸습니다. 119/112에 연락하시고... |

## 사용법

```python
from deployment_pipeline_v2 import DeploymentPipelineV2

p = DeploymentPipelineV2()
result = p.process("거실 불 켜줘")
print(result['response'])  # "네, 거실 조명을 켭니다."

# DST (멀티턴)
result = p.process("안방도", use_dst=True)  # "네, 안방 조명을 켭니다."
p.reset_dst()  # 세션 초기화

# 복합 명령
result = p.process_compound("거실 불 끄고 안방 불 켜")
for action in result['actions']:
    print(action['response'])
```

## v1 백업 사용

```python
from deployment_pipeline_v1_baseline import DeploymentPipeline  # v1
# 기존 동작 그대로
```

## 확장 방법

새로운 응답 패턴 추가:
1. `response_generator_v2.py`의 `SPECIFIC_PATTERNS`에 (regex, response) 추가
2. 또는 `control_response` / `query_response` / `judge_response`에 케이스 추가
3. `python3 scripts/eval_v2_similarity.py`로 검증

## 평가 결과 파일

- `data/eval_v2_ruel_results.csv`: 219개 (utterance, expected, actual, fn, exec, dir) 전체 결과
