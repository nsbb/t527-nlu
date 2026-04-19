# KoELECTRA 데이터 병합 핸드오프

## 요약

KoELECTRA v8 (건희 팀, `wewonnim/koelectra_wallpad`) 학습데이터 13,540개를
멀티헤드 CNN 5-head 형식으로 자동 변환하여 기존 데이터에 합침.

**결과:** 21,672개 → **34,513개** (+12,841, 중복 699 제거)

## 생성된 파일

| 파일 | 설명 | 용도 |
|------|------|------|
| `data/train_final_v33.json` | 합친 학습데이터 (34,513개) | **학습에 사용** |
| `data/val_final_v33.json` | 기존 val 복사 (2,550개) | 검증용 |
| `data/koelectra_converted_train.json` | 변환된 KoELECTRA train (13,540개) | 참고/디버깅용 |
| `data/koelectra_converted_val.json` | 변환된 KoELECTRA val (1,536개) | **교차검증 테스트셋** |
| `data/koelectra_conversion_report.txt` | 변환 통계 리포트 | 참고용 |

## 학습 방법

```bash
# v33 데이터로 학습 (기존 train 스크립트에 경로만 변경)
python3 scripts/train_cnn_multihead.py \
    --train data/train_final_v33.json \
    --val data/val_final_v33.json \
    --version 33
```

## 변환 규칙

### KoELECTRA flat label → 5축 매핑

79개 라벨 **전부** 자동 변환 (스킵 0). 매핑 로직:

- **fn**: 라벨 prefix로 결정 (예: `light_control_on` → `light_control`)
- **exec_type**: `_control_`/`_open`/`_lock` → `control_then_confirm`, `_query` → `query_then_respond`
- **param_direction**: 라벨 suffix로 결정 (`_on`→`on`, `_off`→`off`, `_up`→`up`, 등)
- **param_type**: 라벨 + 발화문 regex로 결정 (온도/밝기/모드/풍량)
- **judge**: 발화문 regex로 결정 (미세먼지→air_quality, 옷→clothing, 등)

### 애매한 11개 라벨 확정 방식

| KoELECTRA label | → fn | 이유 |
|---|---|---|
| `call_front/guard/neighbor/history` | `home_info` | 단지 서비스 |
| `access_history`, `card_manage`, `visitor_video`, `emergency_report` | `security_mode` | 보안 계열 |
| `cooktop_control_off` | `gas_control` | 가스 계열 |
| `outlet_control_on/off` | `light_control` | 전기 제어 계열 |

### 발화문 기반 보정 (param_type, judge)

라벨만으로 `none`인데 발화문에 힌트가 있으면 보정:
- "에어컨 **온도** 올려줘" → `param_type: temperature`
- "**미세먼지** 어때" → `judge: air_quality`
- "**옷** 뭐 입을까" → `judge: clothing`

## 데이터 분포 변화

| fn | 병합 전 | 병합 후 | 증가 |
|---|---|---|---|
| ac_control | 1,588 | 3,563 | +1,975 |
| light_control | 974 | 2,497 | +1,523 |
| heat_control | 796 | 1,940 | +1,144 |
| home_info | 1,987 | 3,068 | +1,081 |
| security_mode | 876 | 1,857 | +981 |
| vent_control | 1,222 | 2,099 | +877 |
| gas_control | 363 | 1,088 | +725 |
| curtain_control | 718 | 1,295 | +577 |
| vehicle_manage | 696 | 1,245 | +549 |
| door_control | 467 | 1,031 | +564 |
| system_meta | 485 | 962 | +477 |

**특히 부족했던 gas_control(363→1,088), door_control(467→1,031)이 대폭 보강됨.**

## 교차검증 (선택사항)

KoELECTRA val (1,536개)을 교차검증 테스트셋으로 활용 가능:

```bash
# 학습 후 KoELECTRA val로 교차검증
python3 scripts/test_interactive.py \
    --checkpoint checkpoints/cnn_multihead_v33_best.pt \
    --test data/koelectra_converted_val.json
```

이 데이터는 우리 증강 파이프라인을 거치지 않은 **완전히 독립적인 발화문**이라
과적합 여부 판단에 유용함.

## 스크립트

| 스크립트 | 용도 |
|---|---|
| `scripts/convert_koelectra_to_multihead.py` | KoELECTRA JSONL → 멀티헤드 JSON 변환 |
| `scripts/merge_koelectra_data.py` | 변환된 데이터를 기존 train에 합침 |
| `scripts/koelectra_to_multihead_map.py` | 매핑 테이블 분석 (참고용) |

## 주의사항

1. **fn 축 20개 유지** — 새 카테고리 추가 안 함. call→home_info, outlet→light_control로 통합
2. **v33 데이터에 source 필드 있음** — `koelectra_v8`로 구분 가능, 문제 시 필터링 가능
3. **KoELECTRA val은 train에 안 넣었음** — 1,536개는 교차검증용으로 보존
4. 학습 후 combo가 **떨어지면** koelectra 데이터의 라벨 품질 문제일 수 있음
   → `source: koelectra_v8`만 필터링해서 제거 후 재학습
