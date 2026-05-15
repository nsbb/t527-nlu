# 르엘 219 GT 셋 — 진짜 정의

## Current state

219 발화에 대한 GT 셋은 **3 종류**가 존재했으며, 그동안 잘못된 셋을 baseline으로 사용해 매번 다른 정확도가 나왔다. 본 결정으로 진짜 GT를 단일 진실로 확정.

### 진짜 GT (사용)
- `data/golden/gt_known_scenarios_v2.json` (204개) + `data/golden/gt_unknown_scenarios.json` (15개) = **219**
- 응답 GT (`response` 필드) + scenario_id/cat/func/stype 메타데이터 포함
- 사람이 직접 검토한 multi-head 라벨

### 자동 매핑 (폐기)
- ~~`data/golden_ruel_219.json`~~ → **삭제됨** (2026-05-15)
- `ruel_scenarios_final.csv` 의 91 intent를 자동 룰로 multi-head 변환
- intent당 라벨 100% 일관성 = 자동 매핑 증거
- 진짜 GT와 라벨 38% (83/219) 다름 — 특히 schedule/system_meta/뉴스 dir

### 회사 원본 (참조용)
- `data/raw/ruel_scenarios_final.csv` — 219줄 (사용자발화문 + 91 intent + AI기대응답)
- 서버 평가 (`eval_v2_ruel_scenarios.py`)가 이걸로 응답 텍스트 정확도 평가

## Known-good settings

- baseline 갱신 명령: `python3 scripts/ci_quick_check.py --update`
- 측정 도구: `scripts/ci_quick_check.py` (0.4초)
- 측정 셋: `golden_99` + `ruel_219` (gt_known_v2 + gt_unknown)
- v72 ensemble + rules 정확도 (2026-05-15 실측):
  - golden_99: fn 94.9% / combo 83.8%
  - ruel_219: fn 96.3% / combo 93.2%

## Open issues

- 디바이스 NPU NB로 진짜 GT 측정 — ART crash 회피 (25 batch + GC) 필요
- v72 cnn_body NB 단독 측정 시 자동매핑 GT 기준 60% → 진짜 GT로 재측정 필요
- 응답 텍스트 정확도 평가 도구 보강 (현재는 exact match만)

## Related sources

- `data/golden/gt_known_scenarios_v2.json`
- `data/golden/gt_unknown_scenarios.json`
- `data/raw/ruel_scenarios_final.csv`
- `wiki/models/version-matrix.md`

## Last updated

- `2026-05-15`
