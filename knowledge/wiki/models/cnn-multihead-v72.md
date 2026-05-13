# cnn-multihead-v72 — Production NLU 모델 (서버)

## Current state

- 아키텍처: ko-sbert frozen embedding (768→256 proj) + CNN 4L (k=3,5,7,3 residual, BN, ReLU) + global mean pool + 5 heads
- 5 head: fn(20) / exec_type(5) / param_direction(9) / param_type(5) / judge(5)
- Ensemble 전략: fn=v72, exec=v28, dir=v72, param=v28, judge=v72 (head별 다른 모델 선택)
- Production ONNX: `checkpoints/nlu_v28_v72_ensemble.onnx` (md5 8bba94b5..., 110MB)
- 학습 데이터: `data/train_final_v72.json` (32,809개)
- 학습 스크립트: train_v72.py 없음. train_v71.py 복사해서 데이터 경로만 바꿔 사용

## Known-good settings

- input_ids shape `[batch, 32]`, int64
- output names: `fn_logits, exec_logits, dir_logits, param_logits, judge_logits`
- post-rules: `scripts/ensemble_inference_with_rules.py` apply
- CPU 추론 latency: ~0.32ms/utterance (서버)

## Open issues

- v72 cnn_body 단독 ONNX 추출 미실행 → NPU 변환 안 됨
- 르엘 219 RAW (no rule) combo 59.4% — 후처리 rule 비중이 큼 (rule 포함 94.06% 주장, 검증 필요)
- exec 헤드가 v28인데 새 학습 데이터 반영 안 됨 → "뉴스 듣기" 같은 query↔control 혼동 27건

## Related sources

- `raw/source-notes/src-model-version-gap-20260513.md`
- 팀 wiki: `wiki/models/koelectra-nlu.md` — 자매 NLU (WallpadAI용, 84-class flat intent)

## Last updated

- `2026-05-13`
