# 사전학습 임베딩 전이학습 — 왜 필요하고 어떻게 했나

## 기존 방식의 문제

### 우리가 했던 것: CNN from scratch

```
Embedding(32000, 128) ← 랜덤 초기화
  → Conv1d(3) → ReLU → BN
  → Conv1d(5) → ReLU → BN
  → Conv1d(7) → ReLU → BN
  → Conv1d(3) → ReLU → BN
  → MeanPool → FC(94)
```

**문제:** 임베딩이 랜덤이라 모델이 "추워"와 "쌀쌀해"가 비슷한 의미인 걸 모른다.
학습 데이터에서 **각각** 배워야 하므로 데이터가 엄청 많이 필요하고, 안 본 표현은 못 맞춘다.

### NLU intent classification 표준 방법

```
[2024 표준]
사전학습된 BERT/RoBERTa → fine-tune (분류 head만 학습)
→ 한국어 수억 문장으로 사전학습된 언어 지식 활용
→ intent당 20~50개 데이터만 있어도 높은 정확도

[우리 방식]
랜덤 CNN → 처음부터 전부 학습
→ 언어 지식 없음 → 만 개 이상 데이터 필요, 일반화 약함
```

| | 우리 (CNN from scratch) | 표준 (pretrained fine-tune) |
|---|---|---|
| 임베딩 | 랜덤 128dim | **BERT 768dim 사전학습** |
| 언어 이해 | 학습 데이터에서만 | **한국어 전체에서 미리 학습** |
| 필요 데이터 | 많음 (만 개+) | **적음 (intent당 20~50개)** |
| 일반화 | 약함 | **강함** |

## 해결: 사전학습 임베딩을 CNN에 주입

### 구조

```
[ko-sbert-sts 사전학습 모델]
  └─ word_embeddings: (32000, 768) ← 이것만 가져온다

[우리 모델]
Embedding(32000, 768) ← ko-sbert에서 가져온 가중치 (frozen)
  → Linear(768, 256) ← 차원 축소 (학습)
  → Conv1d(3,5,7,3) + ReLU + BN + Dropout ← CNN body (학습)
  → MeanPool → FC(94) ← 분류기 (학습)
```

**핵심:** 임베딩은 고정(frozen), CNN 부분만 학습.
- 전체 파라미터: 25.98M (임베딩 24.58M + CNN 1.40M)
- **학습 파라미터: 1.40M만** (임베딩은 안 건드림)

### 왜 가능한가

1. **같은 Tokenizer:** 우리 tokenizer가 ko-sbert-sts와 완전 동일 (vocab 32000, 같은 token ID)
2. **Embedding은 lookup table:** 단어 → 벡터 변환 테이블일 뿐. CNN 입장에서는 입력 벡터만 오면 됨
3. **사전학습 임베딩의 장점:** "추워"와 "쌀쌀해"가 이미 가까운 벡터로 학습되어 있음

```
랜덤 임베딩:  "추워" = [0.3, -0.1, ...]  "쌀쌀해" = [0.8, 0.5, ...]  ← 완전 다름
사전학습:     "추워" = [0.9, 0.2, ...]  "쌀쌀해" = [0.85, 0.18, ...] ← 이미 비슷!
```

### frozen vs fine-tune

| 방식 | 르엘 정확도 | 이유 |
|------|-----------|------|
| **frozen (임베딩 고정)** | **94.0%** | 과적합 방지, 사전학습 지식 보존 |
| fine-tune (임베딩도 학습) | 93.6% | 학습 데이터에 과적합 |

→ frozen이 더 좋다. 임베딩을 건드리면 적은 학습 데이터에 맞춰져서 일반화 능력이 떨어진다.

## 결과

| 모델 | 임베딩 | 르엘 219개 | 간접 표현 | 비유 표현 |
|------|--------|-----------|---------|----------|
| CNN (랜덤 128d) | 랜덤 | 92.2% | 93.4% | 37.2% |
| **CNN (ko-sbert 768d frozen)** | **사전학습** | **96.3%** | **96.1%** | **51.2%** |

- 르엘: 92.2% → **96.3%** (+4.1%p)
- 비유: 37.2% → **51.2%** (+14%p) — 학습에 없는 비유도 의미적으로 가까우면 잡음

---

## 한국어 사전학습 모델 비교

### 주요 모델 (vocab 32000, 우리 tokenizer와 호환)

| 모델 | 기반 | 학습 데이터 | 특징 | 크기 |
|------|------|-----------|------|------|
| **KLUE-RoBERTa-base** | RoBERTa | 63GB (모두의말뭉치+CC100+나무위키+뉴스) | **KLUE 벤치마크 1위**, 가장 강력 | 111M |
| KLUE-BERT-base | BERT | 63GB | KLUE 기준 모델 | 111M |
| ko-sbert-sts | KLUE-BERT + STS fine-tune | KorNLI + KorSTS | 문장 유사도 특화 | 111M |
| KoSimCSE-roberta | RoBERTa + SimCSE | 비지도 학습 | 유사도 (비지도) | 111M |

### 비호환 모델 (vocab 다름)

| 모델 | vocab | 비고 |
|------|-------|------|
| KoELECTRA-base | 35000 | 우리 tokenizer(32000)와 안 맞음 |
| KoBERT | 8002 | SKT, vocab 너무 작음 |
| HanBERT | 54000 | TwoBlock AI |

### 선택 기준

- **KLUE-RoBERTa-base가 최선** — 가장 많은 데이터로 학습, KLUE 벤치마크 전체 1위
- ko-sbert-sts는 KLUE-BERT 기반 — RoBERTa보다 하위
- vocab 32000 동일 → **임베딩만 교체하면 바로 사용 가능**

### KLUE-RoBERTa vs ko-sbert-sts

| | ko-sbert-sts (현재) | KLUE-RoBERTa (시도 예정) |
|---|---|---|
| 기반 | KLUE-BERT | **KLUE-RoBERTa (더 강력)** |
| 사전학습 | 63GB + STS fine-tune | **63GB (순수 사전학습, 더 일반적)** |
| 벤치마크 | STS 특화 | **KLUE 전체 1위** |
| 임베딩 차원 | 768 | 768 |
| vocab | 32000 | 32000 |
| 호환성 | **검증 완료** | **토큰 ID 동일 확인** |

---

## 다음 단계

1. **KLUE-RoBERTa 임베딩으로 교체** → ko-sbert보다 더 좋은 결과 기대
2. 비유 표현 정확도 추가 개선
3. NPU 변환 테스트 (CNN body만 NPU, 임베딩은 CPU)
