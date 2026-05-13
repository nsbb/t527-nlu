# t527-nlu Knowledge Wiki

이 디렉토리는 `t527-nlu` 프로젝트의 **persistent current-truth wiki** 입니다.

팀 wiki (`wewonnim/t527_llm_wiki`)의 schema를 그대로 따르며, 추후 팀 wiki에 흡수/병합 가능합니다.

## 구조

- `index.md` — entry point
- `log.md` — ingest/update 연대기
- `wiki/` — current-truth 페이지 (Current state / Known-good / Open issues / Related / Last updated)
- `raw/` — source notes (검증된 fact의 evidence)
- `schema/` — 운영 규칙 (팀 wiki와 동일)

## 읽는 순서

1. `index.md`
2. 관련 `wiki/...` 페이지
3. 필요 시 `raw/source-notes/...`
4. 그 다음에만 실제 코드/리포트

## 새 정보 ingest 시 5개 모두 갱신

1. `raw/catalog.tsv`
2. `raw/source-notes/<id>.md`
3. 관련 `wiki/...`
4. `index.md`
5. `log.md`

상세 규칙: `schema/AGENTS.md`.

## 팀 wiki와의 관계

- 팀 wiki: `/home/nsbb/travail/claude/T527/wewonnim/t527_llm_wiki/`
- 우리 wiki: 이 디렉토리
- 같은 schema. 추후 팀장님 wiki의 `wiki/projects/t527-nlu.md` 등으로 흡수 가능.
