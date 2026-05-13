# Knowledge Wiki Agent Rules

This directory is the persistent knowledge layer for `/Users/gunhee/T527`.

## Layers

- `raw/`: immutable source pointers, snapshots, and observation records
- `wiki/`: maintained current-truth pages
- `schema/`: operating rules for future agents

## Read order for answering questions

1. `index.md`
2. relevant `wiki/...` pages
3. `raw/source-notes/...`
4. only then drill into live project files or raw snapshots if needed

## Mandatory ingest workflow

Every new logical source ingest must produce all five outputs:

1. add or update one row in `raw/catalog.tsv`
2. create one `raw/source-notes/<source_id>.md`
3. update one or more `wiki/...` pages
4. update `index.md`
5. append one dated entry in `log.md`

## Page conventions

Every maintained wiki page should contain these sections in this order:

- `Current state`
- `Known-good settings`
- `Open issues`
- `Related sources`
- `Last updated`

Keep pages short and current. Historical detail belongs in `raw/` or `log.md`.

## Truth model

- `wiki/` is the current synthesis.
- `raw/` is the evidence and historical substrate.
- If a historical handoff conflicts with code-backed current state, prefer code-backed current state and record the conflict in the wiki.

## Lint checklist

Run a manual health check at least weekly or every five source ingests:

- stale claims that no longer match code or reports
- orphan pages with no inbound references from `index.md` or overview pages
- contradictory settings across project and model pages
- historical notes that have not been distilled into current-truth wiki pages
