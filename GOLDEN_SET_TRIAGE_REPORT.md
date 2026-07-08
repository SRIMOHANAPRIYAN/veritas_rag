# Golden Set Review — Triage Report
Reviewer: Claude (Fable 5) · 2026-07-04 · Method: full read of all 100 query↔chunk pairs + corpus-wide ambiguity check (inverted index over 29,501 chunks; `amb` = chunks containing the query's 3 rarest terms)

**Verdict totals: KEEP 16 · FIX 73 · DELETE 11** → 89 queries survive.
Machine-readable decisions (incl. all 73 rewritten queries): `evaluation/benchmarks/golden_set_review_decisions.json`

---

## ⚠ YOUR SPOT-CHECK LIST — 18 queries where my call needs human eyes

Open `golden_set_review.html`, check these cards against my verdict below. Everything else you can trust or skim.

| qid | My verdict | Why flagged |
|---|---|---|
| q037 | DELETE | Signature-block chunk lists "MOSHE MIZRAHY / CEO" between two companies' columns — I judged the attribution unsafe. Confirm delete. |
| q086 | DELETE | Chunk ends mid-sentence ("...period of [***] days from the date of Delivery to") — see systemic note 3 below. |
| q084 | DELETE | Redacted parts-table fragment; query invented a "vehicle". Confirm delete. |
| q062 | DELETE | Policy call: waiver boilerplate exists near-identically in many contracts — I deleted rather than anchor. Agree? |
| q017 | DELETE | Same policy call for no-waiver boilerplate. |
| q069 | FIX | Most extreme ambiguity found (amb=1724: "What is the term of this Agreement?"). My rewrite anchors on "December 31, 2034". Check it reads naturally. |
| q020 | FIX | Rewrite anchors on CISG-waiver + Delaware — verify that's distinctive enough. |
| q035 | FIX | amb=159; rewrite anchors on "VRRM-MFP Shares". |
| q092 | FIX | amb=63; rewrite anchors on "Transferred Equipment". |
| q073 | FIX | amb=54; rewrite anchors on CEO meet-and-confer escalation. |
| q057 | FIX | amb=52; also original had wrong party (JAC doesn't supply). |
| q041 | FIX | amb=23; rewrite shifts topic from "name of element" to insurance requirements — meaning changed, verify. |
| q033 | FIX | Rewrite converts to definition query ("Additional Country") — verify answerable. |
| q024 | FIX | amb=7; anchored on Business/Development Plan. |
| q009 | FIX | Rewrite shifts from "what are the specs" to "who may change the specs" — meaning changed, verify. |
| q002 | FIX | Original asked for a page "name" that doesn't exist; verify my rewrite matches chunk content. |
| q004 | FIX | Same pattern — "name of member" → representations. |
| q063 | FIX | Table-layout chunk (UBS AG) — borderline keep/delete; I salvaged it. |

## Systemic findings (matter beyond this file)

1. **~35 queries were verbatim clause copies** wrapped in "What does…?" — these make BM25 trivially win and would have inflated Gate 1 metrics. All rewritten as paraphrases. Expect honest (lower) numbers, which is what we want.
2. **Three degenerate repetition-loop queries** (q015, q070, q098) — flan-t5-small generation failures, deleted.
3. **Chunk-quality signals for Phase 1 retro** (not golden-set problems, chunker observations): q086's chunk ends mid-sentence (hard max-token cut severing a sentence — worth checking how often this happens: it hurts Phase 4 span alignment); q042/q043/q084 are table-fragment chunks from plain-text contracts where `is_table` detection can't trigger (TXT has no table markup — known limitation, fine for now).
4. **11 deletions leave 89 queries.** That's statistically fine for Gate 1 and the drift baseline. Option: I can draft 11 replacement queries from fresh chunks to restore 100 — say the word.

## Process after your spot-check

1. Tell me your overrides (e.g., "q062 keep it, q041 delete") — or "accept all".
2. I apply all decisions to `golden_set.json` (fixes applied, deletes removed, status → HUMAN_VERIFIED, count updated).
3. Antigravity re-validates schema + chunk_ids, then runs `compare_chunking_strategies` — the real one.

Honest portfolio line this earns: "golden set curated via LLM-assisted review with corpus-wide ambiguity analysis and manual spot-verification."
