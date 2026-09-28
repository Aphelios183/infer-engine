# Lesson 2 implementation ledger
Plan: archive/development/superpowers/plans/2026-09-25-qwen35-text-baseline.md
Scope: current course's Tasks 4–5 slice, not completion of Tasks 3–8.
Ruling: Continue in main learning directory — user explicitly consolidated worktree and requested continuing here — cost: new work remains beside prior uncommitted lesson docs; preserve them.
Ruling: Use fixed ModelScope Git/LFS local verification instead of generic Task 3 downloader/manifest — user approved alternate official source, already downloaded/pinned assets — cost: other layouts and revisions are intentionally rejected; generic Task 3 remains pending.
Ruling: Complete course loop only — do not implement state observer and teacher-forced comparisons ahead of lessons — cost: numerical parity remains unverified.
Ruling: Use minimal_generate --max-new-tokens 2 instead of separate reference_runtime --smoke CLI — avoids two frontends for this lesson — cost: historical plan smoke command is not executable validation.
Tests: initial 20 tests RED (missing implementation), then GREEN. Extra-asset regression RED (2 subcases), then GREEN. Full61 discovered/59 passed/2 skipped.
Review: independent read-only reviewer Peirce; Important untracked assets could override pinned checkpoint fixed with fail-first regression.
Minor deferred: more load_reference failure path tests and wrong origin/revision/dirty/LFS-failure tests.
Minor deferred: same-step/model CPU success/failure consecutive isolation coverage; true numerical model-state isolation belongs to later lesson.
Review declined areas: GPU, actual assets, hybrid behavior, remote tests and docs evaluated by main agent; numerical parity intentionally not claimed.
GPU: three short prompts, actual eos/length stops, metadata/IDs/trace retained under week3/results. GPU5 nonexclusive low-load, no performance claim.
Prior lesson files and Week1/2 left intact. No external push. Generic Task3 not completed, Tasks6–8 pending.
