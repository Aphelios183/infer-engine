# Lesson 3 progress
Plan: archive/development/superpowers/plans/2026-09-25-qwen35-text-baseline.md, Task6 course slice.
Ruling: Continue in user-selected main learning directory; preserve user's reference_runtime.py comment edit without staging it.
Ruling: Reuse run_greedy observe_cache callback, no extra loop/scheduler or CLI flag; cost: experiment entry is archived diagnostic script rather than production CLI.
Ruling: Validate value changes/history only for selected original layers 0 and3; all32 layer shapes/dtypes/lengths recorded; cost: do not claim all-layer numerical correctness.
Ruling: B request fresh-start length checked; full numerical isolation/exception recovery belongs to Task7 and is not claimed complete.
CPU: 10 tests RED (missing observer), then GREEN; full71 tests/69pass/2skip.
GPU: A35->38, B13->16, each Prefill+3Decode; selected full history preserved, selected linear shape same and values changed. State bytes exclude diagnostic clones.
Review: Beauvoir independent read-only review; no Critical/Important.
Minor deferred: zero-byte storage_index can conflate distinct empty storages; totals unaffected, documented in report.
Minor deferred: dedicated tensor-identity/reference-retention test coverage.
Review declined GPU/runtime/docs checked by main via actual run and report; generic compatibility and numerical parity outside current scope.
