# SDD ledger — plan: docs/superpowers/plans/2026-09-25-qwen35-text-baseline.md

Execution: native, course-by-course, explicitly approved by user.
Workspace: /home/ubuntu/infer-engine/.worktrees/qwen35-week3 (learn/qwen35-week3).
Spec: docs/superpowers/specs/2026-09-25-qwen35-text-baseline-design.md (approved).

Ruling: Deliver Tasks 1–2 as the first lesson and stop at the learner exercise — the user explicitly requested course-coupled, lesson-by-lesson execution, overriding the generic execute-all-at-once skill default — cost: later tasks remain unimplemented until their lessons.
Ruling: Use remote Git worktree fallback — the app native worktree tool targets the Windows task repository, not the SSH-hosted infer-engine repository — cost: remote worktree must be managed explicitly.
Ruling: Preserve seven existing modified/untracked learning files as an unchanged baseline snapshot only on the new isolated branch — avoids losing in-progress work or staging user changes on main — cost: baseline snapshot commit must be considered during eventual integration.

Pre-flight: Task 1 -> 2: describe_config dict and pure-CPU inspection; matching signatures.
Pre-flight: Task 2 -> 5: inspect_prompt returns chat_ids; changing default model does not imply assets exist.
Pre-flight: Task 3 -> 4: validate_assets requires complete weight manifest; matching signatures, deferred.
Pre-flight: Task 4 -> 5/7: forward_last returns [1,V] logits and cache; matching signatures, deferred.
Pre-flight: Task 6 -> 5/7: snapshot_cache requires layer_types, observer binds it in closure; compatible.
Pre-flight: Task 7 -> 8: diagnostic/needs_review must never count as pass; preserved.

Todo:
- [ ] Task 1: config dispatch, validation, arithmetic.
- [ ] Task 2: offline CLI, prompt inspection, first lesson migration.
- [ ] Task 3: pinned model assets.
- [ ] Task 4: reference forward.
- [ ] Task 5: generation loop.
- [ ] Task 6: state observation.
- [ ] Task 7: numerical verification.
- [ ] Task 8: report and learning acceptance.

No weights downloaded, dependencies changed, or GPU kernels run for this lesson.
Task 1: complete (commits e6ecdaa..d71b66e, tests: env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m unittest week3.test_inspect_model -v → OK)

Task 2: Ruling: Add --config-file for an explicitly labelled teaching fixture — Lesson 1A must run offline before real assets exist — cost: one additional CLI argument to maintain.
Task 2: Ruling: Separate unit tests from opt-in tokenizer integration — missing assets must not masquerade as passing integration — cost: real Qwen3.5 integration remains pending Lesson 1B.
Task 2: Ruling: Rebase the course-plan replacement on the actual worktree file after a full-file patch rejected a stale local baseline — inspected the current plan before replacing it, preserving the adopted handwritten-loop objective and completed Week 2 status — cost: initial transfer reject retained in ignored diagnostics.
Milestone review: independent read-only reviewer approved Tasks 1–2 with two minor follow-ups, no Critical/Important findings.
Milestone: minor (deferred): Expose template source and full decoded text in the diagnostic report; current hashes/rendered input plus round-trip check do not constitute the full planned evidence report.
Milestone: minor (deferred): Add mocked normal-CLI loader regression tests for local_files_only/trust_remote_code and fixture rejection; current implementation retains these guards.
Milestone: Ruling: Course docs are author-checked, not independently reviewed — checked curriculum scope, file links, transferred hashes and current remote tests — cost: pedagogical clarity still needs learner feedback.
Milestone: Ruling: Tasks 3+ and actual asset provenance/tokenizer behavior remain unjudged and unimplemented — user requested lesson-by-lesson work — cost: do not claim native adaptation or complete Task 2 diagnostic reporting.
Remote verification: Python 3.11.15 CPU regression ran 40 tests, 38 passed and 2 skipped; historical real tokenizer separately passed 1 test. Fixture CLI emitted 128 MiB Full Attention KV and null/unmeasured linear state.
Teaching checkpoint: Lesson 1A exercises not yet answered. Tasks 3–8 remain pending. Keep this ledger for continuation; do not merge the branch automatically.
Task 2: complete (commits d71b66e..5103129, tests: env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTHONPATH=week1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m unittest week1.test_profile_attention week2.verify_kv_cache week2.test_benchmark_kv_cache week3.test_inspect_model week3.test_tokenizer_integration -v → OK (skipped=2))
