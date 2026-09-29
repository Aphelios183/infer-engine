# Lesson 4 execution ledger

Plan: archive/development/superpowers/plans/2026-09-25-qwen35-text-baseline.md, Task 7.

- Ruling: user requests direct real-model verification; keep a small CPU verifier guardrail suite, not another teaching exercise. Actual model integration is the main acceptance evidence.
- Ruling: continue in user-selected main-directory workflow; preserve existing user comment in reference_runtime.py and prior uncommitted Lesson 4 documents. No model/dependency edits.
- Scope: same-model BF16 full recompute vs cached Prefill + four fixed Decode steps; short pure-text batch=1. Three existing acceptance prompts plus ordinary T=1. Request B baseline compared with B after A success and after injected stateful-forward error.
- CPU guardrails: initial five tests all failed because module missing, then all passed. Includes forced cache corruption detected at first Decode, fixed input/position trace, nonfinite/zero logits, matching top1 but large error, and approval/fingerprint rejection.
- Calibration follows independent texts, BF16 repeat full recompute, then FP32 full recompute; no concurrent GPU weight copies. Candidate tolerances remain unapproved until review. Calibration data and code fingerprint saved.
- Initial real run failed before numerical experiments: fingerprint assumed optional generation_config.json existed. Actual pinned model snapshot has no such file. Fix records explicit null for that file; required assets still hashed. Error log preserved.
- Calibration review rejected candidate thresholds before acceptance diagnostic run: max_abs limit 1.28086853 vs minimum FP32 margin 0.02321339. Keep approved=false. No test-data-driven threshold changes.
- Final review: independent read-only reviewer found one Important isolation coverage issue. Prefill-only injection did not exercise an established caller-owned cache. Added regression (failed on missing fault evidence), then inject after successful Prefill and stateful Decode. B also runs directly through run_greedy after A; fixed-prefix B remains complementary. Save traces and exact fault position.
- Final: fixed isolation coverage — red then green; full final regression 77 discovered, 75 passed, 2 skipped. Original guardrail red/green logs and initial bad test-setup log retained honestly.
- Review declined runtime/CUDA/calibration: main agent supplies real run evidence and rejects broad tolerance. Non-CUDA devices, batching, multimodal, long context and performance remain explicitly out of scope.
- Ruling: CLI candidate filename is derived from output stem, preserving multiple experiment versions rather than one shared tolerance filename. Calibration and diagnostic code hashes may differ after isolation-only fix; no approval is transferred across fingerprints.
- Formal Task 7 completion is blocked by unapproved tolerance, not by absence of real-model execution. Do not mark Week 3 numerical acceptance complete.
