"""Lesson 2: our greedy loop; Transformers only performs forward.

Read run_greedy first. No model.generate(), scheduler, batching or GPU benchmarks.
"""
import argparse
import json
from pathlib import Path

import torch

from week3.reference_runtime import forward_last, load_reference, normalize_eos, validate_ids


def run_greedy(model, input_ids, eos_ids, max_new_tokens=16, *, step=forward_last,
               observe_cache=None):
    """Batch=1, no padding. Return plain Python values, never a live cache."""
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens 必须为非负整数")
    vocab = None if model is None else model.config.text_config.vocab_size
    validate_ids(input_ids, vocab)
    eos_ids = normalize_eos(eos_ids, vocab)

    # 1. Every request starts with fresh state. Zero budget performs no forward.
    generated_ids = []
    trace = []
    processed_tokens = 0
    stop_reason = "length"
    cache = None
    logits = None
    current_ids = input_ids
    phase = "prefill"
    try:
        for index in range(max_new_tokens):
            # 2. First iteration is Prefill; later iterations input exactly one token.
            phase = "prefill" if index == 0 else "decode"
            start = processed_tokens
            logits, cache = step(model, current_ids, processed_tokens=start,
                                 cache=cache, use_cache=True)
            if (not isinstance(logits, torch.Tensor) or logits.ndim != 2
                    or logits.shape[0] != 1 or logits.shape[1] == 0
                    or (vocab is not None and logits.shape[1] != vocab)
                    or not torch.isfinite(logits).all().item()):
                raise ValueError("step 必须返回有限的 [1,V] logits")
            if cache is None:
                raise ValueError("step 未返回 cache")
            normalize_eos(eos_ids, logits.shape[1])
            processed_tokens += current_ids.shape[1]

            # 3. Keep tensor input and integer output distinct.
            next_ids = logits.argmax(dim=-1, keepdim=True)  # [1,1]
            next_token_id = int(next_ids.item())
            generated_ids.append(next_token_id)

            # 4. Stop BEFORE feeding the final chosen token back to the model.
            finished = False
            if next_token_id in eos_ids:
                stop_reason, finished = "eos", True
            elif len(generated_ids) >= max_new_tokens:
                stop_reason, finished = "length", True

            event = {"phase": phase, "input_ids": current_ids.tolist()[0],
                     "position_start": start, "position_end_exclusive": processed_tokens,
                     "processed_tokens": processed_tokens, "token_id": next_token_id,
                     "generated_tokens": len(generated_ids),
                     "stop_reason": stop_reason if finished else None}
            if observe_cache is not None:
                metadata = observe_cache(cache, processed_tokens)
                # Detach diagnostics by JSON roundtrip; reject tensors/nonfinite metadata.
                event["cache"] = json.loads(json.dumps(metadata, allow_nan=False))
            trace.append(event)
            if finished:
                break
            current_ids = next_ids  # [1,1], not the Python integer next_token_id.
    except Exception as exc:
        exc.add_note(f"phase={phase}, processed_tokens={processed_tokens}")
        raise
    finally:
        # Drop our references on success AND error. This does not empty CUDA allocator.
        cache = None
        logits = None

    return {"input_ids": input_ids.tolist()[0], "output_ids": generated_ids,
            "processed_tokens": processed_tokens, "generated_tokens": len(generated_ids),
            "stop_reason": stop_reason, "trace": trace}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/home/ubuntu/huggingface/Qwen3.5-4B"))
    parser.add_argument("--prompt", default="用一句话解释 KV Cache。")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.max_new_tokens < 0 or not args.prompt.strip():
        parser.error("需要非空提示词与非负生成预算")
    if args.output.exists():
        parser.error("输出文件已存在；拒绝覆盖，请选择新的文件名")
    from week3.inspect_model import inspect_prompt
    model, tokenizer, manifest, eos_ids = load_reference(args.model, args.device)
    prompt = inspect_prompt(tokenizer, args.prompt, thinking=False)
    ids = torch.tensor([prompt["chat_ids"]], dtype=torch.long, device=args.device)
    result = run_greedy(model, ids, eos_ids, args.max_new_tokens)
    result.update({"prompt": args.prompt, "text": tokenizer.decode(
        result["output_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False),
        "model": manifest, "dtype": str(model.dtype), "device": args.device,
        "attention": "eager", "thinking": False, "max_new_tokens": args.max_new_tokens,
        "eos_ids": sorted(eos_ids), "template_sha256": prompt["template_sha256"],
        "scope": "generation only; numerical parity not verified"})
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
