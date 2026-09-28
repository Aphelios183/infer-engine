"""Lesson 2 backend: offline pinned model loading and explicit text-only forward.

Transformers owns model math and hybrid cache implementation.
Our code owns input validation, positions/mask and the last-logits contract.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

import torch

MODEL_REVISION = "ed182e32090db791077e12e0f58d22f3daafa173"
MODEL_ORIGIN = "https://www.modelscope.cn/Qwen/Qwen3.5-4B.git"


def verify_local_snapshot(model_dir):
    """Validate this course's existing Git/LFS snapshot, not a general downloader."""
    root = Path(model_dir).resolve()
    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(root), *args], check=True,
                                  capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError("模型 Git/LFS 校验失败；请保留原文件并检查固定版本") from exc
    if git("rev-parse", "--show-toplevel") != str(root):
        raise ValueError("模型目录必须是独立的固定版本仓库")
    if git("rev-parse", "HEAD") != MODEL_REVISION:
        raise ValueError("模型 revision 与本课锁定版本不同")
    if git("remote", "get-url", "origin") != MODEL_ORIGIN:
        raise ValueError("模型来源与本课锁定的 ModelScope 仓库不同")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("模型受跟踪文件已修改，拒绝混用资产")
    tracked = set(git("ls-files", "-z").split("\0"))
    for file in root.rglob("*"):
        relative = file.relative_to(root)
        if relative.parts[0] == ".git":
            continue
        if file.is_symlink() or (file.is_file() and relative.as_posix() not in tracked):
            raise ValueError(f"模型目录含未跟踪或链接资产：{relative}")
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "qwen3_5":
        raise ValueError("仅支持 Qwen3.5")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("权重 index 缺少 weight_map")
    shards = sorted(set(weight_map.values()))
    for name in shards:
        if not isinstance(name, str):
            raise ValueError("非法 shard 路径")
        p = PurePosixPath(name)
        file = root / name
        if (p.is_absolute() or len(p.parts) != 1 or p.suffix != ".safetensors"
                or file.is_symlink() or not file.is_file()
                or file.resolve().parent != root):
            raise ValueError("非法或缺失 shard 路径")
    # Validates LFS content hashes against the pinned Git pointers.
    git("lfs", "fsck")
    return {"source": MODEL_ORIGIN, "revision": MODEL_REVISION,
            "verification": "clean tracked Git files + git lfs fsck",
            "config_sha256": hashlib.sha256((root / "config.json").read_bytes()).hexdigest(),
            "shards": [{"name": name, "bytes": (root/name).stat().st_size} for name in shards]}


def normalize_eos(value, vocab_size=None):
    if isinstance(value, int) and not isinstance(value, bool):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        raise ValueError("eos_ids 必须是整数或非空整数集合")
    if not values or any(type(x) is not int or x < 0 or
                         (vocab_size is not None and x >= vocab_size) for x in values):
        raise ValueError("eos_ids 非法或越界")
    return set(values)


def check_loading_info(info):
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(key):
            raise ValueError(f"模型加载不完整：{key}={info[key]}")


def load_reference(model_dir, device="cuda:0", dtype_name="bf16"):
    if dtype_name not in {"bf16", "fp32"}:
        raise ValueError("dtype_name 仅支持 bf16/fp32")
    manifest = verify_local_snapshot(model_dir)
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
        dtype={"bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name],
        attn_implementation="eager", output_loading_info=True)
    check_loading_info(loading)  # Never accept randomly initialized missing weights.
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False)
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = normalize_eos(eos, model.config.text_config.vocab_size)
    return model.eval().to(device), tokenizer, manifest, eos_ids


def validate_ids(ids, vocab_size=None):
    if (not isinstance(ids, torch.Tensor) or ids.dtype != torch.long
            or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0):
        raise ValueError("input_ids 必须是非空 [1,T] 的 torch.long 张量")
    if ids.min().item() < 0 or (vocab_size is not None and ids.max().item() >= vocab_size):
        raise ValueError("token ID 越界")


def forward_last(model, ids, *, processed_tokens=0, cache=None, use_cache=True):
    """Return [1,V] last logits and updated state; no hidden per-request state here."""
    vocab = model.config.text_config.vocab_size
    validate_ids(ids, vocab)
    if type(processed_tokens) is not int or processed_tokens < 0 or type(use_cache) is not bool:
        raise ValueError("processed_tokens/use_cache 非法")
    if not use_cache and (cache is not None or processed_tokens != 0):
        raise ValueError("完整重算必须从位置0开始且不携带 cache")
    if use_cache:
        if cache is None and processed_tokens != 0:
            raise ValueError("Decode 丢失历史 cache")
        if cache is not None and cache.get_seq_length() != processed_tokens:
            raise ValueError("cache 长度与 processed_tokens 不一致")
    for name in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        token = getattr(model.config, name, None)
        if token is not None and (ids == token).any().item():
            raise ValueError("本课仅支持纯文本，不接受图像/视频占位 token")
    length = ids.shape[1]
    # Transformers 5.8.0: text position axis + three equal RoPE axes for pure text.
    positions = torch.arange(processed_tokens, processed_tokens + length, device=ids.device)
    positions = positions.view(1, 1, length).expand(4, 1, length)
    mask = torch.ones((1, processed_tokens + length), dtype=torch.long, device=ids.device)
    with torch.inference_mode():
        out = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                    past_key_values=cache, use_cache=use_cache,
                    logits_to_keep=1, return_dict=True)
        if tuple(out.logits.shape) != (1, 1, vocab):
            raise ValueError("模型返回的 logits shape 不符合 [1,1,V]")
        logits = out.logits[:, -1, :]
        if not torch.isfinite(logits).all().item():
            raise ValueError("logits 包含 NaN/Inf")
    updated = out.past_key_values
    if use_cache and (updated is None or updated.get_seq_length() != processed_tokens + length):
        raise ValueError("模型没有返回正确增长的 cache")
    return logits, updated
