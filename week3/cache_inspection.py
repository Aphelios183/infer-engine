"""Read-only metadata for Full Attention KV and Linear Attention states.

No cloning, casting, CUDA synchronization or retained tensor references.
Diagnostic clones belong to the experiment, not to this observer.
"""
import torch

_FIELDS = {"full_attention": ("keys", "values"),
           "linear_attention": ("conv_states", "recurrent_states")}


def snapshot_cache(cache, layer_types, processed_tokens):
    """Return JSON metadata. Unknown totals are None, never fabricated zero."""
    if type(processed_tokens) is not int or processed_tokens < 0:
        raise ValueError("processed_tokens 必须是非负整数")
    if not isinstance(layer_types, list) or not layer_types or any(
            kind not in _FIELDS for kind in layer_types):
        raise ValueError("layer_types 必须是非空的合法层类型列表")
    if cache is None:
        layers = [None] * len(layer_types)
    else:
        layers = getattr(cache, "layers", None)
        if not isinstance(layers, (list, tuple)) or len(layers) != len(layer_types):
            raise ValueError("cache.layers 接口缺失或层数不匹配")

    seen = {}
    logical_bytes = 0
    storage_bytes = 0
    complete = True
    records = []
    for index, (kind, layer) in enumerate(zip(layer_types, layers)):
        entry = {"layer_index": index, "layer_type": kind,
                 "tensors": {}, "complete": True}
        values = {}
        for name in _FIELDS[kind]:
            tensor = getattr(layer, name, None)
            values[name] = tensor
            if tensor is None:
                entry["tensors"][name] = None
                entry["complete"] = False
                complete = False
                continue
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"第{index}层{name}不是Tensor或None")
            expected_rank = 3 if name == "conv_states" else 4
            if tensor.ndim != expected_rank or tensor.device.type == "meta" or tensor.layout != torch.strided:
                raise ValueError(f"第{index}层{name}的维度/设备/布局不支持")
            if kind == "full_attention" and tensor.shape[-2] != processed_tokens:
                raise ValueError(f"第{index}层{name}历史长度与processed_tokens不一致")
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen[key] = len(seen)
                storage_bytes += storage.nbytes()
            size = tensor.numel() * tensor.element_size()
            logical_bytes += size
            entry["tensors"][name] = {
                "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "device": str(tensor.device), "logical_bytes": size,
                "storage_bytes": storage.nbytes(), "storage_index": seen[key]}
        if kind == "full_attention":
            k, v = values["keys"], values["values"]
            if k is not None and v is not None and k.shape != v.shape:
                raise ValueError(f"第{index}层K/V形状不匹配")
            entry["effective_length"] = None if k is None else k.shape[-2]
        records.append(entry)

    return {"processed_tokens": processed_tokens, "layers": records, "complete": complete,
            "logical_tensor_bytes": logical_bytes if complete else None,
            "unique_storage_bytes": storage_bytes if complete else None,
            "observed_logical_bytes": logical_bytes,
            "observed_unique_storage_bytes": storage_bytes,
            "scope": "cache tensors only; excludes weights, allocator and diagnostic clones"}
