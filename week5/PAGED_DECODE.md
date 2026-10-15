# 分页 Decode 首版：2026-10-14

## 实现范围

显式 `Qwen35LLM(..., decode_backend='flash_paged', block_size=256)` 开启；默认 `dense` 不变。只改变 Full Attention 单 token Decode。Prefill、GDN、调度策略和采样保持原路径；逐请求循环仍存在，不声称完成批量化或手写kernel。

- gpu_engine.py：新增后端选择，提前拒绝未知值和非256倍数页大小。
- pool_backend.py：Decode 时跳过历史地址生成和index_select；块表/有效长度每批上传一次、跨层复用；提供当前层池和请求元数据。新路径不再执行旧的FullState写回。
- layer_compute.py：新增PagedDecode契约；沿用Q/gate拆分、Q/K Norm与partial RoPE；先index_copy_写新KV，再调用已安装flash_attn_with_kvcache；保留输出gate/o_proj。
- 外部已写入，因此不传新k/v、rotary参数；cache_seqlens含本轮token。写读均在当前stream，不额外添加全局同步。原Runner的完成同步与失败回收保持不变。

## 已执行验证

GPU6 NVIDIA L40，运行前291MiB/0%利用率；Torch2.5.1+cu124、FlashAttention2.8.3。没有安装依赖或性能计时。

历史实验入口为`nanovllm_qwen35/verify_paged_decode.py`（2026-10-15按用户要求已删除）。固定seed42，使用真实4B第一Full层参数和合成hidden/历史，块表[5,2]、页大小256；比较当前稠密参考，逐项检查整个KV池仅目标槽改变。

| past | 整个Full Attention模块输出相对L2 | 最大绝对误差 |
|---|---:|---:|
| 0 | 0 | 0 |
| 1 | 0.00309045 | 0.00390625 |
| 254 | 0.00347432 | 0.0009765625 |
| 255 | 0.00389362 | 0.00390625 |
| 256 | 0.00290692 | 0.0009765625 |
| 257 | 0.00284595 | 0.0009765625 |

真实模型同一实例顺序执行dense/flash_paged两条轨迹，prompt长度256和3，每请求生成3个token，ignore_eos=True。每轮两请求token均相同；结束状态槽和块均回收。

| 步 | logits相对L2 | 最大绝对误差 | token一致 |
|---|---:|---:|---|
| Prefill | 0 | 0 | 是 |
| Decode1 | 0.00804155 | 0.1484375 | 是 |
| Decode2 | 0.00629404 | 0.125 | 是 |

脚本预先采用有限值、相对L2<2%、模型token一致作为探索性冒烟门槛，不是经任务质量验证的容差，更不是原HF精确对拍门槛。没有逐层采集整模型state误差，也没有长生成质量验证。不得由上述结果宣称无损或完整数值验收；两路径均可能有浮点误差，不以dense作为绝对真值。

## 留存与复测边界

2026-10-15按用户要求清理本轮测试脚本和本地补丁/源码备份，仅保留生产实现与本文指标。原测试为Git未跟踪文件，不能用Git恢复；如需复测须重新建立验证入口。未保存权重副本或大张量快照，没有自动commit。

## 下一步

先由学生解释PagedDecode如何把“当前层”和“当前请求”连接起来，再补真实层Q/K/V的独立高精度参考，定位不同Attention数值路径的差异；补长轨迹、换序/取消/执行后失败回收验收。之后才做无profiler基准和独立profiling，判断历史gather消除对实际TPOT的贡献。一次只改变一个后端，不同时引入Graph/混批/GDN融合。
