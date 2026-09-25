# Week 3 环境与第一课验证记录

日期：2026-09-25。今天只进行 CPU 配置/分词检查，不加载模型权重、不启动 GPU、不安装或升级依赖。

- 主目录：/home/ubuntu/infer-engine。
- Python：/home/ubuntu/enter/envs/nanovllm/bin/python。
- 模型与 tokenizer：/home/ubuntu/huggingface/Qwen3-0.6B。
- 上次环境核查：Python 3.11.15、PyTorch 2.5.1+cu124、Transformers 5.8.0。
- 今日验证：AutoTokenizer 使用 local_files_only=True、trust_remote_code=False 正常加载。
- 网络/GPU边界：执行时设置 HF_HUB_OFFLINE=1、CUDA_VISIBLE_DEVICES=''。
- nano-vLLM 源码路径及未解决的导入定位见 PLAN.md；今天的脚本不依赖 nanovllm 导入，也未声称解决完整引擎兼容性。

## 今日实测输出

输入：用一句话解释 KV Cache。

- 原始文本 token 数：6。
- 单次应用 chat template 后输入 token 数：18。
- 模板设置：add_generation_prompt=True、enable_thinking=False。
- 模板直接编码与先格式化再 encode(add_special_tokens=False) 一致。
- 完整 token 序列 decode（保留特殊标记、不清理空白）与模板文本一致。
- EOS：<|im_end|>，ID 151645。
- tokenizer 基础词表：151643；含新增 token：151669；模型配置词表维度：151936。
- 当前版本 apply_chat_template(tokenize=True) 返回 BatchEncoding；脚本兼容其 input_ids 字段及旧版列表返回值。
- 28 层、16 个 Q 头、8 个 KV 头、head_dim=128；BF16/FP16、单请求、4096 位置的理论 KV 为 448 MiB，未实际分配。

## 验证命令

```bash
cd /home/ubuntu/infer-engine
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTHONPATH=week1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m unittest week1.test_profile_attention week2.verify_kv_cache week2.test_benchmark_kv_cache week3.test_inspect_model -v
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 /home/ubuntu/enter/envs/nanovllm/bin/python -B week3/inspect_model.py
```

20 项 CPU 测试通过（既有 16 项 + 本课新增 4 项）；默认观察脚本运行成功。Tokenizer 测试绑定当前本地模型文件与固定提示词，换模型/模板后需要重新确认期望值。

尚未验证：权重加载、实际 logits 形状、真实生成、缓存增量 logits 对拍、nano-vLLM 推理与 GPU 性能。不能把本记录当作模型已跑通。
