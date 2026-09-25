# Week 3 环境与证据边界

更新：2026-09-25。当前目标 Qwen3.5-4B；原 Qwen3-0.6B 记录已归档到 [历史记录](HISTORICAL_QWEN3_06B.md)。

## 工作区

- 主目录：/home/ubuntu/infer-engine。
- 课程迁移工作区：/home/ubuntu/infer-engine/.worktrees/qwen35-week3。
- 分支：learn/qwen35-week3；主目录尚未合并课程改动。
- 创建工作区时，原有7个修改/未跟踪学习文件按原内容复制并逐个比较，在隔离分支保存基线；没有覆盖主目录的学习内容。

## 已核查运行环境

| 项目 | 值 |
| --- | --- |
| Python | /home/ubuntu/enter/envs/nanovllm/bin/python，3.11.15 |
| torch | 2.5.1+cu124 |
| transformers | 5.8.0 |
| flash_attn | 2.8.3 |
| fla / causal_conv1d | 当前未安装 |
| GPU | 八张 NVIDIA L40，每张 nvidia-smi 显示46068 MiB |

本次没有安装/升级/降级依赖，没有下载模型，没有 GPU 实验。
Qwen3.5-4B 的目标目录尚未作为可用资产验证；教学 fixture 不是完整 config/权重/tokenizer，也没有代替模型 revision 锁定。

## 当前验证结果

- 隔离工作区初始回归：20项通过。
- Task 1 回归：30项通过。
- Task 2 CPU 回归：发现40项，38项通过，2项真实 tokenizer 集成测试默认跳过。
- 另外显式运行旧 Qwen3 tokenizer 集成回归：1项通过；这不是 Qwen3.5 集成成功。
- 子进程测试确认 config-only 不导入 torch 或 transformers。
- fixture 推导结果：8层 Full Attention、24层 Linear Attention，B=1/T=4096/BF16 时 Full Attention KV 为128 MiB。
- linear state 字节数标记为未测，不把配置 dtype 当成运行时实测。

## 复现

在上述隔离工作区运行：

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTHONPATH=week1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m unittest week1.test_profile_attention week2.verify_kv_cache week2.test_benchmark_kv_cache week3.test_inspect_model week3.test_tokenizer_integration -v

RUN_TOKENIZER_INTEGRATION=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m unittest week3.test_tokenizer_integration.TokenizerIntegrationTest.test_qwen3_history -v

CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m week3.inspect_model --config-only --config-file week3/fixtures/qwen35_4b_config_minimal.json
```

## 尚未验证

真实 Qwen3.5 tokenizer、权重加载、forward、生成、混合缓存生命周期、logits 对拍、GPU 性能、nano-vLLM 原生适配均不能标为完成。
两个工具级后续小项：输出完整模板源码/完整解码诊断，以及用模拟 loader 覆盖普通 CLI 的离线加载参数。
学习进度与工程工具进度分别见 [计划](PLAN.md)。
