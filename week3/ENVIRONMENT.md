# Week 3 环境与证据边界

更新：2026-09-28。当前目标 Qwen3.5-4B；原 Qwen3-0.6B 记录已归档到 [历史记录](../archive/old-models/QWEN3_06B.md)。

## 工作区

- 主目录：/home/ubuntu/infer-engine。
- 当前学习工作区：主目录；原隔离工作区整合后移除。
- 当前分支：main；Qwen3.5 第一课已合并，旧资料与开发记录归档到 archive/。
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

2026-09-25 仅运行配置实验；2026-09-28 经用户授权，从 ModelScope 下载模型。没有安装/升级/降级依赖；随后已完成GPU5上的两次forward及三条短提示词生成，详见 [生成报告](LESSON_02_GENERATION_RESULTS.md)。

## 2026-09-28 模型资产与 tokenizer 实测

- 来源：https://modelscope.cn/models/Qwen/Qwen3.5-4B；Hugging Face DNS 超时后，经用户明确同意切换来源。
- Git revision：ed182e32090db791077e12e0f58d22f3daafa173（ModelScope commit，不声称与 HF SHA 相同）。
- 本地目录：/home/ubuntu/huggingface/Qwen3.5-4B，detached HEAD 固定版本。
- 两个 safetensors 分片约9.3 GB；git lfs fsck 输出 Git LFS fsck OK。
- 真实 tokenizer 集成测试显式执行，1项通过；不是默认SKIP。
- 提示词“用一句话解释 KV Cache”：原文5个token，模板输入17个token；tokenizer EOS为248046。生成停止配置仍应按实际模型配置读取。
- 模板两种编码一致、完整解码往返一致；thinking关闭改变模板，不代表模型生成行为已验证。
- 通用 model_assets.py 和 infer_engine_manifest.json 尚未实现；当前Git/LFS下载校验不冒充已完成原计划Task 3全部接口。
- 本次生成配置EOS为248044，与tokenizer消息结束248046不同，停止判断优先使用model.generation_config。
- 完整循环与结果见 [Lesson 02](LESSON_02.md) 和 [生成报告](LESSON_02_GENERATION_RESULTS.md)。

## 当前验证结果

- 隔离工作区初始回归：20项通过。
- Task 1 回归：30项通过。
- Task 2 CPU 回归：发现40项，38项通过，2项真实 tokenizer 集成测试默认跳过。
- 另外显式运行旧 Qwen3 tokenizer 集成回归：1项通过；这不是 Qwen3.5 集成成功。
- 子进程测试确认 config-only 不导入 torch 或 transformers。
- fixture 推导结果：8层 Full Attention、24层 Linear Attention，B=1/T=4096/BF16 时 Full Attention KV 为128 MiB。
- linear state 字节数标记为未测，不把配置 dtype 当成运行时实测。

## 复现

在主目录运行：

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTHONPATH=week1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m unittest week1.test_profile_attention week2.verify_kv_cache week2.test_benchmark_kv_cache week3.test_inspect_model week3.test_tokenizer_integration -v

RUN_TOKENIZER_INTEGRATION=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m unittest week3.test_tokenizer_integration.TokenizerIntegrationTest.test_qwen3_history -v

CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m week3.inspect_model --config-only --config-file week3/fixtures/qwen35_4b_config_minimal.json
```

## 尚未验证

已完成权重加载、forward冒烟和三条短提示词的手写循环生成；混合缓存完整生命周期、logits对拍、GPU性能及nano-vLLM原生适配仍未验收。
两个工具级后续小项：输出完整模板源码/完整解码诊断，以及用模拟 loader 覆盖普通 CLI 的离线加载参数。
学习进度与工程工具进度分别见 [计划](PLAN.md)。
