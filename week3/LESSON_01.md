# Lesson 01：先读懂 Qwen3.5，再把文本变成输入

本课分两段：**1A 配置实验现在做；1B 真实 tokenizer 实验待资产准备后做。**
使用 Qwen3.5-4B，暂不加载模型，不使用 GPU。

## 1. 把第二周的知识接过来

第二周学的 KV Cache 没有失效，只是现在不能给每一层都套同一种缓存公式。

| 层类型 | 教学配置层数 | 要保存的历史信息 |
| --- | --- | --- |
| Full Attention | 8 | 历史 token 的 K/V，随上下文长度增长 |
| Linear Attention（Gated DeltaNet） | 24 | recurrent/conv state，不能套普通 KV 公式 |

固定大小的状态仍会随输入更新内容，不等于“不记历史”。本课先识别两类路径，不要求现在推导 DeltaNet 数学。

## 2. 先预测，再检查配置

先想：如果把 32 层都按普通 Attention 计算，会高估哪一部分？

```bash
cd /home/ubuntu/infer-engine/.worktrees/qwen35-week3
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m week3.inspect_model --config-only --config-file week3/fixtures/qwen35_4b_config_minimal.json
```

这里是教学 fixture，输出明确标记 `is_teaching_fixture=true`，不是模型已安装的证据。
fixture 的来源链接见文件中的 `_source`；它不替代后续固定 revision 的模型资产清单。

关键字段：

- 外层 `model_type=qwen3_5`，文本配置在 `text_config`，其类型为 `qwen3_5_text`。
- `hidden_size=2560`，总层数 32。
- Q 头数 16，KV 头数 4，显式 `head_dim=256`。
- Full Attention 原始层索引为 3、7、11、15、19、23、27、31（从零开始）。

## 3. token ID 不是 hidden state

假设输入 L 个 token：

```text
input_ids       [1, L]          整数编号
embedding 输出  [1, L, 2560]    每个编号对应一个向量
单 token decode [1, 1] → [1, 1, 2560]
```

这些是配置推导，不是已运行真实 forward 得到的张量。
不要再漏掉序列维度：L 个 token 就有 L 个向量。

这里不能用 hidden_size / Q头数替代显式 head_dim。
Q 的注意力特征宽度为 16 × 256 = 4096，输出投影再回到 2560。
配置还包含 attention output gate；4096 只指 Q 特征，不是包含 gate 在内的整个打包投影宽度。

## 4. KV 公式要限定统计范围

取 batch=1、历史长度4096、BF16每元素2字节：

```text
Full Attention KV bytes
= 2 × Full层数 × B × T × KV头数 × head_dim × 元素字节
= 2 × 8 × 1 × 4096 × 4 × 256 × 2
= 134217728 bytes
= 128 MiB
```

这里没有统计 linear state、模型权重、临时张量、内存分配器预留或其他 CUDA 开销。
所以 128 MiB 不是进程总显存，也不是全部混合缓存的实测值。

`linear_state_bytes=null` 表示**尚未加载模型并测量状态张量**，不代表零。
配置声明的 state dtype 也不能冒充运行时测量；这一点留到 Lesson 3 验证。

## 5. 现在完成的练习（Lesson 1A）

先不看参考脚本实现，回答：

1. 输入 IDs 为 [1,12]，embedding 输出形状是什么？只输入一个新 token 呢？
2. 在其余条件不变时，把 T 从4096减为2048，Full Attention KV 是多少 MiB？进程总显存也必然减半吗？
3. 为什么 linear_state_bytes=null 不能解释成“这24层没有缓存”？
4. 原始模型第3层是第一个 Full Attention 层；为什么不能直接把原始层编号当作紧凑的0–7号 KV 索引？

动手：读取 fixture，自己用循环分类 `layer_types`，输出两类层的原始索引，再与检查脚本对照。
先交第1–3题答案；确认理解后再进入1B，而不是直接加载权重。

## 6. 下一段：真实文本到 IDs（Lesson 1B，尚未执行）

准备好真实 tokenizer 资产后才运行：

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m week3.inspect_model --model /home/ubuntu/huggingface/Qwen3.5-4B --prompt '用一句话解释 KV Cache' --thinking off
```

目录不存在时明确报错，不自动下载、不偷偷换回 Qwen3-0.6B。
实际目录与固定版本由资产准备课确认。

学习链路为：原始消息 → 一次 chat template → token IDs → 下一课的 embedding/forward。
观察原文与模板文本的区别，并核对：

- 模板文本 encode(add_special_tokens=False) 与直接 apply_chat_template(tokenize=True) 的 IDs 相同。
- 完整解码能还原模板文本；模板只套一次。
- 新模型的 EOS、token 数要实际读取，不能照搬旧课的18个token或151645。
- thinking 开关能改变模板输出，只证明输入格式开关有效，不证明模型生成行为已验证。
- 输入里出现结束标记不等于立刻终止生成；生成循环的停止规则针对新选出的 token。

## 7. 实验与验收不混淆

当前模拟 tokenizer 单测检验工具逻辑，不证明真实 Qwen3.5 tokenizer 已可用。
资产准备完后，显式运行：

```bash
RUN_TOKENIZER_INTEGRATION=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m unittest week3.test_tokenizer_integration.TokenizerIntegrationTest.test_qwen35_target -v
```

默认 SKIP 不算通过；显式启用后缺少资产应失败。
当前工具展示模板哈希和渲染后的输入；模板源码及完整解码文本的持久化报告留作后续诊断增强，不宣称已有完整模板证据报告。

## 8. 然后才进入 Lesson 2

Lesson 2 使用模型 forward，但你自己控制首 token、decode、EOS 和长度上限，不用现成 generate() 代替学习。
本课不产生吞吐、TTFT、TPOT 等性能结论；能看配置不代表原生引擎适配成功。
